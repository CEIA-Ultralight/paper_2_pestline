"""Fase 2 finita, somente train/val; CLI: --out --source <out>/source --parent.

Somente biblioteca padrão. O launcher externo fornece Slurm h100n2, Apptainer,
uma GPU e no máximo 8h. Nunca submete jobs, modifica a fase 1 ou lê o test.
Um prazo UTC persistido de 8h inclui reinícios/downtime (conservador). Cada
estágio também desconta suas tentativas anteriores do orçamento original.
SIGUSR1 é encaminhado ao filho; pausa retorna 75. Aos 8h o filho é encerrado
se não tiver concluído o checkpoint/export durante a graça de 120 segundos.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading

from . import campaign as c


COMBINATIONS = ('parts_supcon', 'hires_supcon', 'hires_parts')
COMPONENTS = {'parts_supcon': ('parts', 'supcon'),
              'hires_supcon': ('hires', 'supcon'), 'hires_parts': ('hires', 'parts')}
ORDER = c.VARIANTS + COMBINATIONS
HOURS = 8
GRACE_SECONDS = 120
PROTOCOL = {'img_size': 384, 'batch': 32, 'seed': 42, 'lr': 8e-5,
            'epochs': 24, 'patience': 5, 'workers': 8, 'no_pretrained': False,
            'limit_train': None, 'limit_val': None}
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'PREEMPTED',
            'NODE_FAIL', 'OUT_OF_MEMORY', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def score(summary):
    value = summary['exported_metrics']['species_macro_f1']
    c.require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
              'macro-F1 inválido')
    return value


def priority(summaries):
    """Heurística max(componentes), NÃO evidência de interação entre componentes."""
    baseline = score(summaries['baseline'])
    estimates = {v: max((score(summaries[x]) for x in COMPONENTS[v] if x in summaries),
                       default=baseline) for v in COMBINATIONS}
    return sorted(COMBINATIONS, key=lambda v: -estimates[v]), estimates


def select_winner(summaries):
    c.require('baseline' in summaries, 'Baseline validado obrigatório')
    return max((v for v in ORDER if v in summaries), key=lambda v: score(summaries[v]))


def conclusion(seed42, seed84, winner):
    if not winner or 'baseline' not in seed42 or winner not in seed42:
        return 'inconclusivo: seleção/baseline seed42 ausente'
    if 'baseline' not in seed84 or winner not in seed84:
        return 'inconclusivo: comparação pareada seed84 incompleta'
    gain42 = score(seed42[winner]) - score(seed42['baseline'])
    gain84 = score(seed84[winner]) - score(seed84['baseline'])
    if (score(seed42[winner]) >= score(seed42['baseline']) + .005
            and score(seed84[winner]) >= score(seed84['baseline']) + .005):
        return 'ganho exploratório consistente nas duas sementes testadas'
    if gain42 > 0 or gain84 > 0:
        return 'melhoria provisória: ganho não consistente de pelo menos 0,005 nas duas sementes'
    return 'nenhum ganho observado nas comparações válidas; busca finita, não exaustiva'


def source_extras(source):
    """campaign.source_identity já inclui este módulo e seus testes."""
    path = Path(source) / 'fly-det/slurm/followup.sh'
    return {str(path.relative_to(source)): hashlib.sha256(c.regular(path).read_bytes()).hexdigest()} if path.exists() else {}


def trainer_hashes(identity):
    return {name: identity['sha256'][f'fly-det/scripts/architecture_lab/{name}']
            for name in ('train.py', 'models.py')}


def validate_protocol(output, variant, seed=42, manifests=None, hashes=None):
    """Current final export, not the possibly stale stage.result/extension result."""
    output = Path(output)
    summary = c.training_artifacts(output, 'crop')
    metadata = summary['metadata']
    args = metadata['args']
    expected = {**PROTOCOL, 'seed': seed, 'variant': variant}
    for key, value in expected.items():
        c.require(key in args and args[key] == value and
                  (value is not False or args[key] is False), f'Protocolo incompatível: {key}')
    mapping = {name: i for i, name in enumerate(c.CLASSES)}
    c.require(metadata.get('class_order') == list(c.CLASSES) and
              metadata.get('class_to_idx') == mapping and metadata.get('smoke') is False,
              'Classes/smoke incompatíveis')
    c.require(metadata.get('dataset_root') == str(c.CROPS) and args.get('data') == str(c.CROPS),
              'População de crops incompatível')
    actual = {}
    for split in ('train', 'val'):
        manifest = c.read_json(output / f'{split}_manifest.json')
        sha = manifest.get('sha256', '')
        entries = manifest.get('entries')
        c.require(isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha), 'SHA de manifesto inválido')
        c.require(manifest.get('class_order') == list(c.CLASSES) and manifest.get('class_to_idx') == mapping,
                  'Mapping do manifesto incompatível')
        c.require(manifest.get('selected_count') == manifest.get('available_count') and
                  isinstance(entries, list) and len(entries) == manifest['selected_count'] and
                  c.digest(entries) == sha, 'Manifesto limitado ou SHA inconsistente')
        c.require(all(isinstance(e, list) and len(e) == 3 and isinstance(e[0], str)
                      and Path(e[0]).parts[0] == split and '..' not in Path(e[0]).parts
                      and not Path(e[0]).is_absolute() and type(e[1]) is int and 0 <= e[1] < 7
                      and c.positive(e[2]) for e in entries), 'Entradas do manifesto inválidas')
        actual[split] = sha
    c.require(manifests is None or actual == manifests, 'SHA train/val diferente do baseline pai')
    if hashes is not None:
        c.require(metadata.get('source', {}).get('file_sha256') == hashes, 'Hash do treinador incompatível')
    return summary, actual


def job_states(ids):
    """Read-only accounting query; unknown/nonterminal fails closed, no sbatch."""
    c.require(ids and all(re.fullmatch(r'[0-9]+', x) for x in ids), 'Job pai não identificável')
    receipt = os.environ.get('FLYDET_PARENT_ACCOUNTING')
    if receipt:
        # The host launcher captures sacct AFTER Slurm's afterany dependency.
        # The SIF need not contain cluster-specific sacct libraries/configuration.
        path = Path(receipt).resolve()
        c.require(path.is_relative_to(c.RAID), 'Recibo Slurm fora do RAID')
        text = c.regular(path).read_text()
    else:
        result = subprocess.run(['sacct', '-n', '-X', '-j', ','.join(sorted(ids)),
                                 '--format=JobIDRaw,State', '--parsable2'],
                                check=True, capture_output=True, text=True, timeout=20)
        text = result.stdout
    states = {}
    for line in text.splitlines():
        fields = line.strip().split('|')
        if len(fields) >= 2 and fields[0] in ids:
            states[fields[0]] = fields[1].split()[0].rstrip('+')
    c.require(set(states) == set(ids) and all(s in TERMINAL for s in states.values()),
              'Fase 1 ainda ativa ou estado Slurm desconhecido; aguardar término')
    return states


def load_parent(parent):
    """Read phase 1 only. A dead controller may leave status=running: accounting
    must prove job termination, and every accepted export is checked separately.
    A nonblocking shared lock prevents reading an actively controlled campaign.
    """
    parent = Path(parent)
    try:
        state = c.read_json(c.regular(parent / 'campaign_state.json'))
    except (OSError, ValueError, TypeError) as exc:
        return {'root': str(parent), 'status': 'desconhecido', 'partial': True,
                'valid': {}, 'excluded': {}, 'coverage': {}, 'roi': {},
                'blocked': f'Baseline pai não validável; estado pai ausente/inválido: {exc}'}
    result = {'root': str(parent), 'status': state.get('status'),
              'partial': state.get('status') != 'completed', 'valid': {}, 'excluded': {},
              'coverage': {v: state.get('stages', {}).get('crop_' + v, {}).get('status', 'não executado')
                           for v in c.VARIANTS},
              'roi': {k: r.get('status') for k, r in state.get('stages', {}).items() if 'roi' in k},
              'blocked': None}
    try:
        # Never create or write the parent's lock, sources, state or artifacts.
        with (parent / '.controller.lock').open('r') as lock:
            try:
                c.fcntl.flock(lock, c.fcntl.LOCK_SH | c.fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError('Controlador pai ainda ativo') from exc
            try:
                c.require(c.read_json(parent / 'campaign_state.json') == state,
                          'Estado pai mudou durante leitura; tentar novamente após término')
                return _load_parent_locked(parent, state, result)
            finally:
                c.fcntl.flock(lock, c.fcntl.LOCK_UN)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        result['blocked'] = str(exc)
        return result


def _load_parent_locked(parent, state, result):
    identity = state['source']
    c.require(Path(identity['root']).resolve() == parent / 'source', 'Snapshot pai fora da campanha')
    actual_identity = c.source_identity(parent / 'source')
    c.require(actual_identity['sha256'] == identity['sha256'], 'Snapshot pai alterado')
    result['source'] = identity
    ids = set()
    # Inspect only known training outputs, never arbitrary paths from state JSON.
    for variant in c.VARIANTS:
        path = parent / 'crops' / variant / 'summary.json'
        if path.is_file():
            try:
                job = c.read_json(c.regular(path)).get('metadata', {}).get('environment', {}).get('slurm_job_id')
            except (OSError, ValueError, AttributeError, TypeError):
                continue  # A broken nonbaseline export is excluded individually below.
            if job:
                ids.add(str(job))
    result['jobs'] = job_states(ids)
    for variant in c.VARIANTS:
        try:
            output = parent / 'crops' / variant
            record = state.get('stages', {}).get('crop_' + variant, {})
            c.require(record.get('kind') == 'crop' and Path(record.get('out', '')).resolve() == output
                      and output.resolve() == output, 'Estágio pai ausente ou saída incompatível')
            summary, manifests = validate_protocol(output, variant,
                manifests=result.get('manifests'), hashes=trainer_hashes(identity))
            c.require(str(summary['metadata'].get('environment', {}).get('slurm_job_id')) in result['jobs'],
                      'Export sem término Slurm comprovado')
            if variant == 'baseline':
                result['manifests'] = manifests
            c.require('manifests' in result, 'Baseline validado obrigatório')
            result['valid'][variant] = summary
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            result['excluded'][variant] = str(exc)
    if 'baseline' not in result['valid']:
        result['blocked'] = 'Baseline pai ausente/incompatível: nenhum treino autorizado pelo controlador'
    return result


def crop_stage(out, variant, seed, smoke=False, resume_smoke=False):
    out = Path(out)
    key = f'smoke_{variant}' if smoke else f'combo_{variant}' if seed == 42 else f'seed84_{variant}'
    output = out / ('smoke' if smoke else 'combos' if seed == 42 else 'seed84') / variant
    cmd = [sys.executable, '-m', 'architecture_lab.train', '--data', str(c.CROPS), '--out', str(output),
           '--variant', variant, '--epochs', '1' if smoke else '24', '--patience', '5',
           '--batch', '4' if smoke else '32', '--img-size', '384', '--workers', '0' if smoke else '8',
           '--lr', '8e-5', '--seed', str(seed), '--max-hours', '.1' if smoke else '1' if seed == 42 else '1.25']
    dependencies = ['tests']
    if smoke:
        cmd += ['--limit-train', '8', '--limit-val', '8', '--no-pretrained']
        if resume_smoke:
            dependencies.append(key)
            key += '_resume'
            cmd.append('--resume')
    else:
        cmd += ['--wandb', '--wandb-entity', 'pestline', '--wandb-project', 'fly-species',
                '--wandb-run-name', f'A2_{variant}_seed{seed}_{out.name}']
        if seed == 42:
            dependencies.append(f'smoke_{variant}_resume')
    return c.stage(key, 'crop', output, cmd, dependencies)


def build_stages(out, source, ordering):
    specs = [c.stage('tests', 'tests', out, [sys.executable, '-m', 'unittest', 'discover',
                   '-s', str(Path(source) / 'tests'), '-p', 'test_architecture_*.py'])]
    for variant in ordering:
        specs += [crop_stage(out, variant, 42, smoke=True),
                  crop_stage(out, variant, 42, smoke=True, resume_smoke=True)]
    return specs + [crop_stage(out, v, 42) for v in ordering]


def load_state(out, source, parent):
    path = Path(out) / 'followup_state.json'
    contract = {'source': c.source_identity(Path(source)), 'source_extras': source_extras(source),
                'parent_root': str(parent), 'hours': HOURS, 'protocol': PROTOCOL}
    if path.exists():
        state = c.read_json(path)
        c.require(state.get('schema_version') == 1 and all(state.get(k) == v for k, v in contract.items()),
                  'Snapshot/contrato mudou; use outra fase 2')
        c.require(isinstance(state.get('stages'), dict) and isinstance(state.get('commands'), dict),
                  'Estado da fase 2 inválido')
        return state
    return {**contract, 'schema_version': 1, 'created_at': c.now(), 'status': 'ready',
            'stages': {}, 'commands': {}, 'plan': [], 'parent': None, 'winner': None}


def elapsed_attempts(record):
    total = 0.
    for attempt in record.get('attempts', []):
        start = datetime.fromisoformat(attempt['started_at'])
        end = datetime.fromisoformat(attempt.get('finished_at', c.now()))
        total += max(0., (end - start).total_seconds())
    return total


class FollowupController(c.Controller):
    def persist(self):
        self.state['updated_at'] = c.now()
        c.write_json(self.out / 'followup_state.json', self.state)
        c.atomic_write(self.out / 'followup_report.md', render_report(self.out, self.state))

    def remaining(self):
        return (datetime.fromisoformat(self.state['deadline']) - datetime.now(timezone.utc)).total_seconds()

    def spawn(self, command, log):
        remaining = self.remaining()
        if remaining <= GRACE_SECONDS:
            self.handle_signal(signal.SIGUSR1, None)
            return 75
        pause = threading.Timer(remaining - GRACE_SECONDS, self.handle_signal, args=(signal.SIGUSR1, None))
        def terminate():
            self.handle_signal(signal.SIGUSR1, None)
            if self.child is not None:
                try:
                    self.child.kill()
                except ProcessLookupError:
                    pass
        stop = threading.Timer(remaining, terminate)
        for timer in (pause, stop):
            timer.daemon = True
            timer.start()
        try:
            return super().spawn(command, log)
        finally:
            pause.cancel()
            stop.cancel()

    def verify(self, spec, record):
        result = super().verify(spec, record)
        if spec['kind'] == 'crop':
            args = result['metadata']['args']
            cmd = spec['command']
            variant = cmd[cmd.index('--variant') + 1]
            seed = int(cmd[cmd.index('--seed') + 1])
            if spec['id'].startswith('smoke_'):
                expected = {**PROTOCOL, 'variant': variant, 'epochs': 1, 'batch': 4, 'workers': 0,
                            'limit_train': 8, 'limit_val': 8, 'no_pretrained': True}
                c.require(all(k in args and args[k] == v for k, v in expected.items()), 'Smoke fora do protocolo')
                c.require(result['metadata'].get('source', {}).get('file_sha256') == trainer_hashes(self.state['source']),
                          'Smoke de outro snapshot')
                if spec['id'].endswith('_resume'):
                    previous = self.state['stages'][spec['id'].removesuffix('_resume')]['result']
                    c.require(args.get('resume') is True and result['experiment_id'] == previous['experiment_id']
                              and result.get('epochs_ran_this_invocation') == 0,
                              'Smoke sem evidência de retomada do checkpoint de uma época')
            else:
                result, _ = validate_protocol(spec['out'], variant, seed,
                    self.state['parent']['manifests'], trainer_hashes(self.state['source']))
        return result

    def execute(self, spec):
        c.require(c.source_identity(self.source) == self.state['source'] and
                  source_extras(self.source) == self.state['source_extras'], 'Snapshot fase 2 alterado')
        output = Path(spec['out'])
        c.require(output.resolve() == output and output.is_relative_to(self.out) and
                  not output.is_relative_to(self.source), 'Saída fora da fase 2 ou symlink')
        old = self.state['commands'].setdefault(spec['id'], spec)
        c.require(old == spec, 'Comando/dependências mudaram na retomada')
        record = self.state['stages'].get(spec['id'], {})
        # Never adopt someone else's checkpoint, including a phase-1 symlink.
        if spec['kind'] == 'crop':
            checkpoint = output / 'last.pt'
            if checkpoint.exists() or checkpoint.is_symlink():
                try:
                    c.regular(checkpoint)
                    owner = (self.state['stages'].get(spec['id'].removesuffix('_resume'), {})
                             if spec['id'].endswith('_resume') else record)
                    c.require(owner.get('attempts'), 'Checkpoint sem tentativa própria rastreada')
                except (OSError, ValueError) as exc:
                    record = self.state['stages'].setdefault(spec['id'], {'attempts': [],
                        'kind': spec['kind'], 'out': spec['out'], 'log': f'logs/{spec["id"]}.log'})
                    record.update(status='failed', reason=str(exc))
                    log = self.out / record['log']
                    log.parent.mkdir(exist_ok=True)
                    with log.open('a', encoding='utf-8') as handle:
                        handle.write(f'[{c.now()}] Checkpoint rejeitado: {exc}\n')
                    self.persist()
                    return False
        self.persist()
        if record.get('status') != 'succeeded' and '--max-hours' in spec['command']:
            command = list(spec['command'])
            index = command.index('--max-hours') + 1
            remaining = float(command[index]) * 3600 - elapsed_attempts(record)
            if remaining <= 0:
                self.state['stages'].setdefault(spec['id'], {'attempts': [], 'kind': spec['kind'],
                    'out': spec['out'], 'log': f'logs/{spec["id"]}.log'}).update(
                        status='budget_exhausted', reason='Orçamento acumulado do estágio consumido')
                self.persist()
                return False
            command[index] = format(remaining / 3600, '.12g')
            spec = {**spec, 'command': command}
        return super().execute(spec)

    def full_results(self, seed):
        results = {}
        variants = COMBINATIONS if seed == 42 else c.VARIANTS + COMBINATIONS
        for variant in variants:
            key = f'combo_{variant}' if seed == 42 else f'seed84_{variant}'
            record = self.state['stages'].get(key, {})
            if record.get('status') == 'succeeded':
                try:
                    results[variant], _ = validate_protocol(record['out'], variant, seed,
                        self.state['parent']['manifests'], trainer_hashes(self.state['source']))
                except (OSError, ValueError, KeyError, TypeError):
                    continue
        return results

    def run(self):
        parent = load_parent(Path(self.state['parent_root']))
        if parent.get('blocked'):
            self.state['parent_observation'] = parent
            self.state['blocked_reason'] = parent['blocked']
            return self.finish('blocked_parent', 1)
        if self.state['parent'] is None:
            self.state['parent'] = parent
            ordering, estimates = priority(parent['valid'])
            self.state.update(ordering=ordering, priority_scores=estimates,
                              plan=build_stages(self.out, self.source, ordering))
        else:
            c.require(c.digest(parent) == c.digest(self.state['parent']), 'Artefatos/estado pai mudaram após decisão')
            c.require(self.state['plan'] == build_stages(self.out, self.source, self.state['ordering']),
                      'Plano mudou na retomada')
        self.state.pop('blocked_reason', None)
        self.state.pop('parent_observation', None)
        if 'deadline' not in self.state:
            self.state['deadline'] = (datetime.now(timezone.utc) + timedelta(hours=HOURS)).isoformat()
        self.state.update(status='running', exit_code=None)
        self.persist()
        if self.signal_number is not None:
            return self.finish('paused', 75)
        if self.remaining() <= GRACE_SECONDS:
            return self.finish('budget_exhausted', 1)
        # Once winner is frozen, no failed seed42 stage gets another chance to
        # change the selection population on restart. Diagnostics may still retry.
        for spec in self.state['plan']:
            if self.state['winner'] is not None and spec['id'] != 'tests':
                continue
            passed = self.execute(spec)
            if self.signal_number is not None:
                return self.finish('paused', 75)
            if spec['id'] == 'tests' and not passed:
                return self.finish('aborted_tests', 1)
        if self.state['winner'] is None:
            candidates = {**parent['valid'], **self.full_results(42)}
            winner = select_winner(candidates)
            self.state['winner'] = {'variant': winner, 'decided_at': c.now(),
                'scores': {v: score(s) for v, s in candidates.items()},
                'exports': {v: c.digest(s) for v, s in candidates.items()},
                'rule': 'max species_macro_f1; empate estável, baseline preferido; somente val seed42'}
            self.persist()  # Durable BEFORE seed84; no re-selection on restart.
        else:
            candidates = {**parent['valid'], **self.full_results(42)}
            c.require({v: c.digest(s) for v, s in candidates.items()} == self.state['winner']['exports'],
                      'Export usado na decisão foi alterado/invalidado; seleção congelada')
        winner = self.state['winner']['variant']
        for variant in dict.fromkeys(('baseline', winner)):
            self.execute(crop_stage(self.out, variant, 84))
            if self.signal_number is not None:
                return self.finish('paused', 75)
        for seed in (42, 84):
            for variant in self.full_results(seed):
                base = crop_stage(self.out, variant, seed)
                self.execute(c.stage('diagnostic_' + base['id'], 'diagnostic', base['out'],
                    [sys.executable, '-m', 'architecture_lab.diagnose', '--run', base['out'],
                     '--samples', '8', '--batch', '32'], ('tests', base['id'])))
                if self.signal_number is not None:
                    return self.finish('paused', 75)
        failed = any(r['status'] != 'succeeded' for r in self.state['stages'].values())
        return self.finish('completed_with_failures' if failed else 'completed', int(failed))

    def finish(self, status, code):
        if self.signal_number is not None:
            status, code = 'paused', 75
        self.state.update(status=status, exit_code=code, signal=self.signal_number)
        self.persist()
        print(f'[{c.now()}] Fase 2: {status}; retorno {code}; followup_report.md atualizado', flush=True)
        return code


def render_report(out, state):
    out = Path(out)
    parent = state.get('parent_observation') or state.get('parent') or {}
    records = state['stages']
    lines = ['# Fase 2 — acompanhamento exploratório de arquiteturas', '', f'**Estado:** {state["status"]}',
        f'Bloqueio: {state.get("blocked_reason", "nenhum")}.',
        f'Prazo UTC não renovável: {state.get("deadline", "não iniciado")}; máximo 8h incluindo retomadas.',
        'Fonte, hashes, comandos exatos e tentativas: [followup_state.json](followup_state.json).', '',
        'Somente train/val; nenhum acesso ao TEST. Crops GT pad75, sete classes; imagens originais intactas. '
        'RoI tem outra população e NÃO entra no ranking. Manifest SHA mede path/target/byte_size, não conteúdo.', '',
        '**Confundidor de custo:** originais da fase 1 têm 1,25h por chamada (eventual extensão +1,5h); '
        'combinações têm 1h, sementes 84 têm 1,25h. Tentativas/export/inicialização também custam tempo. '
        'Não são comparações de custo igual: não se pode atribuir causalmente diferenças só à arquitetura.', '',
        '## Protocolo e comandos planejados', '',
        '- CLI: --out obrigatório; --source obrigatório e exatamente <out>/source; --parent obrigatório.',
        '- Todos os 3 combos: parts_supcon, hires_supcon, hires_parts. '
        'Ordenação = max(macro-F1 dos componentes válidos), fallback baseline; heurística, não evidência sobre combos.',
        '- Full: --epochs 24 --patience 5 --batch 32 --img-size 384 --workers 8 --lr 8e-5 '
        '--seed 42 --max-hours 1; pretrained habilitado; sem limites train/val.',
        '- Smoke de cada combo: --epochs 1 --patience 5 --batch 4 --img-size 384 --workers 0 --lr 8e-5 '
        '--seed 42 --max-hours .1 --limit-train 8 --limit-val 8 --no-pretrained; depois mesma chamada --resume '
        '(outra .1h; mesma identidade, zero épocas adicionais, export positivo).',
        '- Seed84: baseline fresco e vencedor fresco; mesmo full, --seed 84 --max-hours 1.25. '
        'Se vencedor=baseline, uma única execução. Sem retomada de pesos da fase 1.',
        '- W&B full: --wandb --wandb-entity pestline --wandb-project fly-species '
        '--wandb-run-name A2_<variante>_seed<semente>_<out.name>.',
        '- Diagnósticos apenas full NOVOS válidos: architecture_lab.diagnose --run <output> --samples 8 --batch 32.',
        '- Testes antes dos treinos: unittest discover -s <source>/tests -p test_architecture_*.py; '
        'marcador antirrecursão herdado. Falha aborta; falhas de combos são independentes.',
        '- Checkpoint atômico a cada época pelo trainer; --resume automático apenas de tentativa própria '
        'no mesmo snapshot/contrato. Logs append; SIGUSR1 encaminhado; retorno 75, nenhuma submissão.', '',
        f'Ordem: {", ".join(state.get("ordering", [])) or "pendente"}.', '',
        '## Fase 1 — exports atuais validados', '',
        f'Estado pai: {parent.get("status", "desconhecido")}; pai parcial: {parent.get("partial", "desconhecido")}. '
        'Estado running residual só é aceito após comprovar término Slurm e validar cada export.',
        f'Jobs pai: {parent.get("jobs", {})}.', '',
        '| Original | Estado registrado | Export elegível / exclusão |', '|---|---|---|']
    valid = parent.get('valid', {})
    for variant in c.VARIANTS:
        reason = 'validado' if variant in valid else parent.get('excluded', {}).get(variant, 'não validado')
        lines.append(f'| {variant} | {parent.get("coverage", {}).get(variant, "não observado")} | {reason} |')
    tests_ok = records.get('tests', {}).get('status') == 'succeeded'
    lines += ['', 'Compatibilidade dos originais: código mudou para incluir combos; '
        'test_original_initialization_rng_and_combination_head_mappings verifica igualdade de inicialização/RNG '
        'com implementação original independente; test_architecture_combinations verifica roteamento/loss. '
        f'Suite no snapshot desta fase: {"aprovada" if tests_ok else "ainda NÃO aprovada"}. '
        'Isso não prova igualdade de trajetórias completas de treino.', '',
        '## Cobertura realmente executada — fase 2', '',
        '| Estágio | Estado | Tentativas | Tempo tentativas (s) | Log | Motivo |', '|---|---|---:|---:|---|---|']
    keys = [s['id'] for s in state.get('plan', [])]
    if not keys:
        keys = ['tests'] + [key for v in COMBINATIONS for key in
                            (f'smoke_{v}', f'smoke_{v}_resume', f'combo_{v}')]
    keys += [f'seed84_{v}' for v in dict.fromkeys(('baseline', (state.get('winner') or {}).get('variant', 'vencedor_pendente')))]
    keys += [k for k in records if k not in keys]
    for key in keys:
        record = records.get(key, {})
        reason = str(record.get('reason') or '—').replace('|', '/').replace('\n', ' ')
        log = c.link(out, out / record['log'], 'log append') if record.get('log') else '—'
        lines.append(f'| {key} | {record.get("status", "não executado")} | {len(record.get("attempts", []))} | '
                     f'{elapsed_attempts(record):.1f} | {log} | {reason} |')
    seed42, seed84 = dict(valid), {}
    groups = [('Fase 1 crops seed42', [(v, {'status': 'export validado',
        'out': str(Path(parent['root']) / 'crops' / v)}, s) for v, s in valid.items()])]
    for prefix, title, results in [('combo_', 'Fase 2 combos seed42', seed42),
                                   ('seed84_', 'Fase 2 verificações seed84', seed84)]:
        rows = []
        for key, record in records.items():
            if key.startswith(prefix) and record.get('status') == 'succeeded' and record.get('result'):
                variant = key.removeprefix(prefix)
                results[variant] = record['result']
                rows.append((variant, record, record['result']))
        groups.append((title, rows))
    for title, rows in groups:
        lines += ['', f'## {title}', '',
            '| Variante | Val geral | Val espécies acc | Val espécies macro-F1 | ΔF1 vs baseline pareado | '
            'Tempo treino acumulado (s) | Budget chamada (h) | Config |', '|---|---:|---:|---:|---:|---:|---:|---|']
        base = seed84.get('baseline') if 'seed84' in title else valid.get('baseline')
        for variant, record, summary in rows:
            m, args = summary['exported_metrics'], summary['metadata']['args']
            delta = f'{score(summary) - score(base):+.4f}' if base else 'não pareado'
            lines.append(f'| {variant} | {m["overall_accuracy"]:.4f} | {m["species_accuracy"]:.4f} | '
                f'{score(summary):.4f} | {delta} | {summary.get("elapsed_seconds", "—")} | '
                f'{summary.get("max_hours_per_invocation", args.get("max_hours", "—"))} | '
                f'img={args.get("img_size")}, batch={args.get("batch")}, seed={args.get("seed")}, '
                f'lr={args.get("lr")}, epochs={args.get("epochs")}, patience={args.get("patience")}, '
                f'workers={args.get("workers")}, no_pretrained={args.get("no_pretrained")} |')
        for variant, record, summary in rows:
            lines += c.report_training(out, variant, record, summary, 'crop')
            key = ('seed84_' if summary['metadata']['args']['seed'] == 84 else 'combo_') + variant
            diag = records.get('diagnostic_' + key, {}) if title != 'Fase 1 crops seed42' else {}
            if diag.get('status') == 'succeeded':
                lines += [c.link(out, Path(record['out']) / 'diagnostics/latency.json', 'Latência verificada'),
                          c.link(out, Path(record['out']) / 'diagnostics/diagnostics.json', 'Grad-CAM verificado')]
    winner = (state.get('winner') or {}).get('variant')
    lines += ['', '## RoI original — cobertura separada, métricas não comparadas', '',
        str(parent.get('roi', {})) or 'Não observado.', '', '## Conclusão', '',
        f'Vencedor seed42 congelado: {winner or "pendente"}.', conclusion(seed42, seed84, winner), '',
        '- Duas sementes não constituem prova estatística; seleção em val é exploratória e enviesada pela busca.',
        '- Holdout independente agrupado por placa/armadilha/local/período continua não resolvido; '
        'SHA igual não demonstra ausência de leakage.',
        '- Meta 0,95 em acurácia de espécies E mAP de detecção NÃO verificada: esta fase não mede mAP.',
        '- Latência é somente classificador FP32, batches 1/32, não API E2E; Grad-CAM qualitativo, não causal.',
        '- Nenhum resultado implica impossibilidade de melhoria ou teto físico máximo. '
        'A evidência cobre hipóteses e custos finitos; a busca não é exaustiva.', '',
        '## Propostas NÃO executadas', '',
        '- Auditar split agrupado e obter holdout independente; ampliar sementes e igualar orçamento total.',
        '- Investigar hard negatives e confusões entre espécies; alternativas fine-grained, calibração/OOD.',
        '- Avaliar novo detector e SAHI/tiling em resolução nativa com mAP apropriado; nenhuma geração '
        'ou submissão adicional automática.', '']
    return '\n'.join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--source', required=True, help='Snapshot imutável <out>/source')
    parser.add_argument('--parent', required=True, help='Raiz da campanha fase 1; somente leitura')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    # Validate non-overlap BEFORE creating cache/output directories in the guard.
    out, source, parent = (Path(x).resolve() for x in (args.out, args.source, args.parent))
    c.require(parent.is_relative_to(c.RAID) and not out.is_relative_to(parent)
              and not parent.is_relative_to(out), 'Fases 1/2 não podem se sobrepor')
    out, source = c.runtime_guard(args)
    c.require(Path(__file__).resolve() == source / 'fly-det/scripts/architecture_lab/followup.py',
              'Executar followup do snapshot, não do workspace')
    out.mkdir(parents=True, exist_ok=True)
    with c.campaign_lock(out):
        state = load_state(out, source, parent)
        controller = FollowupController(out, source, state)
        previous = {s: signal.signal(s, controller.handle_signal)
                    for s in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
        try:
            return controller.run()
        except Exception as exc:
            state['controller_error_type'] = type(exc).__name__
            return controller.finish('failed_controller', 1)
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)


if __name__ == '__main__':
    raise SystemExit(main())