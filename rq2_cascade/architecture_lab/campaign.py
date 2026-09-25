"""Sequential, validation-only campaign for architecture_night.sh snapshots.

Standard library only, including import/help and artifact checks. GPU children
are launched only after the Slurm/Apptainer guard. Never submits jobs, edits
sources, reads the dataset test split, or unpickles checkpoints. The launcher
owns the cross-campaign single-GPU lock; this module also locks its output tree.

Failed ordinary stages retry on the next invocation (trusted last.pt => resume).
The optional extension is ONE invocation, durably consumed BEFORE spawning:
even an interrupted/failed extension is not granted another 1.5h on restart.
Slurm's 12h allocation is the overall wall-time limit, not a renewable GPU budget
implemented here. Child budgets exclude some initialization/export costs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
from urllib.parse import quote


RAID = Path('/raid/user_marcospaulo')
CROPS = RAID / 'datasets/DS-F2_crops_pad75'
YOLO_DATA = RAID / 'datasets/DS-F2_v8.1.1'
DETECTOR = RAID / 'flydet_runs/E0_yolo26m_baseline/runs/yolo26m.pt/weights/best.pt'
VARIANTS = ('baseline', 'supcon', 'arcface', 'parts', 'hires')
CLASSES = ('INS', 'MAR', 'MC', 'MD', 'MF', 'MV', 'NOISE')
GOOD = ('completed', 'early_stop', 'budget')
SCHEMA = 1
TEST_MARKER = 'ARCHITECTURE_CAMPAIGN_TESTING'


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path, text):
    """fsync + same-directory replace; a failed write preserves the old state."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def read_json(path):
    with Path(path).open(encoding='utf-8') as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f'Objeto JSON esperado: {path.name}')
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    return type(value) is int and value > 0


def regular(path):
    path = Path(path)
    require(not path.is_symlink() and path.is_file() and path.stat().st_size > 0,
            f'Artefato ausente, vazio ou symlink: {path.name}')
    return path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, separators=(',', ':'),
                                     sort_keys=True).encode()).hexdigest()


def valid_metrics(metrics):
    require(isinstance(metrics, dict), 'Métricas ausentes')
    for key in ('overall_accuracy', 'species_accuracy', 'species_macro_recall',
                'species_macro_f1', 'macro_f1', 'loss'):
        value = metrics.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0
                and (key == 'loss' or value <= 1), f'Métrica inválida: {key}')
    require(positive(metrics.get('num_samples')), 'Validação vazia')
    require(metrics.get('class_order') == list(CLASSES), 'Ordem de classes inválida')
    matrix = metrics.get('confusion')
    require(isinstance(matrix, list) and len(matrix) == 7 and all(
        isinstance(row, list) and len(row) == 7 and all(type(n) is int and n >= 0 for n in row)
        for row in matrix), 'Confusão inválida')
    require(sum(map(sum, matrix)) == metrics['num_samples'], 'Suporte/confusão incompatíveis')
    per_class = metrics.get('per_class', {})
    require(isinstance(per_class, dict) and set(per_class) == set(CLASSES), 'Métricas por classe ausentes')
    for name in CLASSES:
        measures = per_class[name]
        require(isinstance(measures, dict), 'Métricas por classe inválidas')
        for key in ('precision', 'recall', 'f1'):
            value = measures.get(key)
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                    f'Métrica por classe inválida: {name}/{key}')


def training_artifacts(output, kind, summary=None):
    """Check exported JSON/files, never torch.load/pickle on the controller."""
    output = Path(output)
    for name in ('summary.json', 'last.pt', 'best.pt', 'confusion.json', 'val_predictions.csv'):
        regular(output / name)
    summary = summary if summary is not None else read_json(output / 'summary.json')
    require(summary.get('exit_code') == 0 and summary.get('status') in GOOD,
            'Treino não terminou com export válido')
    require(positive(summary.get('best_epoch')), 'Nenhuma melhor época treinada')
    if kind == 'crop':
        require(summary.get('schema_version') == 1 and summary.get('split') == 'val', 'Schema/split inválido')
        require(positive(summary.get('epochs_ran')) and positive(summary.get('exported_epoch')),
                'Export de época zero não é sucesso')
        require(summary.get('export_complete') is True and summary.get('initial_checkpoint_fallback') is False,
                'Export incompleto ou fallback inicial')
        require(summary['exported_epoch'] == summary['best_epoch'] <= summary['epochs_ran'],
                'Épocas do export inconsistentes')
        valid_metrics(summary.get('best_metrics'))
        metrics = summary.get('exported_metrics')
        metadata = summary.get('metadata', {})
        require(metadata.get('splits_accessed') == ['train', 'val'], 'Splits de treino inesperados')
        for split in ('train', 'val'):
            manifest = read_json(regular(output / f'{split}_manifest.json'))
            require(manifest.get('split') == split and positive(manifest.get('selected_count')),
                    'Manifesto incompleto')
            require(manifest.get('sha256') == metadata.get('manifests', {}).get(split, {}).get('sha256'),
                    'Manifesto não corresponde ao resumo')
    else:
        require(positive(summary.get('epoch')) and summary['best_epoch'] <= summary['epoch'],
                'RoI sem época completa')
        require(summary.get('cache_metadata', {}).get('complete') is True, 'Cache RoI incompleto')
        metrics = summary.get('metrics')
    valid_metrics(metrics)
    confusion = read_json(output / 'confusion.json')
    require(confusion.get('split') == 'val' and confusion.get('epoch') == summary['best_epoch']
            and confusion.get('metrics') == metrics, 'Confusão não corresponde ao export')
    # Streaming count avoids retaining a large validation CSV in the controller.
    import csv
    with (output / 'val_predictions.csv').open(newline='', encoding='utf-8') as handle:
        rows = csv.DictReader(handle)
        require(rows.fieldnames == ['path', 'true', 'pred', 'confidence'], 'CSV incompleto')
        count = 0
        for row in rows:
            require(row.get('true') in CLASSES and row.get('pred') in CLASSES, 'CSV inválido')
            count += 1
        require(count == metrics['num_samples'], 'CSV sem todas as predições')
    return summary


def cache_artifacts(output, smoke):
    output = Path(output)
    metadata = read_json(regular(output / 'metadata.json'))
    contract = metadata.get('contract', {})
    entries = contract.get('entries')
    require(metadata.get('complete') is True and metadata.get('status') == 'complete', 'Cache incompleto')
    require(metadata.get('digest') == digest(contract), 'Digest do cache inválido')
    require(contract.get('schema') == 1 and contract.get('classes') == list(CLASSES), 'Schema/classes RoI inválidos')
    require(contract.get('limit_images_per_split') == (1 if smoke else None), 'Limite do cache incorreto')
    require(isinstance(entries, list) and entries and
            {e.get('split') for e in entries} == {'train', 'val'}, 'Splits do cache inválidos')
    require(metadata.get('images_checked') == len(entries) and positive(metadata.get('channels')),
            'Cache não verificou todas as imagens')
    for index in range(len(entries)):
        regular(output / f'{index:06d}.pt')
    return metadata


def diagnostic_artifacts(output):
    output = Path(output)
    summary = training_artifacts(output, 'crop')
    report = read_json(regular(output / 'diagnostics/diagnostics.json'))
    latency = read_json(regular(output / 'diagnostics/latency.json'))
    require(report.get('status') == 'completed' and report.get('splits_accessed') == ['val'],
            'Diagnóstico incompleto')
    require(report.get('experiment_id') == summary.get('experiment_id')
            and report.get('checkpoint_epoch') == summary['exported_epoch'], 'Diagnóstico de outro export')
    require(report.get('csv_sha256') == hashlib.sha256((output / 'val_predictions.csv').read_bytes()).hexdigest(),
            'Diagnóstico de outro CSV')
    require(report.get('latency') == latency and latency.get('scope') == 'classifier_only; NOT API E2E',
            'Latência ausente ou escopo inválido')
    results = latency.get('results', [])
    require({r.get('batch_size') for r in results} == {1, 32}, 'Batches de latência incompletos')
    for result in results:
        for key in ('batch_p50_ms', 'batch_p95_ms', 'item_p50_ms', 'item_p95_ms'):
            value = result.get(key)
            require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                    'Latência inválida')
    samples = report.get('samples', [])
    require(positive(report.get('selected_samples')) and len(samples) == report['selected_samples'],
            'Amostras diagnósticas incompletas')
    for sample in samples:
        for key in ('heatmap', 'overlay'):
            path = output / sample[key]
            require(path.resolve().is_relative_to(output / 'diagnostics'), 'PNG fora de diagnostics')
            regular(path)
    return report


def choose_extension(summaries, consumed=False):
    """Best NONbaseline first, then budget and >= baseline + .005; stable ties.

    The caller supplies only verified successful full-crop exports. RoI/smokes
    cannot enter this decision. A better converged candidate prevents extending
    a worse budget-limited candidate. No test or accuracy tiebreak is used.
    """
    if consumed or 'baseline' not in summaries:
        return None
    baseline = summaries['baseline']['exported_metrics']['species_macro_f1']
    candidates = [v for v in VARIANTS[1:] if v in summaries]
    if not candidates:
        return None
    best = max(candidates, key=lambda v: summaries[v]['exported_metrics']['species_macro_f1'])
    summary = summaries[best]
    score = summary['exported_metrics']['species_macro_f1']
    if math.isfinite(score) and math.isfinite(baseline) and score >= baseline + .005 and summary['status'] == 'budget':
        return best
    return None


def stage(key, kind, output, command, dependencies=()):
    return {'id': key, 'kind': kind, 'out': str(output), 'command': command,
            'dependencies': list(dependencies)}


def build_stages(out, source):
    out, source = Path(out), Path(source)
    python = [sys.executable, '-m']
    stages = [stage('tests', 'tests', out, python + ['unittest', 'discover', '-s', str(source / 'tests'),
                                                    '-p', 'test_architecture_*.py'])]
    for smoke in (True, False):
        for variant in VARIANTS:
            key = f'smoke_{variant}' if smoke else f'crop_{variant}'
            output = out / ('smoke' if smoke else 'crops') / variant
            command = python + ['architecture_lab.train', '--data', str(CROPS), '--out', str(output),
                '--variant', variant, '--epochs', '1' if smoke else '24', '--patience', '5',
                '--batch', '4' if smoke else '32', '--img-size', '384', '--workers', '0' if smoke else '8',
                '--lr', '8e-5', '--seed', '42', '--max-hours', '.1' if smoke else '1.25']
            command += (['--limit-train', '8', '--limit-val', '8', '--no-pretrained'] if smoke else
                        wandb_flags(variant, out))
            stages.append(stage(key, 'crop', output, command,
                                ('tests',) if smoke else ('tests', f'smoke_{variant}')))
        if smoke:
            stages.extend(roi_stages(out, smoke=True))
    stages.extend(roi_stages(out, smoke=False))
    return stages


def wandb_flags(variant, out):
    return ['--wandb', '--wandb-entity', 'pestline', '--wandb-project', 'fly-species',
            '--wandb-run-name', f'A1_{variant}_{Path(out).name}']


def roi_stages(out, smoke):
    prefix = 'smoke_roi' if smoke else 'roi'
    root = out / 'smoke/roi' if smoke else out / 'roi'
    cache = stage(prefix + '_cache', 'cache', root / 'cache',
        [sys.executable, '-m', 'architecture_lab.roi', 'cache', '--data', str(YOLO_DATA),
         '--detector', str(DETECTOR), '--out', str(root / 'cache')] + (['--limit-images', '1'] if smoke else []),
        ('tests',) if smoke else ('tests', 'smoke_roi_train'))
    head = stage(prefix + '_train', 'roi', root / 'head',
        [sys.executable, '-m', 'architecture_lab.roi', 'train', '--cache', str(root / 'cache'),
         '--out', str(root / 'head'), '--epochs', '1' if smoke else '30', '--patience', '5',
         '--batch', '32', '--seed', '42', '--max-hours', '.1' if smoke else '1'] +
        ([] if smoke else wandb_flags('roi', out)), ('tests', prefix + '_cache'))
    return [cache, head]


def extension_stage(base):
    result = {**base, 'id': 'extension_' + base['id'].removeprefix('crop_'),
              'dependencies': list(base['dependencies'])}
    command = list(base['command'])
    command[command.index('--max-hours') + 1] = '1.5'
    result['command'] = command + ['--resume']
    return result


def attempt_command(spec):
    command = list(spec['command'])
    if spec['kind'] in ('crop', 'roi') and (Path(spec['out']) / 'last.pt').is_file() and '--resume' not in command:
        command.append('--resume')
    return command


def source_identity(source):
    paths = [source / 'ARCHITECTURE_NIGHT.md', source / 'fly-det/slurm/architecture_night.sh']
    paths += sorted((source / 'fly-det/scripts/architecture_lab').glob('*.py'))
    paths += sorted((source / 'tests').glob('test_architecture_*.py'))
    require(any(p.name == 'test_architecture_campaign.py' for p in paths), 'Snapshot sem testes da campanha')
    return {'root': str(source), 'commit': os.environ.get('FLYDET_SOURCE_COMMIT'),
            'sha256': {p.relative_to(source).as_posix(): hashlib.sha256(regular(p).read_bytes()).hexdigest()
                       for p in paths}}


def load_state(path, identity, plan):
    if not path.exists():
        return {'schema_version': SCHEMA, 'source': identity, 'plan': plan, 'created_at': now(),
                'status': 'ready', 'stages': {}, 'extension': None}
    state = read_json(path)
    require(state.get('schema_version') == SCHEMA and state.get('source') == identity
            and state.get('plan') == plan, 'Snapshot/contrato mudou; use outra campanha')
    require(isinstance(state.get('stages'), dict), 'Estado da campanha inválido')
    return state


def runtime_guard(args):
    # In particular, unit-test discovery must NEVER recursively start a campaign.
    if os.environ.get(TEST_MARKER):
        raise RuntimeError('Campanha recursiva proibida durante testes')
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Campanha requer Slurm; nenhum import de Torch no login')
    if not any(os.environ.get(key) for key in ('APPTAINER_CONTAINER', 'APPTAINER_NAME',
                                              'SINGULARITY_CONTAINER', 'SINGULARITY_NAME')):
        raise RuntimeError('Campanha requer Apptainer')
    if (os.environ.get('SLURM_JOB_PARTITION') != 'h100n2'
            or int(os.environ.get('WORLD_SIZE', '1')) != 1
            or int(os.environ.get('SLURM_PROCID', '0')) != 0
            or int(os.environ.get('SLURM_NTASKS', '1')) != 1):
        raise RuntimeError('Requer um processo Slurm na partição h100n2')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or len(visible.split(',')) != 1 or visible in ('-1', 'NoDevFiles'):
        raise RuntimeError('Exatamente uma GPU deve estar visível na alocação')
    if os.environ.get('WANDB_MODE', 'online') != 'online':
        raise RuntimeError('Treinos completos requerem W&B online')
    out, source = Path(args.out).resolve(), Path(args.source).resolve()
    require(out.is_relative_to(RAID) and source.is_relative_to(RAID), 'Saída/snapshot devem estar no RAID')
    require(source == out / 'source', '--source deve ser o snapshot <out>/source do launcher')
    require(Path(__file__).resolve() == source / 'fly-det/scripts/architecture_lab/campaign.py',
            'Executar o módulo do snapshot, não do workspace mutável')
    for data in (CROPS, YOLO_DATA, DETECTOR):
        require(not out.is_relative_to(data) and not data.is_relative_to(out), 'Saída sobrepõe dados/pesos')
    # These directories are inherited by unittest BEFORE any optional Torch import.
    for key, suffix in {'XDG_CACHE_HOME': 'xdg', 'TORCH_HOME': 'torch', 'HF_HOME': 'hf',
        'HF_HUB_CACHE': 'hf/hub', 'HUGGINGFACE_HUB_CACHE': 'hf/hub', 'PIP_CACHE_DIR': 'pip',
        'WANDB_CACHE_DIR': 'wandb/cache', 'WANDB_CONFIG_DIR': 'wandb/config',
        'WANDB_DATA_DIR': 'wandb/data', 'WANDB_DIR': 'wandb', 'YOLO_CONFIG_DIR': 'ultralytics',
        'MPLCONFIGDIR': 'matplotlib', 'CUDA_CACHE_PATH': 'cuda', 'TMPDIR': 'tmp'}.items():
        target = (RAID / 'cache/architecture_campaign' / suffix).resolve()
        require(target.is_relative_to(RAID), 'Cache fora do RAID')
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    tempfile.tempdir = os.environ['TMPDIR']
    return out, source


class Controller:
    def __init__(self, out, source, state):
        self.out, self.source, self.state = Path(out), Path(source), state
        self.child = None
        self.signal_number = None

    def handle_signal(self, number, _frame):
        self.signal_number = number
        if self.child is not None:
            try:
                self.child.send_signal(signal.SIGUSR1)
            except ProcessLookupError:
                pass

    def persist(self):
        self.state['updated_at'] = now()
        write_json(self.out / 'campaign_state.json', self.state)
        atomic_write(self.out / 'report.md', render_report(self.out, self.state))

    def spawn(self, command, log):
        require(self.child is None, 'Já existe um subprocesso computacional')
        env = os.environ.copy()
        env.update(PYTHONPATH=str(self.source / 'fly-det/scripts'), PYTHONUNBUFFERED='1',
                   PYTHONDONTWRITEBYTECODE='1')
        if command[2:4] == ['unittest', 'discover']:
            env.update(ARCHITECTURE_ROI_TORCH_TESTS='1', **{TEST_MARKER: '1'})
        if self.signal_number is not None:
            return 75
        self.child = subprocess.Popen(command, cwd=self.source, env=env,
                                      stdout=log, stderr=subprocess.STDOUT)
        try:
            # Covers a signal delivered between the pre-spawn check and assignment.
            if self.signal_number is not None:
                self.handle_signal(self.signal_number, None)
            return self.child.wait()
        finally:
            self.child = None

    def verify(self, spec, record):
        kind, output = spec['kind'], Path(spec['out'])
        if kind == 'tests':
            with regular(self.out / record['log']).open('rb') as handle:
                handle.seek(record['attempts'][-1]['log_offset'])
                text = handle.read().decode('utf-8', errors='replace')
            require(re.search(r'Ran [1-9][0-9]* tests? in ', text) is not None
                    and re.search(r'^OK(?: \(.*\))?\s*$', text, re.MULTILINE) is not None,
                    'Testes não confirmaram execução positiva/OK')
            return {'status': 'completed', 'scope': 'unittest architecture; sem treino recursivo'}
        if kind == 'cache':
            return cache_artifacts(output, spec['id'].startswith('smoke_'))
        if kind == 'diagnostic':
            return diagnostic_artifacts(output)
        return training_artifacts(output, kind)

    def archive_partial(self, spec, record):
        output = Path(spec['out'])
        target = output / 'diagnostics' if spec['kind'] == 'diagnostic' else output
        needs_archive = (spec['kind'] == 'diagnostic' or
                         (spec['kind'] in ('crop', 'roi') and not (output / 'last.pt').exists()))
        if needs_archive and target.exists() and any(target.iterdir()):
            require(len(record['attempts']) > 0, 'Saída não rastreada; não sobrescrever')
            destination = self.out / 'archives' / f"{spec['id']}_before_attempt_{len(record['attempts']) + 1}"
            destination.parent.mkdir(exist_ok=True)
            require(not destination.exists(), 'Arquivo de tentativa já existe')
            os.replace(target, destination)
            record.setdefault('archives', []).append(destination.relative_to(self.out).as_posix())

    def execute(self, spec):
        if self.signal_number is not None:
            return False
        key = spec['id']
        record = self.state['stages'].setdefault(key, {'status': 'pending', 'attempts': [],
            'kind': spec['kind'], 'out': spec['out'], 'log': f'logs/{key}.log'})
        # Once extension is consumed, it owns this output. Never restart its base
        # with the original budget or silently grant an additional extension.
        extension = self.state.get('extension')
        if key.startswith('crop_') and extension and extension.get('claimed') and key == 'crop_' + extension['variant']:
            print(f'[{now()}] {key}: saída delegada à extensão já consumida', flush=True)
            try:
                self.verify(spec, record)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                record.update(status='failed', reason=f'Export sob extensão consumida inválido: {exc}')
                self.persist()
                return False
            return record['status'] == 'succeeded'
        blocked = [dep for dep in spec['dependencies']
                   if self.state['stages'].get(dep, {}).get('status') != 'succeeded']
        if blocked:
            record.update(status='blocked', reason='Dependências sem sucesso: ' + ', '.join(blocked))
            self.persist()
            return False
        if record['status'] == 'succeeded':
            try:
                self.verify(spec, record)
                print(f'[{now()}] {key}: sucesso anterior e artefatos verificados', flush=True)
                return True
            except (OSError, ValueError, KeyError, TypeError) as exc:
                record.update(status='failed', reason=f'Artefatos invalidados: {exc}')
        require(source_identity(self.source) == self.state['source'], 'Snapshot alterado durante campanha')
        log_path = self.out / record['log']
        log_path.parent.mkdir(exist_ok=True)
        try:
            self.archive_partial(spec, record)
            command = attempt_command(spec)
        except (OSError, ValueError) as exc:
            record.update(status='failed', reason=str(exc))
            with log_path.open('a', encoding='utf-8') as log:
                log.write(f'\n[{now()}] Preparação falhou: {exc}\n')
            self.persist()
            return False
        with log_path.open('ab', buffering=0) as log:
            log.write(f'\n=== {now()} {shlex.join(command)} ===\n'.encode())
            attempt = {'started_at': now(), 'command': command, 'log_offset': log.tell()}
            record['attempts'].append(attempt)
            record.update(status='running', reason=None)
            self.persist()
            print(f'[{now()}] INÍCIO {key}; log: {record["log"]}', flush=True)
            try:
                code = self.spawn(command, log)
                attempt['exit_code'] = code
                if self.signal_number is not None or code == 75:
                    self.signal_number = self.signal_number or signal.SIGUSR1
                    record.update(status='paused', reason='Sinal/manutenção; retomar manualmente, sem sbatch automático')
                elif code != 0:
                    record.update(status='failed', reason=f'Subprocesso retornou {code}')
                else:
                    result = self.verify(spec, record)
                    record.update(status='succeeded', result=result, reason=None)
                    attempt['result'] = result
            except (OSError, ValueError, KeyError, TypeError) as exc:
                record.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
                log.write(f'\nControlador: {record["reason"]}\n'.encode())
            attempt['finished_at'] = now()
        self.persist()
        print(f'[{now()}] FIM {key}: {record["status"]}', flush=True)
        return record['status'] == 'succeeded'

    def crop_results(self):
        results = {}
        for variant in VARIANTS:
            record = self.state['stages'].get('crop_' + variant, {})
            if record.get('status') == 'succeeded':
                try:
                    results[variant] = training_artifacts(record['out'], 'crop')
                except (OSError, ValueError, KeyError, TypeError):
                    continue
        return results

    def run(self):
        self.state['status'] = 'running'
        self.state.pop('exit_code', None)
        self.state['signal'] = None
        self.persist()
        for spec in self.state['plan']:
            passed = self.execute(spec)
            if self.signal_number is not None:
                return self.finish('paused', 75)
            if spec['id'] == 'tests' and not passed:
                return self.finish('aborted_tests', 1)
        if self.state['extension'] is None:
            results = self.crop_results()
            chosen = choose_extension(results)
            self.state['extension'] = {'variant': chosen, 'claimed': False, 'decided_at': now(),
                'selection': {v: {'status': s['status'], 'species_macro_f1': s['exported_metrics']['species_macro_f1']}
                              for v, s in results.items()},
                'rule': 'melhor não-baseline; >= baseline + 0.005; status budget; uma invocação'}
            self.persist()
        decision = self.state['extension']
        if decision['variant'] and not decision['claimed'] and self.signal_number is None:
            base = next(s for s in self.state['plan'] if s['id'] == 'crop_' + decision['variant'])
            decision.update(claimed=True, claimed_at=now())
            self.persist()  # At-most-once even across process/host crashes.
            self.execute(extension_stage(base))
        if self.signal_number is not None:
            return self.finish('paused', 75)
        for variant in self.crop_results():
            output = self.out / 'crops' / variant
            self.execute(stage('diagnostic_' + variant, 'diagnostic', output,
                [sys.executable, '-m', 'architecture_lab.diagnose', '--run', str(output),
                 '--samples', '8', '--batch', '32'], ('tests', 'crop_' + variant)))
            if self.signal_number is not None:
                return self.finish('paused', 75)
        failed = any(r['status'] != 'succeeded' for r in self.state['stages'].values())
        if decision.get('claimed'):
            failed |= self.state['stages'].get('extension_' + decision['variant'], {}).get('status') != 'succeeded'
        return self.finish('completed_with_failures' if failed else 'completed', 1 if failed else 0)

    def finish(self, status, code):
        if self.signal_number is not None:
            status, code = 'paused', 75
        self.state.update(status=status, exit_code=code, signal=self.signal_number)
        self.persist()
        print(f'[{now()}] Campanha: {status}; retorno {code}; report.md atualizado', flush=True)
        return code


def link(out, path, label):
    relative = os.path.relpath(path, out)
    return f'[{label}]({quote(relative, safe="/._-")})'


def render_report(out, state):
    lines = ['# Campanha de arquiteturas — validação', '', f'**Estado:** {state["status"]}',
        f'Atualizado (UTC): {state.get("updated_at", state["created_at"])}', '',
        f'Fonte: {link(out, state["source"]["root"], "snapshot imutável")}; '
        f'commit: {state["source"].get("commit") or "não informado"}. '
        'Hashes dos fontes, comandos e tentativas: [campaign_state.json](campaign_state.json).', '',
        'Somente train/val existentes; proveniência por placa/local/período ainda não comprovada. '
        'Nenhum acesso ao split test nesta campanha. Seleção por macro-F1 das quatro espécies na matriz de sete classes. '
        'Imagens originais intactas. Resultados exploratórios de validação, sem confirmação independente.', '',
        '## Estágios', '', '| Estágio | Estado | Tentativas | Log | Motivo |', '|---|---|---:|---|---|']
    records = state['stages']
    for key, record in records.items():
        reason = str(record.get('reason') or '—').replace('|', '/').replace('\n', ' ')
        lines.append(f'| {key} | {record["status"]} | {len(record["attempts"])} | '
                     f'{link(out, out / record["log"], "log (append)")} | {reason} |')
    pending = [s['id'] for s in state['plan'] if s['id'] not in records]
    if pending:
        lines += ['', 'Ainda não executados: ' + ', '.join(pending) + '.']
    lines += ['', '## Smokes — somente verificação técnica, fora da seleção', '']
    for key, record in records.items():
        result = record.get('result')
        if key.startswith('smoke_') and record.get('kind') in ('crop', 'roi') and result:
            lines += report_training(out, key, record, result, record['kind'])
    lines += ['', '## Crops — mesma população prevista; RoI não participa do ranking', '']
    available = {}
    for variant in VARIANTS:
        base = records.get('crop_' + variant, {})
        extended = records.get('extension_' + variant, {})
        record = extended if extended.get('status') == 'succeeded' else base
        summary = record.get('result')
        if not summary:
            continue
        lines += report_training(out, variant, record, summary, 'crop')
        if base.get('status') == 'succeeded' and (not extended or extended.get('status') == 'succeeded'):
            available[variant] = summary
        if extended and extended.get('status') != 'succeeded':
            lines += ['**Extensão não concluída:** métricas acima são do export anterior, não da extensão.', '']
        diagnostic = records.get('diagnostic_' + variant, {})
        report = diagnostic.get('result') if diagnostic.get('status') == 'succeeded' else None
        lines += [f'Diagnóstico: {diagnostic.get("status", "pendente/não executado")}.']
        if report:
            lines += [link(out, Path(base['out']) / 'diagnostics/diagnostics.json', 'Grad-CAM/diagnostics.json'),
                      '', '| Batch | p50 batch (ms) | p95 batch (ms) | p50/crop (ms) | p95/crop (ms) |',
                      '|---:|---:|---:|---:|---:|']
            for r in report['latency']['results']:
                lines.append(f'| {r["batch_size"]} | {r["batch_p50_ms"]:.4f} | {r["batch_p95_ms"]:.4f} | '
                             f'{r["item_p50_ms"]:.4f} | {r["item_p95_ms"]:.4f} |')
        lines.append('')
    lines += ['## RoI — população GT distinta, fora do ranking de crops', '',
              'E0 congelado, caixas GT, imagens nativas 1920×1080 com padding. '
              'Não mede propostas do detector nem treinamento conjunto.', '']
    roi = records.get('roi_train', {})
    if roi.get('result'):
        lines += report_training(out, 'roi', roi, roi['result'], 'roi')
    else:
        lines += [f'Estado RoI: {roi.get("status", "pendente")}.', '']
    lines += ['## Extensão e próximos experimentos propostos', '']
    decision = state.get('extension')
    if decision is None:
        lines.append('Decisão de extensão ainda pendente.')
    else:
        lines.append(f'Candidato à extensão: {decision["variant"] or "nenhum elegível"}; '
                     f'invocação consumida: {"sim" if decision["claimed"] else "não"}. '
                     'Uma única chamada adicional, inclusive em caso de falha/sinal; sem auto-submissão.')
    if available:
        best = max(available, key=lambda v: available[v]['exported_metrics']['species_macro_f1'])
        m = available[best]['exported_metrics']
        weakest = min(('MD', 'MV', 'MC', 'MF'), key=lambda c: m['per_class'][c]['recall'])
        lines += [f'Melhor crop observado com estágio bem-sucedido: **{best}**, '
                  f'macro-F1 espécies val = **{m["species_macro_f1"]:.4f}** (empates preservam ordem fixa).',
                  f'Propostas, NÃO executadas: replicar {best} e baseline com outras sementes; '
                  f'auditar confusões e informação visual de {weakest} (menor recall de espécie); '
                  'confirmar split agrupado por placa/local/período e avaliar em conjunto independente.']
    else:
        lines.append('Sem candidato válido para recomendação; corrigir falhas dos logs antes de comparar arquiteturas.')
    lines += ['', '## Limites de interpretação', '',
        '- Não há estimativa de mAP nem afirmação de ganho na API ponta a ponta nesta campanha.',
        '- Latências são microbenchmarks CUDA FP32 somente do classificador (batch 1/32); '
        'excluem detector, leitura, preprocessamento, transporte e API. Latência/crop em batch é amortizada.',
        '- Grad-CAM é qualitativo, não causal, nem segmentação.',
        '- Smokes não são resultados científicos; métricas e histórico dos exports estão nas tentativas do estado.',
        '- O controlador verifica JSON/arquivos completos; tensors/checkpoints são validados pelos filhos, '
        'não desserializados pelo controlador. Snapshots e datasets devem permanecer imutáveis.',
        '- Orçamentos por chamada: crops 1,25h, extensão 1,5h, RoI 1h; cache/export são adicionais. '
        'Limite total externo: Slurm 12h. Pausa retorna 75, sem submissão automática.', '']
    return '\n'.join(lines)


def report_training(out, name, record, summary, kind):
    metrics = summary['exported_metrics'] if kind == 'crop' else summary['metrics']
    output = Path(record['out'])
    args = summary['metadata']['args']
    url = (f'https://wandb.ai/{quote(args.get("wandb_entity", "pestline"), safe="")}/'
           f'{quote(args.get("wandb_project", "fly-species"), safe="")}/runs/'
            f'{quote(str(summary["wandb_run_id"]), safe="")}') if args.get('wandb') and summary.get('wandb_run_id') else None
    lines = [f'### {name}', '', f'Estágio: {record["status"]}; treino: {summary["status"]}; '
        f'épocas completas: {summary.get("epochs_ran", summary.get("epoch"))}; melhor época: {summary["best_epoch"]}.',
        f'{link(out, output / "summary.json", "Resumo atual")} · '
        f'{link(out, output / "confusion.json", "Confusão val")} · '
        f'{link(out, output / "val_predictions.csv", "Predições val")} · ' +
        (f'[W&B]({url})' if url else 'W&B indisponível'), '',
        '| Métrica val | Valor |', '|---|---:|']
    for key, value in metrics.items():
        if type(value) in (int, float):
            lines.append(f'| {key} | {value:.6g} |')
    lines += ['', '| Classe | Precision | Recall | F1 | Suporte |', '|---|---:|---:|---:|---:|']
    for name in CLASSES:
        m = metrics['per_class'][name]
        lines.append(f'| {name} | {m["precision"]:.4f} | {m["recall"]:.4f} | {m["f1"]:.4f} | {m.get("support", "—")} |')
    lines += ['', 'Confusão: linhas verdadeiras, colunas preditas.', '',
              '| Verdadeiro / predito | ' + ' | '.join(CLASSES) + ' |', '|---|' + '---:|' * 7]
    for name, row in zip(CLASSES, metrics['confusion']):
        lines.append('| ' + name + ' | ' + ' | '.join(map(str, row)) + ' |')
    return lines + ['']


@contextmanager
def campaign_lock(out):
    with (out / '.controller.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Outra instância controla esta campanha') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--source', required=True, help='Snapshot root: <out>/source')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    out, source = runtime_guard(args)  # Before discovery/import of ANY tests.
    out.mkdir(parents=True, exist_ok=True)
    with campaign_lock(out):
        state = load_state(out / 'campaign_state.json', source_identity(source), build_stages(out, source))
        controller = Controller(out, source, state)
        previous = {s: signal.signal(s, controller.handle_signal)
                    for s in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
        try:
            return controller.run()
        except Exception as exc:
            # No credential-bearing exception messages in the automatic report.
            state['controller_error_type'] = type(exc).__name__
            return controller.finish('failed_controller', 1)
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)


if __name__ == '__main__':
    raise SystemExit(main())