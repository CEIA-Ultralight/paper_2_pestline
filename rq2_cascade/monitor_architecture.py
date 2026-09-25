"""Lightweight Slurm control-plane monitor, not training or an AI agent.

Runs from cron every ten minutes, stdlib only. No GPU, torch, checkpoints,
credentials, dataset access, job submissions, or Git operations. Automatically
stops querying after both jobs terminate or after 72 hours from first execution.
The installed cron entry then becomes a no-op until manually removed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path('/raid/user_marcospaulo')
OUT = ROOT / 'flydet_runs/architecture-monitor'
JOBS = ('32005', '32009')
CAMPAIGNS = (
    ('fase1', ROOT / 'flydet_runs/architecture-night-v1-20260907', 'campaign_state.json', 'report.md'),
    ('fase2', ROOT / 'flydet_runs/architecture-followup-v1-20260908', 'followup_state.json', 'followup_report.md'),
)
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'PREEMPTED',
            'NODE_FAIL', 'OUT_OF_MEMORY', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def atomic(path, text):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json(path):
    return json.loads(path.read_text())


def collect():
    process = subprocess.run(
        ['/usr/bin/sacct', '-n', '-X', '-j', ','.join(JOBS),
         '--format=JobIDRaw,State,Elapsed,ExitCode', '--parsable2'],
        check=True, capture_output=True, text=True, timeout=25)
    jobs = {}
    for line in process.stdout.splitlines():
        fields = line.strip().split('|')
        if len(fields) >= 4 and fields[0] in JOBS:
            jobs[fields[0]] = dict(state=fields[1].split()[0].rstrip('+'),
                                   elapsed=fields[2], exit_code=fields[3])
    campaigns = {}
    for label, root, state_name, report_name in CAMPAIGNS:
        state_path = root / state_name
        if not state_path.exists():
            campaigns[label] = {'status': 'not_started', 'stages': []}
            continue
        state = read_json(state_path)
        stages = []
        for name, stage in state.get('stages', {}).items():
            item = {'name': name, 'status': stage.get('status'), 'reason': stage.get('reason')}
            # Use small per-training summaries; never deserialize .pt files.
            if stage.get('kind') == 'crop' and not name.startswith('smoke_'):
                output = Path(stage.get('out', '')).resolve()
                if output.is_relative_to(root) and (output / 'summary.json').is_file():
                    summary = read_json(output / 'summary.json')
                    metrics = summary.get('exported_metrics') or {}
                    item.update(epochs=summary.get('epochs_ran'),
                                species_accuracy=metrics.get('species_accuracy'),
                                species_macro_f1=metrics.get('species_macro_f1'),
                                wandb_run_id=summary.get('wandb_run_id'))
            stages.append(item)
        campaigns[label] = {'status': state.get('status'), 'stages': stages,
                            'report': str(root / report_name)}
    return jobs, campaigns


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'monitor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        now = datetime.now(timezone.utc)
        state_path = OUT / 'status.json'
        state = read_json(state_path) if state_path.exists() else {
            'started_at': now.isoformat(), 'deadline': (now + timedelta(hours=72)).isoformat(),
            'checks': 0, 'stopped': False,
        }
        if state.get('stopped'):
            return 0
        state.update(checked_at=now.isoformat(), checks=state['checks'] + 1)
        if now >= datetime.fromisoformat(state['deadline']):
            state.update(stopped=True, stop_reason='72h_limit', needs_review=True)
        else:
            try:
                jobs, campaigns = collect()
                all_finished = set(jobs) == set(JOBS) and all(j['state'] in TERMINAL for j in jobs.values())
                problems = [s for c in campaigns.values() for s in c['stages']
                            if s['status'] in {'failed', 'blocked', 'paused', 'budget_exhausted'}]
                state.update(jobs=jobs, campaigns=campaigns, error=None, stopped=all_finished,
                             needs_review=all_finished or bool(problems) or len(jobs) != len(JOBS))
                if all_finished:
                    state['stop_reason'] = 'all_jobs_terminal'
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
                state.update(error=type(exc).__name__, needs_review=True)
        lines = ['# Monitor automático de arquiteturas', '',
                 f"Última verificação UTC: {state['checked_at']}",
                 f"Verificações: {state['checks']}; encerrado: {state['stopped']}",
                 f"Revisão necessária: {state.get('needs_review', False)}", '',
                 '**Este monitor não é um agente de IA e não desperta o chat.**',
                 'Não submete novos treinos. Os controladores existentes executam suas decisões programadas.', '',
                 '| Job | Estado | Tempo | Saída |', '|---|---|---|---|']
        for job, data in state.get('jobs', {}).items():
            lines.append(f"| {job} | {data['state']} | {data['elapsed']} | {data['exit_code']} |")
        for name, campaign in state.get('campaigns', {}).items():
            lines += ['', f"## {name}: {campaign['status']}"]
            for stage in campaign['stages']:
                detail = ''
                if stage.get('species_macro_f1') is not None:
                    detail = f"; val macro-F1 espécies={stage['species_macro_f1']:.4f}; épocas={stage['epochs']}"
                lines.append(f"- {stage['name']}: {stage['status']}{detail}")
        lines += ['', f"Erro de consulta: {state.get('error') or 'nenhum'}",
                  f"Motivo de encerramento: {state.get('stop_reason', 'ainda monitorando')}"]
        atomic(state_path, json.dumps(state, indent=2) + '\n')
        atomic(OUT / 'monitor.md', '\n'.join(lines) + '\n')
        with (OUT / 'events.jsonl').open('a') as events:
            events.write(json.dumps({'at': state['checked_at'], 'jobs': state.get('jobs'),
                                     'error': state.get('error'), 'stopped': state['stopped'],
                                     'needs_review': state.get('needs_review')}) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())