#!/usr/bin/env python3
"""Monitor leve da fila de campanha 2026-09-09 (jobs 32144–32149).

Somente leitura (sacct/squeue/logs): escreve status.md em
/raid/user_marcospaulo/flydet_runs/autonomous-20260909/. Não reinicia jobs,
não cancela, não submete nada. Roda via cron a cada 10 min; sai em silêncio
quando a fila terminar (marca FINISHED e pode ser removida do cron).
"""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

JOBS = {32144: 'dinov2', 32145: 'proposal-crops', 32146: 'proposal-classifier',
        32147: 'yolo-extended', 32148: 'policy-study', 32149: 'val-study'}
OUT = Path('/raid/user_marcospaulo/flydet_runs/autonomous-20260909')


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for jid in sorted(JOBS):
        info = run(['sacct', '-j', str(jid), '--format=JobID,State,Elapsed,ExitCode', '-n'])
        head = [ln.split() for ln in info.splitlines() if ln.strip() and not ln.strip().startswith(('321',)) or True]
        lines = [ln.split() for ln in info.splitlines() if ln.strip().startswith(str(jid))]
        state, elapsed, code = ('PENDING', '-', '-') if not lines else (
            lines[0][1], lines[0][2], lines[0][3])
        rows.append((jid, JOBS[jid], state, elapsed, code))
    done = all(r[2] not in ('PENDING', 'RUNNING', 'CONFIGURING') for r in rows)
    ts = datetime.now(timezone.utc).isoformat(timespec='seconds')
    lines = [f'# Fila autônoma 2026-09-09 — {ts}', '',
             '| Job | Etapa | Estado | Tempo | Exit |', '|---|---|---|---|---|']
    lines += [f'| {j} | {n} | {s} | {e} | {c} |' for j, n, s, e, c in rows]
    if done:
        lines += ['', '**Fila terminada.** Resultados:',
                  '- dinov2: /raid/user_marcospaulo/flydet_runs/dinov2-20260909/',
                  '- crops propostas: /raid/user_marcospaulo/datasets/DS-F2_crops_proposals/manifest.json',
                  '- classificador propostas: /raid/user_marcospaulo/flydet_runs/proposal-cls-20260909/',
                  '- yolo estendido: /raid/user_marcospaulo/flydet_runs/E0_yolo26m_baseline/runs/E6_*,E7_*',
                  '- políticas VAL: /raid/user_marcospaulo/flydet_runs/policy_study_val/report.md',
                  '- estudo VAL: /raid/user_marcospaulo/flydet_runs/val-study-20260909/',
                  '',
                  'Pendente (não automatizado): avaliar configuração selecionada no TEST e atualizar o relatório.']
        marker = OUT / 'FINISHED'
        marker.write_text(ts)
    (OUT / 'status.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'events.jsonl').open('a').write(json.dumps(
        {'ts': ts, 'rows': rows, 'done': done}) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
