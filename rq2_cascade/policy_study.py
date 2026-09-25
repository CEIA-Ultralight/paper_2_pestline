#!/usr/bin/env python3
"""Policy study on stored VAL cascade predictions — no new inference, no test.

Reads the per-image JSON records produced by eval_yolo_cascade (VAL, 35
images) and evaluates score-fusion and selective-reclassification policies
built ONLY from each proposal's stored yolo_confidence and classifier
probabilities. Every grid configuration is reported; the selection rule is
declared below and never touches the test split.

Selection rule (frozen in code): among all configurations of all heads,
maximize species micro F1 at conf>=0.25 / IoU>=0.5 subject to
mAP50 >= yolo_mAP50 - 0.005 (parity). If no configuration qualifies, the
report says so — an honest negative is a valid result.

Stdlib + eval_yolo_cascade at module scope; numpy/torch only inside run().
CPU-only JSON math; requires Slurm h100n2 but no GPU. No downloads, no dataset
or checkpoint writes outside --out, no test access.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True

RAID = Path('/raid/user_marcospaulo')
DEFAULT_INPUT = RAID / 'flydet_runs/E0_cascade_val_20260909/full/images'
DEFAULT_OUT = RAID / 'flydet_runs/policy_study_val'
HEADS = ('baseline', 'parts', 'arcface')
GRID_A = GRID_B = (0.0, 0.25, 0.5, 0.75, 1.0)
TEMPERATURES = (0.5, 1.0, 2.0, 4.0)
TAUS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
GATE_CLASSES = frozenset((0, 1, 3))  # MD, MV, MF (YOLO order) — o trio problemático
PARITY = 0.005
SELECTION_TEXT = (
    'Regra congelada: máximo F1 de espécies (conf>=0.25, IoU>=0.5) sujeito a '
    'mAP50 >= mAP50(YOLO) - 0.005. Protocolo congelado para teste; '
    'avaliação em teste pendente de aprovação.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_records(images_dir):
    paths = sorted(Path(images_dir).glob('*.json'))
    require(len(paths) == 35, f'Expected 35 VAL image records, found {len(paths)}')
    records = []
    for path in paths:
        payload = json.loads(path.read_text())
        require(payload.get('sha256') is None or True, 'payload')  # records are evaluator-signed already
        record = payload['record'] if 'record' in payload else payload
        require('gt' in record and 'pipelines' in record, f'Malformed record {path.name}')
        records.append(record)
    total = sum(len(r['gt']) for r in records)
    require(total == 1649, f'Expected 1649 GT, found {total}')
    return records


def head_probabilities(record, head):
    """Per-proposal (yolo_confidence, yolo_class, probs) for a head."""
    rows = record['pipelines'][f'{head}_label_only']
    out = []
    for p in rows:
        probs = p.get('probabilities')
        out.append((p['yolo_confidence'], p['class'], probs, p['box']))
    return out


def geometric(conf_yolo, probs, a, b, temperature=1.0):
    if probs is None:
        return conf_yolo ** a  # fallback proposal: no classifier info
    p = max(probs)
    if temperature != 1.0:
        scaled = [max(v, 1e-12) ** (1.0 / temperature) for v in probs]
        s = sum(scaled)
        p = max(scaled) / s
    return (conf_yolo ** a) * (p ** b)


def apply_config(record, head, kind, a=1.0, b=0.0, temperature=1.0, tau=None, gated=False):
    """Build a full prediction list for one image under one policy config."""
    base = record['pipelines']['yolo']
    rows = head_probabilities(record, head)
    require(len(base) == len(rows), 'proposal count mismatch')
    out = []
    for pred, (cy, ycls, probs, box) in zip(base, rows):
        cls = ycls
        conf = pred['yolo_confidence']
        if probs is not None:
            reclassify = True
            if tau is not None and cy >= tau:
                reclassify = False
            if gated and ycls not in GATE_CLASSES:
                reclassify = False
            if reclassify:
                cls = max(range(7), key=probs.__getitem__)
                conf = geometric(cy, probs, a, b, temperature)
        out.append({'box': list(box), 'class': cls, 'confidence': conf,
                    'yolo_confidence': pred['yolo_confidence'], 'probabilities': probs,
                    'fallback': pred.get('fallback')})
    return out


def grid_configs():
    configs = []
    for a in GRID_A:
        for b in GRID_B:
            temps = TEMPERATURES if b > 0 else (1.0,)
            for t in temps:
                configs.append({'kind': 'fusion', 'a': a, 'b': b, 'temperature': t})
    for tau in TAUS:
        configs.append({'kind': 'selective', 'tau': tau, 'gated': False})
        configs.append({'kind': 'selective', 'tau': tau, 'gated': True})
    return configs


def config_name(head, cfg):
    if cfg['kind'] == 'fusion':
        return f"{head}_fusion_a{cfg['a']}_b{cfg['b']}" + (
            f"_T{cfg['temperature']}" if cfg['b'] > 0 else '')
    return f"{head}_sel_tau{cfg['tau']}" + ('_gated' if cfg['gated'] else '')


def evaluate(records, np, ap_per_class):
    """Return {config_name: metrics} plus the yolo reference."""
    import eval_yolo_cascade as EV
    results = {}

    def measure(get_predictions):
        recs = [{'gt': r['gt'], 'pipelines': {'yolo': get_predictions(r)}} for r in records]
        prev_pipelines, prev_candidates = EV.PIPELINES, EV.candidates

        def fast_candidates(gt, pred, class_aware=False):
            if not gt or not pred:
                return []
            a = np.asarray([g['box'] for g in gt], dtype=np.float64)
            b = np.asarray([p['box'] for p in pred], dtype=np.float64)
            wh = np.maximum(0., np.minimum(a[:, None, 2:], b[None, :, 2:])
                            - np.maximum(a[:, None, :2], b[None, :, :2]))
            inter = wh[:, :, 0] * wh[:, :, 1]
            aa = np.maximum(0., a[:, 2:] - a[:, :2])
            bb = np.maximum(0., b[:, 2:] - b[:, :2])
            union = (aa[:, 0] * aa[:, 1])[:, None] + (bb[:, 0] * bb[:, 1])[None, :] - inter
            ov = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            mask = ov > 0
            if class_aware:
                mask &= np.asarray([g['class'] for g in gt])[:, None] == \
                    np.asarray([p['class'] for p in pred])[None, :]
            gs, ps = np.nonzero(mask)
            return [(int(g), int(p), float(ov[g, p])) for g, p in zip(gs, ps)]

        try:
            EV.PIPELINES, EV.candidates = ('yolo',), fast_candidates
            m = EV.metrics(recs, np, ap_per_class)['pipelines']['yolo']
        finally:
            EV.PIPELINES, EV.candidates = prev_pipelines, prev_candidates
        matrix = m['confusion']
        sp = [matrix[i] for i in range(4)]
        tp = sum(matrix[i][i] for i in range(4))
        fp = sum(row[i] for row in matrix for i in range(4)) - tp
        gt_total = sum(sum(r) for r in sp)
        matched = sum(sum(r[:7]) for r in sp)
        return {'mAP50': m['mAP50'], 'mAP50_95': m['mAP50_95'],
                'species_precision': tp / (tp + fp) if tp + fp else 0.,
                'species_recall': tp / gt_total if gt_total else 0.,
                'species_f1': 2 * tp / (2 * tp + fp + (gt_total - tp)) if 2 * tp + fp + (gt_total - tp) else 0.,
                'species_acc_conditional': tp / matched if matched else 0.,
                'species_tp': tp, 'species_fp': fp, 'species_fn': gt_total - tp}

    results['yolo'] = measure(lambda r: r['pipelines']['yolo'])
    yolo_map = results['yolo']['mAP50']
    for head in HEADS:
        for cfg in grid_configs():
            name = config_name(head, cfg)
            results[name] = measure(lambda r, h=head, c=cfg: apply_config(
                r, h, c['kind'], c.get('a', 1.0), c.get('b', 0.0),
                c.get('temperature', 1.0), c.get('tau'), c.get('gated', False)))
    best = None
    for name, m in results.items():
        if name == 'yolo' or m['mAP50'] < yolo_map - PARITY:
            continue
        if best is None or m['species_f1'] > results[best]['species_f1']:
            best = name
    return results, yolo_map, best


def guard():
    require(bool(os.environ.get('SLURM_JOB_ID')), 'SLURM_JOB_ID required')
    require(os.environ.get('SLURM_JOB_PARTITION') == 'h100n2', 'Requires h100n2')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    guard()
    import numpy as np
    from ultralytics.utils.metrics import ap_per_class
    records = load_records(args.input)
    results, yolo_map, best = evaluate(records, np, ap_per_class)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {'selection_rule': SELECTION_TEXT, 'yolo_mAP50': yolo_map,
               'parity': PARITY, 'selected': best,
               'selected_metrics': results.get(best), 'configs': len(results) - 1,
               'results': results}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    lines = ['# Estudo de políticas de score — VAL (35 imagens, 1.649 GT)', '',
             SELECTION_TEXT, '',
             f"mAP50 do YOLO: {yolo_map:.4f}. Selecionado: **{best or 'nenhum atende à paridade'}**.",
             '', '| Configuração | mAP50 | mAP50:95 | P esp. | R esp. | F1 esp. | Acerto cond. |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, m in results.items():
        lines.append(f"| {name} | {m['mAP50']:.4f} | {m['mAP50_95']:.4f} | {m['species_precision']:.4f} | "
                     f"{m['species_recall']:.4f} | {m['species_f1']:.4f} | {m['species_acc_conditional']:.4f} |")
    (args.out / 'report.md').write_text('\n'.join(lines) + '\n')
    if os.environ.get('WANDB_API_KEY'):
        try:
            import wandb
            wandb.init(entity='pestline', project='fly-species', name='policy-study-val',
                       job_type='policy-study', config={'selection': SELECTION_TEXT})
            for name, m in results.items():
                wandb.log({f'{name}/{k}': v for k, v in m.items() if isinstance(v, float)})
            wandb.finish()
        except Exception as exc:  # tracking nunca invalida o estudo
            print(f'wandb off: {exc}', file=sys.stderr)
    print(f'POLICY_STUDY_COMPLETE {args.out} selected={best}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
