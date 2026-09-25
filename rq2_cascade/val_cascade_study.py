#!/usr/bin/env python3
"""Generic VAL cascade evaluation: any detector x any discovered head.

Reuses eval_yolo_cascade metrics/matching and test_campaign_heads adapters.
VAL only (35 original images / 1649 GT); the test split is never opened.
Selection happens here; TEST evaluation is a separate, approved step.

Stdlib at module scope; heavy imports and CUDA inside Slurm h100n2 Apptainer.
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
RUNS = RAID / 'flydet_runs'
YOLO_CLASSES = ('MD', 'MV', 'MC', 'MF', 'INS', 'NOISE', 'MAR')
POLICIES = ('label_only', 'product')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sibling(name):
    import types
    path = Path(__file__).resolve().with_name(name + '.py')
    module = types.ModuleType('_vcascade_' + name)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


EV = sibling('eval_yolo_cascade')
HEADS_MOD = sibling('test_campaign_heads')


def guard():
    EV.allocation_guard()


def entries_val(Image, yaml):
    data = RAID / 'datasets/DS-F2_v8.1.1'
    config = yaml.safe_load((data / 'data.yaml').read_text())
    EV.class_names(config['names'])
    entries = []
    for path in sorted((data / 'images/val').rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        label = (data / 'labels/val' / path.relative_to(data / 'images/val')).with_suffix('.txt')
        gt = EV.parse_gt(label.read_text())
        with Image.open(path) as im:
            require(im.size == (1920, 1080), f'non-native size {path}')
            im.verify()
        entries.append({'image': str(path), 'gt': gt,
                        'image_sha256': EV.sha(path), 'label_sha256': EV.sha(label)})
    require(len(entries) == 35 and sum(len(e['gt']) for e in entries) == 1649,
            'Expected VAL 35 images / 1649 GT')
    return entries


def detector_predictions(YOLO, Image, torch, checkpoint, entries):
    model = YOLO(str(checkpoint))
    EV.class_names(model.names)
    model.model.float().eval().requires_grad_(False)
    rows = []
    with Image.open(entries[0]['image']) as warm:  # warmup excluído das métricas
        model.predict(source=warm.convert('RGB'), **EV.PREDICT)
    for entry in entries:
        with Image.open(entry['image']) as image:
            result = model.predict(source=image.convert('RGB'), **EV.PREDICT)[0]
        preds = []
        for box, cls, score in zip(result.boxes.xyxy.cpu().tolist(),
                                   result.boxes.cls.cpu().tolist(),
                                   result.boxes.conf.cpu().tolist()):
            preds.append({'box': box, 'class': int(cls), 'confidence': float(score),
                          'yolo_confidence': float(score)})
        rows.append(preds)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--detector', type=Path, required=True)
    ap.add_argument('--heads', nargs='*', default=[],
                    help='Head spec ids from test_campaign_heads (empty = detector only)')
    ap.add_argument('--campaign-roots', nargs='*', default=None)
    args = ap.parse_args(argv)
    guard()
    require(args.detector.is_file(), f'detector missing: {args.detector}')
    require(args.out.is_absolute() and args.out.resolve().is_relative_to(RAID), '--out on RAID')

    import numpy as np
    import torch
    import yaml
    from PIL import Image
    from ultralytics import YOLO
    from ultralytics.utils.metrics import ap_per_class
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    entries = entries_val(Image, yaml)
    allow_missing = args.campaign_roots is not None
    specs = HEADS_MOD.discover_heads(campaign_roots=args.campaign_roots,
                                     allow_missing_campaigns=allow_missing)
    by_id = {s['id']: s for s in specs}
    # Aliases curtos para as campanhas novas
    aliases = {
        'dinov2_42': 'dinov2-20260909/seed42', 'dinov2_84': 'dinov2-20260909/seed84',
        'propcls_42': 'proposal-cls-20260909/seed42', 'propcls_84': 'proposal-cls-20260909/seed84',
        'parts': 'architecture-night-v1-20260907/crops/parts',
    }
    chosen = []
    for h in args.heads:
        hid = aliases.get(h, h)
        require(hid in by_id, f'unknown head {h} (have {len(by_id)})')
        spec = by_id[hid]
        require(spec['supports_cascade'], f'{hid} does not support cascade')
        chosen.append(spec)

    args.out.mkdir(parents=True, exist_ok=True)
    records = [{'gt': e['gt'], 'pipelines': {}} for e in entries]
    preds = detector_predictions(YOLO, Image, torch, args.detector, entries)
    for rec, p in zip(records, preds):
        rec['pipelines']['yolo'] = p

    def measure():
        prev = EV.PIPELINES
        try:
            EV.PIPELINES = tuple(records[0]['pipelines'])
            return EV.metrics(records, np, ap_per_class)['pipelines']
        finally:
            EV.PIPELINES = prev

    results = {'detector': str(args.detector), 'pipelines': {}}
    results['pipelines']['yolo'] = measure()['yolo']
    print('yolo mAP50', round(results['pipelines']['yolo']['mAP50'], 4), flush=True)

    for spec in chosen:
        adapter = HEADS_MOD.load_head(spec)
        try:
            for rec, entry in zip(records, entries):
                base = rec['pipelines']['yolo']
                probs = [None] * len(base)
                valid = [(i, EV.padded_box(p['box'])) for i, p in enumerate(base)]
                valid = [(i, b) for i, b in valid if b is not None]
                from PIL import Image as Im
                with Im.open(entry['image']) as source:
                    image = source.convert('RGB')
                    for start in range(0, len(valid), spec['batch_size']):
                        batch = valid[start:start + spec['batch_size']]
                        crops = [image.crop(b) for _, b in batch]
                        got = adapter.predict(crops)
                        for (i, _), pr in zip(batch, got):
                            probs[i] = pr
                for policy in POLICIES:
                    name = f"{spec['id']}_{policy}"
                    rec['pipelines'][name] = [
                        dict(p, **{'class': (max(range(7), key=pr.__getitem__) if pr else p['class']),
                                   'confidence': (p['yolo_confidence'] * (max(pr) if policy == 'product' and pr else 1.)),
                                   'probabilities': pr})
                        for p, pr in zip(base, probs)]
        finally:
            adapter.close()
        m = measure()
        for policy in POLICIES:
            name = f"{spec['id']}_{policy}"
            results['pipelines'][name] = m[name]
            print(name, 'mAP50', round(m[name]['mAP50'], 4),
                  'speciesF1', round(m[name]['species_macro_f1'], 4)
                  if 'species_macro_f1' in m[name] else '', flush=True)
        # limpa pipelines da cabeça para o próximo ciclo de medição
        for rec in records:
            for policy in POLICIES:
                rec['pipelines'].pop(f"{spec['id']}_{policy}", None)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'summary.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    print(f'VAL_STUDY_COMPLETE {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
