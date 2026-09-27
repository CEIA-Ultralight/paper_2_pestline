#!/usr/bin/env python3
"""Evaluation only: E0 and three fixed, label-independent crop classifiers on VAL.

Run in an externally scheduled, 30-minute h100n2 Apptainer CUDA allocation.
No training, downloads, test split access, dataset writes, or additional NMS.
Imports at module scope are stdlib only; --help and geometry tests are CPU-safe.
Runtime dependencies: torch, torchvision, ultralytics, numpy, Pillow, PyYAML
(already an Ultralytics dependency), and wandb unless explicitly doing a smoke.
Only trusted local checkpoints may be supplied (torch.load uses pickle).
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import traceback
import uuid

RAID = Path('/raid/user_marcospaulo')
RUNS = RAID / 'flydet_runs'
NIGHT = RUNS / 'architecture-night-v1-20260907'
YOLO_CLASSES = ('MD', 'MV', 'MC', 'MF', 'INS', 'NOISE', 'MAR')
CROP_CLASSES = ('INS', 'MAR', 'MC', 'MD', 'MF', 'MV', 'NOISE')
REMAP = tuple(YOLO_CLASSES.index(name) for name in CROP_CLASSES)
HEADS = ('baseline', 'parts', 'arcface')
POLICIES = ('label_only', 'product')
PIPELINES = ('yolo',) + tuple(f'{h}_{p}' for h in HEADS for p in POLICIES)
THRESHOLDS = tuple(0.5 + i * 0.05 for i in range(10))
PREDICT = dict(imgsz=1920, conf=0.001, iou=0.5, max_det=300, rect=True,
               device=0, half=False, verbose=False, augment=False, save=False)
WARNINGS = [
    'VAL only; checkpoints were selected on crop VAL: not an unbiased held-out TEST estimate.',
    'Primary label_only preserves the original class-specific YOLO score, NOT objectness.',
    'Secondary product = YOLO score * max softmax: heuristic, NOT a calibrated posterior; no policy selection.',
    'Training crops used JPEG quality 95; evaluation uses native PIL RGB crops in memory without JPEG recompression.',
    'Pad 0.75 per side, int truncation then clipping; no <8px discard. Degenerate crops retain original class/score.',
    'Original VAL has 35 images / 1649 GT, versus 1645 prior extracted crops; no GT dropped here.',
    'Detection floor .001 / max_det 300 limits localization ceiling. YOLO26 end2end may ignore NMS iou=.5.',
    'AP IoUs .50:.95 are independent of predict iou=.5; confusion uses score>=.25 and class-agnostic IoU>=.5.',
    'All seven classes retained; mAP averages present GT classes; absent class AP is null. Species are MD/MV/MC/MF.',
    'FP32, no AMP or TF32 (training exports used BF16). All models resident: VRAM is aggregate, not isolated.',
    'One timing per decoded image: observational p50/p95 and variation, not repeated benchmarks or full API latency.',
    'Detector timing includes predict preprocessing/postprocessing/H2D. Extra includes crop/transform/batches/softmax/D2H.',
    'End-to-end = detector + one head, excludes shared full-image disk/decode and network/API; warmups excluded.',
    'Resume combines invocation timings; external job limit 30 minutes, internal soft deadline 27 minutes.',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic(path, value, text=False):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(value if text else json.dumps(value, indent=2, allow_nan=False))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def raid_path(value):
    path = Path(value).resolve()
    require(path.is_relative_to(RAID), f'Path outside RAID: {path}')
    return path


GPU_PARTITIONS = tuple(
    p.strip() for p in os.environ.get('FLYDET_GPU_PARTITIONS', 'h100n2,b200n1').split(',') if p.strip()
)


def allocation_guard():
    require(bool(os.environ.get('SLURM_JOB_ID')), 'SLURM_JOB_ID required; no login-node execution')
    require(os.environ.get('SLURM_JOB_PARTITION') in GPU_PARTITIONS,
            f'Requires a GPU partition in {GPU_PARTITIONS}')
    require(any(os.environ.get(k) for k in ('APPTAINER_CONTAINER', 'APPTAINER_NAME')),
            'Requires Apptainer, not a venv or CPU fallback')
    require(int(os.environ.get('WORLD_SIZE', '1')) == 1 and
            int(os.environ.get('SLURM_PROCID', '0')) == 0, 'Single-process evaluation only')


def prepare(args):
    allocation_guard()  # Before heavy imports, cache creation, or output writes.
    require(args.limit_images is None or args.limit_images > 0, '--limit-images must be positive')
    require(not args.no_wandb or args.limit_images is not None, '--no-wandb is smoke-only')
    require(args.no_wandb or os.environ.get('WANDB_MODE', 'online') == 'online', 'W&B must be online')
    require(args.no_wandb or bool(os.environ.get('WANDB_API_KEY')), 'W&B credentials must be inherited via WANDB_API_KEY')
    for key in ('data', 'out', 'detector', 'classifier_root', 'model_source'):
        setattr(args, key, raid_path(getattr(args, key)))
    for source in (args.data, args.detector.parent, args.classifier_root, args.model_source.parent):
        require(not args.out.is_relative_to(source) and not source.is_relative_to(args.out),
                'Output must be separate from data/model/source trees')
    if not args.resume:
        require(not args.out.exists() or not any(args.out.iterdir()), 'Nonempty --out; use --resume')
    else:
        require((args.out / 'state.json').is_file(), '--resume requires state.json')
    args.out.mkdir(parents=True, exist_ok=True)
    for key, suffix in {
        'XDG_CACHE_HOME': 'xdg', 'XDG_CONFIG_HOME': 'config', 'TORCH_HOME': 'torch',
        'HF_HOME': 'hf', 'HF_HUB_CACHE': 'hf/hub', 'HUGGINGFACE_HUB_CACHE': 'hf/hub',
        'TRANSFORMERS_CACHE': 'hf/transformers', 'PIP_CACHE_DIR': 'pip',
        'WANDB_CACHE_DIR': 'wandb/cache', 'WANDB_CONFIG_DIR': 'wandb/config',
        'WANDB_DATA_DIR': 'wandb/data', 'TMPDIR': 'tmp', 'CUDA_CACHE_PATH': 'cuda',
        'TRITON_CACHE_DIR': 'triton', 'TORCHINDUCTOR_CACHE_DIR': 'inductor',
        'MPLCONFIGDIR': 'matplotlib', 'YOLO_CONFIG_DIR': 'ultralytics',
    }.items():
        target = raid_path(RAID / 'cache/yolo_cascade' / suffix)
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    os.environ['WANDB_DIR'] = str(args.out)
    tempfile.tempdir = os.environ['TMPDIR']
    sys.dont_write_bytecode = True


def class_names(names):
    if isinstance(names, dict):
        require(set(names) == set(range(7)), 'Class IDs must be integers 0..6')
        names = [names[i] for i in range(7)]
    require(tuple(names) == YOLO_CLASSES, f'Unexpected YOLO class order: {names}')


def padded_box(box, width=1920, height=1080):
    """Same int()/pad/clipping semantics as extract_crops.py, without min-size=8."""
    require(len(box) == 4 and all(math.isfinite(v) for v in box), 'Nonfinite/invalid box')
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return None
    pw, ph = (x1 - x0) * .75, (y1 - y0) * .75
    crop = max(0, int(x0 - pw)), max(0, int(y0 - ph)), min(width, int(x1 + pw)), min(height, int(y1 + ph))
    return crop if crop[2] > crop[0] and crop[3] > crop[1] else None


def parse_gt(text):
    result = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        require(len(fields) == 5, f'Malformed GT line {number}')
        cls = int(fields[0])
        cx, cy, w, h = map(float, fields[1:])
        require(0 <= cls < 7 and all(math.isfinite(v) for v in (cx, cy, w, h)), 'Invalid GT class/coordinates')
        require(0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1, 'GT must be normalized')
        require(cx - w/2 >= -1e-6 and cy - h/2 >= -1e-6 and
                cx + w/2 <= 1 + 1e-6 and cy + h/2 <= 1 + 1e-6, 'GT outside image')
        result.append({'class': cls, 'box': [(cx-w/2)*1920, (cy-h/2)*1080, (cx+w/2)*1920, (cy+h/2)*1080]})
    return result


def manifest(args, yaml, Image):
    yaml_path = raid_path(args.data / 'data.yaml')
    config = yaml.safe_load(yaml_path.read_text())
    class_names(config['names'])
    root = Path(config.get('path', args.data))
    root = raid_path(root if root.is_absolute() else args.data / root)
    require(root == args.data, 'YAML path must match --data')
    val_root, label_root = args.data / 'images/val', args.data / 'labels/val'
    entries, images = [], []
    val = config['val']  # Never resolve/open train or test.
    for item in val if isinstance(val, list) else [val]:
        path = raid_path(root / item)
        if path.is_file() and path.suffix.lower() == '.txt':
            images.extend(raid_path(path.parent / line.strip()) for line in path.read_text().splitlines() if line.strip())
        else:
            require(path.is_relative_to(val_root) and path.is_dir(), 'VAL directory must be images/val')
            images.extend(p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})
    images = sorted(raid_path(p) for p in images)
    require(images and len(images) == len(set(images)), 'Empty or duplicate VAL images')
    for image in images:
        require(image.is_relative_to(val_root), f'Image outside original VAL: {image}')
        label = raid_path(label_root / image.relative_to(val_root).with_suffix('.txt'))
        require(label.is_relative_to(label_root) and label.is_file(), f'Missing/outside VAL label: {label}')
        gt = parse_gt(label.read_text())
        with Image.open(image) as im:
            require(im.size == (1920, 1080), f'Native dimensions not 1920x1080: {image}')
            im.verify()
        entries.append({'image': str(image), 'label': str(label), 'image_sha256': sha(image),
                        'label_sha256': sha(label), 'gt': gt})
    require(len({e['label'] for e in entries}) == len(entries), 'Images share GT label files')
    return {'yaml_sha256': sha(yaml_path), 'hash_definition': 'Full SHA256 image and label contents, not size/mtime',
            'available_images': len(entries), 'available_gt': sum(len(e['gt']) for e in entries),
            'entries': entries[:args.limit_images]}


def overlap(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = max(0., a[2]-a[0])*max(0., a[3]-a[1]) + max(0., b[2]-b[0])*max(0., b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


def candidates(gt, predictions, class_aware=False):
    return [(g, p, overlap(t['box'], d['box'])) for g, t in enumerate(gt) for p, d in enumerate(predictions)
            if not class_aware or t['class'] == d['class']]


def greedy(pairs, threshold):
    """Ultralytics greedy: descending IoU, unique predictions, then unique GT.

    np.unique(..., return_index=True) returns indices in sorted ID order, NOT
    descending-IoU order. Preserve that detail (no second sort/no reassignment).
    Exact IoU ties use ascending GT/pred IDs for deterministic stdlib execution.
    """
    by_prediction, by_gt = {}, {}
    for g, p, iou in sorted(pairs, key=lambda x: (-x[2], x[0], x[1])):
        if iou >= threshold:
            by_prediction.setdefault(p, (g, p, iou))
    for p in sorted(by_prediction):
        match = by_prediction[p]
        by_gt.setdefault(match[0], match)
    return [by_gt[g] for g in sorted(by_gt)]


def confusion(gt, predictions):
    selected = [p for p in predictions if p['confidence'] >= .25]
    matches = greedy(candidates(gt, selected), .5)
    matrix = [[0]*8 for _ in range(8)]
    used_g, used_p = {g for g, _, _ in matches}, {p for _, p, _ in matches}
    for g, p, _ in matches:
        matrix[gt[g]['class']][selected[p]['class']] += 1
    for g, t in enumerate(gt):
        if g not in used_g:
            matrix[t['class']][7] += 1
    for p, d in enumerate(selected):
        if p not in used_p:
            matrix[7][d['class']] += 1
    return matrix


def relabel(prediction, probabilities):
    if probabilities is None:
        return dict(prediction, probabilities=None, fallback='degenerate_crop')
    require(len(probabilities) == 7 and all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
            and abs(sum(probabilities)-1) < 1e-5, 'Invalid classifier softmax')
    index = max(range(7), key=probabilities.__getitem__)
    mapped = [probabilities[CROP_CLASSES.index(name)] for name in YOLO_CLASSES]
    cls = REMAP[index]
    require(mapped[cls] == max(mapped), 'Classifier argmax remap failed')
    return dict(prediction, **{'class': cls, 'probabilities': mapped, 'fallback': None})


def policy(predictions, name):
    require(name in POLICIES, 'Unknown score policy')
    return [dict(p, confidence=p['yolo_confidence'] *
                 (max(p['probabilities']) if name == 'product' and p['probabilities'] is not None else 1.))
            for p in predictions]


def metrics(records, np, ap_per_class):
    require({'tp', 'conf', 'pred_cls', 'target_cls'} <= set(inspect.signature(ap_per_class).parameters),
            'Unsupported ap_per_class signature')
    targets = [g['class'] for r in records for g in r['gt']]
    present = sorted(set(targets))
    results = {}
    for name in PIPELINES:
        tp, scores, classes = [], [], []
        matrix = np.zeros((8, 8), dtype=np.int64)
        for record in records:
            pred, gt = record['pipelines'][name], record['gt']
            pairs = candidates(gt, pred, class_aware=True)
            correct = [[False]*10 for _ in pred]
            for t, threshold in enumerate(THRESHOLDS):
                for _, p, _ in greedy(pairs, threshold):
                    correct[p][t] = True
            tp.extend(correct)
            scores.extend(p['confidence'] for p in pred)
            classes.extend(p['class'] for p in pred)
            matrix += np.asarray(confusion(gt, pred))
        if present:
            output = ap_per_class(np.asarray(tp, dtype=bool).reshape(-1, 10), np.asarray(scores),
                                  np.asarray(classes, dtype=int), np.asarray(targets, dtype=int), plot=False)
            require(len(output) >= 7, 'Unsupported ap_per_class return contract')
            ap, ids = np.asarray(output[5]), np.asarray(output[6])
            require(ap.shape == (len(present), 10) and ids.tolist() == present and
                    np.isfinite(ap).all() and ((ap >= 0) & (ap <= 1)).all(), 'Unexpected AP shape/classes/range')
        else:
            ap, ids = np.zeros((0, 10)), np.asarray([], dtype=int)  # Explicit empty-GT smoke convention.
        per_class = {label: {'support': targets.count(i), 'AP50': None, 'AP50_95': None} for i, label in enumerate(YOLO_CLASSES)}
        for row, i in enumerate(ids):
            per_class[YOLO_CLASSES[i]].update(AP50=float(ap[row, 0]), AP50_95=float(ap[row].mean()))
        species_rows = [row for row, i in enumerate(ids) if i < 4]
        support, matched = int(matrix[:4].sum()), int(matrix[:4, :7].sum())
        correct = int(sum(matrix[i, i] for i in range(4)))
        results[name] = {'mAP50': float(ap[:, 0].mean()) if present else 0.,
                         'mAP50_95': float(ap.mean()) if present else 0.,
                         'species_mAP50': float(ap[species_rows, 0].mean()) if species_rows else 0.,
                         'species_mAP50_95': float(ap[species_rows].mean()) if species_rows else 0.,
                         'present_class_ids': ids.tolist(), 'per_class': per_class,
                         'confusion': matrix.tolist(), 'species_support': support, 'species_matched': matched,
                         'species_correct': correct, 'species_accuracy_all_gt_including_misses': correct/support if support else 0.,
                         'species_classification_accuracy_conditional_matched': correct/matched if matched else 0.}
    ceiling = []
    for threshold in THRESHOLDS:
        matches = [(r, greedy(candidates(r['gt'], r['pipelines']['yolo']), threshold)) for r in records]
        found = sum(len(m) for _, m in matches)
        species_found = sum(r['gt'][g]['class'] < 4 for r, m in matches for g, _, _ in m)
        species_total = sum(t < 4 for t in targets)
        ceiling.append({'iou': threshold, 'matched_gt': found, 'recall': found/len(targets) if targets else 0.,
                        'species_recall': species_found/species_total if species_total else 0.})
    return {'pipelines': results, 'localization_ceiling_conf_001_class_agnostic': ceiling,
            'images': len(records), 'gt_objects': len(targets), 'ap_per_class_signature': str(inspect.signature(ap_per_class)),
            'ap_return_indices': {'ap': 5, 'class_ids': 6}, 'iou_thresholds': list(THRESHOLDS),
            'confusion_axes': {'rows': 'GT', 'columns': 'prediction', 'labels': [*YOLO_CLASSES, 'background']}}


def validate_metadata(metadata, head, source_sha):
    require(tuple(metadata['class_order']) == CROP_CLASSES and
            metadata['class_to_idx'] == dict(zip(CROP_CLASSES, range(7))), 'Classifier classes differ')
    require(metadata['source']['file_sha256']['models.py'] == source_sha, f'{head}: source SHA mismatch')
    require(metadata['architecture']['variant'] == head and metadata['args']['variant'] == head, 'Variant mismatch')
    pre = metadata['preprocessing']
    require(pre['resize'] == [384, 384] and pre['interpolation'] == 'bilinear' and pre['antialias'] is True
            and 'pad75' in pre['input'], 'Unsupported preprocessing/padding')
    norm = pre['normalization']
    require(len(norm['mean']) == len(norm['std']) == 3 and
            all(math.isfinite(x) for x in norm['mean'] + norm['std']) and min(norm['std']) > 0, 'Invalid normalization')


def load_heads(args, summaries, torch, transforms):
    source_sha = sha(args.model_source)
    checkpoints = {}
    for head in HEADS:
        checkpoint = torch.load(args.classifier_root / head / 'best.pt', map_location='cpu', weights_only=False)
        meta, summary = checkpoint['metadata'], summaries[head]
        validate_metadata(meta, head, source_sha)
        require(digest(meta) == digest(summary['metadata']), f'{head}: checkpoint/summary metadata mismatch')
        require(checkpoint['epoch'] == summary['exported_epoch'] and checkpoint['epoch'] > 0 and
                not checkpoint['is_initial_fallback'] and summary['export_complete'], 'Incomplete/initial checkpoint')
        checkpoints[head] = checkpoint
    spec = importlib.util.spec_from_file_location('cascade_pinned_models', args.model_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Only after every checkpoint's source SHA is validated.
    heads = {}
    for head, checkpoint in checkpoints.items():
        meta = checkpoint['metadata']
        model = module.build_model(head, 7, pretrained=False).float().eval()
        actual = dict(model.architecture_metadata, pretrained=meta['architecture']['pretrained'])
        require(digest(actual) == digest(meta['architecture']), 'Architecture reconstruction mismatch')
        model.load_state_dict(checkpoint['model'], strict=True)
        if head == 'arcface':
            require(model.scale == meta['training']['arcface_scale'], 'ArcFace inference scale mismatch')
        norm = meta['preprocessing']['normalization']
        transform = transforms.Compose([transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BILINEAR,
                                                           antialias=True), transforms.ToTensor(),
                                        transforms.Normalize(norm['mean'], norm['std'])])
        heads[head] = (model.cuda(), transform)
    return heads


def classify(image, predictions, head, torch):
    model, transform = head
    output = [relabel(p, None) for p in predictions]
    valid = [(i, padded_box(p['box'])) for i, p in enumerate(predictions)]
    valid = [(i, box) for i, box in valid if box is not None]
    for start in range(0, len(valid), 32):
        batch = valid[start:start+32]
        tensor = torch.stack([transform(image.crop(box)) for _, box in batch]).to('cuda', dtype=torch.float32)
        logits = model(tensor)['logits']  # Never supply GT, labels, or ArcFace training margin.
        require(logits.shape == (len(batch), 7) and logits.dtype == torch.float32 and
                torch.isfinite(logits).all().item(), 'Invalid FP32 inference logits')
        probabilities = logits.softmax(dim=1).cpu().tolist()
        for (i, _), probs in zip(batch, probabilities):
            output[i] = relabel(predictions[i], probs)
    return output


def timed(torch, function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def validate_record(record, entry, fingerprint):
    require(record['fingerprint'] == fingerprint and record['input'] == entry and record['gt'] == entry['gt'],
            'Image checkpoint identity mismatch')
    require(record['probability_class_order'] == list(YOLO_CLASSES), 'Probability class order changed')
    pipelines = record['pipelines']
    require(set(pipelines) == set(PIPELINES), 'Missing pipelines')
    original = pipelines['yolo']
    require(len(original) <= 300, 'Detector exceeded max_det')
    for name, predictions in pipelines.items():
        require(len(predictions) == len(original), 'Cascade dropped boxes')
        for p, old in zip(predictions, original):
            require(p['box'] == old['box'] and p['old_class'] == old['class'] and
                    p['yolo_confidence'] == old['confidence'], 'Cascade changed geometry/original identity')
            require(isinstance(p['class'], int) and 0 <= p['class'] < 7 and math.isfinite(p['confidence'])
                    and 0 <= p['confidence'] <= 1 and len(p['box']) == 4
                    and all(math.isfinite(x) for x in p['box']), 'Invalid prediction')
            if name != 'yolo':
                require((p['probabilities'] is None) == (padded_box(p['box']) is None), 'Invalid crop fallback')
                source_probs = None if p['probabilities'] is None else [p['probabilities'][i] for i in REMAP]
                expected = policy([relabel(old, source_probs)], 'product' if name.endswith('_product') else 'label_only')[0]
                require(p == expected, 'Invalid remapped class/score/fallback')
            else:
                require(p['probabilities'] is None and p['fallback'] is None and p['confidence'] >= .001-1e-8,
                        'Invalid original YOLO output')
    require(set(record['seconds']) == {'detector', *HEADS} and
            all(math.isfinite(v) and v >= 0 for v in record['seconds'].values()), 'Invalid latency')


def latency(records, np):
    result = {}
    for name in ('detector', *HEADS):
        extra = [r['seconds'][name] for r in records]
        total = extra if name == 'detector' else [v+r['seconds']['detector'] for v, r in zip(extra, records)]
        result[name] = {kind: {'n': len(values), 'mean': float(np.mean(values)), 'std': float(np.std(values)),
                               'p50': float(np.percentile(values, 50)), 'p95': float(np.percentile(values, 95)),
                               'min': min(values), 'max': max(values)}
                        for kind, values in (('extra_seconds' if name != 'detector' else 'predict_seconds', extra),
                                             ('decoded_image_e2e_seconds', total))}
    return result


def publish(run, summary, out, wandb, Image, ImageDraw):
    values = {}
    for name, result in summary['metrics']['pipelines'].items():
        values.update({f'{name}/{k}': v for k, v in result.items() if isinstance(v, (int, float))})
        for cls, measures in result['per_class'].items():
            values.update({f'{name}/{cls}/{k}': v for k, v in measures.items() if v is not None})
        canvas = Image.new('RGB', (850, 650), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), f'{name}: rows GT / columns prediction; conf >= .25, IoU >= .5', fill='black')
        labels = (*YOLO_CLASSES, 'background')
        for i, label in enumerate(labels):
            draw.text((10, 90+i*65), label, fill='black')
            draw.text((130+i*88, 50), label, fill='black')
            for j, count in enumerate(result['confusion'][i]):
                draw.rectangle((125+j*88, 80+i*65, 210+j*88, 140+i*65), outline='gray', fill='#dceeff' if i == j else 'white')
                draw.text((140+j*88, 100+i*65), str(count), fill='black')
        values[f'{name}/confusion'] = wandb.Image(canvas)
    run.log(values)
    run.summary['evaluation'] = summary
    for pattern in ('summary.json', 'report.md', 'state.json', 'images/*.json'):
        run.save(str(out / pattern), base_path=str(out), policy='now')


def evaluate(args, stop, started):
    allocation_guard()  # Also protect direct calls that bypass the CLI.
    import torch  # Strict environment guard already passed; CUDA checked before other heavy imports.
    require(torch.cuda.is_available(), 'CUDA required; CPU fallback forbidden')
    torch.cuda.set_device(0)
    import numpy as np
    import torchvision
    from torchvision import transforms
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils.metrics import ap_per_class
    from PIL import Image, ImageDraw
    import yaml

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    data = manifest(args, yaml, Image)
    source_sha = sha(args.model_source)
    summaries, identities = {}, {}
    for head in HEADS:
        folder = args.classifier_root / head
        summaries[head] = read_json(raid_path(folder / 'summary.json'))
        validate_metadata(summaries[head]['metadata'], head, source_sha)
        identities[head] = {name: {'path': str(raid_path(folder / name)), 'sha256': sha(raid_path(folder / name))}
                            for name in ('summary.json', 'best.pt')}
    versions = {'torch': str(torch.__version__), 'torchvision': torchvision.__version__,
                'ultralytics': ultralytics.__version__, 'numpy': np.__version__, 'Pillow': Image.__version__,
                'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}
    contract = {'schema': 1, 'evaluator_sha256': sha(__file__), 'data': data, 'heads': identities,
                'model_source': {'path': str(args.model_source), 'sha256': source_sha},
                'detector': {'path': str(args.detector), 'sha256': sha(args.detector)}, 'predict': PREDICT,
                'versions': versions, 'classes': YOLO_CLASSES, 'crop_classes': CROP_CLASSES,
                'batch': 32, 'dtype': 'float32', 'pad': .75, 'policies': POLICIES,
                'limit_images': args.limit_images, 'wandb': not args.no_wandb, 'warnings': WARNINGS}
    fingerprint = digest(contract)
    state_path = args.out / 'state.json'
    state = read_json(state_path) if args.resume else {
        'fingerprint': fingerprint, 'contract': contract, 'wandb_id': uuid.uuid4().hex[:8],
        'run_name': 'EVAL_E0_cascade_val_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'), 'attempts': []}
    require(state['fingerprint'] == fingerprint and digest(state['contract']) == fingerprint, 'Resume contract changed; use new --out')
    image_dir = raid_path(args.out / 'images')
    require(image_dir.is_relative_to(args.out), 'Image output symlink escapes --out')
    image_dir.mkdir(exist_ok=True)
    records, run, wandb, code, error, summary = [], None, None, 1, None, {}
    attempt = {'job_id': os.environ['SLURM_JOB_ID'], 'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'container': os.environ.get('APPTAINER_CONTAINER', os.environ.get('APPTAINER_NAME'))}
    state['attempts'].append(attempt)
    state['status'] = 'running'
    atomic(state_path, state)  # Persist W&B ID BEFORE contacting W&B or loading weights.
    try:
        if not args.no_wandb:
            import wandb
            run = wandb.init(entity='pestline', project='fly-species', id=state['wandb_id'],
                             name=state['run_name'], resume='allow', dir=str(args.out), config=contract)
            require(run is not None and not getattr(run, 'disabled', False), 'W&B initialization failed')
        heads = load_heads(args, summaries, torch, transforms)
        detector = YOLO(str(args.detector))
        detector.model.float().eval()
        class_names(detector.names)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for _ in range(2):
                detector.predict(source=np.zeros((1080, 1920, 3), dtype=np.uint8), **PREDICT)
            for model, _ in heads.values():
                for batch in (1, 32):
                    model(torch.zeros((batch, 3, 384, 384), device='cuda', dtype=torch.float32))['logits'].softmax(1).cpu()
            torch.cuda.synchronize()
            backend = detector.predictor.model
            attempt['backend'] = {'type': type(backend).__name__, 'end2end': getattr(backend, 'end2end', None),
                                   'model_end2end': getattr(detector.model, 'end2end', None), 'fp16': getattr(backend, 'fp16', None)}
            require(not getattr(backend, 'fp16', False), 'Detector silently enabled FP16')
            for i, entry in enumerate(data['entries']):
                if stop[0] or time.monotonic()-started >= 27*60:
                    code = 75
                    break
                path = image_dir / f'{i:05d}.json'
                if path.exists():
                    saved = read_json(path)
                    require(saved['sha256'] == digest(saved['record']), 'Corrupt image checkpoint; refusing overwrite')
                    record = saved['record']
                    validate_record(record, entry, fingerprint)
                    records.append(record)
                    continue
                require(sha(entry['image']) == entry['image_sha256'] and sha(entry['label']) == entry['label_sha256'], 'Input changed mid-run')
                with Image.open(entry['image']) as raw:
                    require(raw.size == (1920, 1080), 'Native dimensions changed')
                    image = raw.convert('RGB')
                bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
                outputs, seconds = timed(torch, lambda: detector.predict(source=bgr, **PREDICT))
                require(len(outputs) == 1 and tuple(outputs[0].orig_shape) == (1080, 1920), 'Unexpected detector output')
                boxes = outputs[0].boxes
                require(boxes is not None and len(boxes.xyxy) == len(boxes.cls) == len(boxes.conf) == len(boxes)
                        and len(boxes) <= 300, 'Invalid detector box arrays/count')
                predictions = []
                for box, cls, score in zip(boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()):
                    require(cls == int(cls) and 0 <= int(cls) < 7 and math.isfinite(score) and 0 <= score <= 1, 'Invalid YOLO class/score')
                    predictions.append({'box': box, 'class': int(cls), 'confidence': score, 'old_class': int(cls),
                                        'yolo_confidence': score, 'probabilities': None, 'fallback': None})
                record = {'fingerprint': fingerprint, 'input': entry, 'gt': entry['gt'], 'job_id': attempt['job_id'],
                          'pipelines': {'yolo': predictions}, 'seconds': {'detector': seconds},
                          'probability_class_order': list(YOLO_CLASSES)}
                for head in HEADS:
                    classified, extra = timed(torch, lambda: classify(image, predictions, heads[head], torch))
                    record['seconds'][head] = extra
                    for score_policy in POLICIES:
                        record['pipelines'][f'{head}_{score_policy}'] = policy(classified, score_policy)
                validate_record(record, entry, fingerprint)
                atomic(path, {'sha256': digest(record), 'record': record})  # Complete image is the transaction boundary.
                records.append(record)
                state['completed_images'] = len(records)
                atomic(state_path, state)
                if run is not None:
                    run.log({'progress/images': len(records)})
            else:
                code = 0
        attempt['vram'] = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                            'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'all_models_resident': True}
        if stop[0]:
            code = 75
        if code == 0:
            summary = {'metrics': metrics(records, np, ap_per_class), 'latency': latency(records, np),
                       'fallback_boxes': {h: sum(p['fallback'] is not None for r in records for p in r['pipelines'][f'{h}_label_only']) for h in HEADS}}
            if args.limit_images is None:
                require(len(records) == 35 and summary['metrics']['gt_objects'] == 1649, 'Original VAL census changed; not 35/1649')
    except Exception as exc:
        error, code = f'{type(exc).__name__}: {exc}', 1
        traceback.print_exc()
    if stop[0]:
        code = 75
    summary.update(status='completed' if code == 0 else 'interrupted' if code == 75 else 'failed', exit_code=code,
                   error=error, fingerprint=fingerprint, wandb_id=state['wandb_id'], warnings=WARNINGS,
                   smoke=args.limit_images is not None, versions=versions, completed_images=len(records),
                   available_images=data['available_images'], available_gt=data['available_gt'], attempts=state['attempts'])
    try:
        attempt.update(elapsed_seconds=time.monotonic()-started, exit_code=code, error=error)
        state.update(status=summary['status'], completed_images=len(records))
        atomic(state_path, state)
        if code == 0:
            lines = ['# E0 cascade — original VAL (no training)', '', '| Pipeline | AP50 | AP50:95 | Species AP50:95 |', '|---|---:|---:|---:|']
            for name, m in summary['metrics']['pipelines'].items():
                lines.append(f"| {name} | {m['mAP50']:.6f} | {m['mAP50_95']:.6f} | {m['species_mAP50_95']:.6f} |")
            lines += ['', f"Images: {len(records)}; GT: {summary['metrics']['gt_objects']}; smoke: {args.limit_images is not None}.", '', *['- '+w for w in WARNINGS]]
            atomic(args.out / 'report.md', '\n'.join(lines)+'\n', text=True)
        atomic(args.out / 'summary.json', summary)
        if run is not None:
            if code == 0:
                publish(run, summary, args.out, wandb, Image, ImageDraw)
            if stop[0]:
                code = 75
            run.summary['status'] = 'interrupted' if code == 75 else summary['status']
            run.finish(exit_code=1 if code else 0)
    except Exception as exc:
        code, error = 1, f'Finalization: {type(exc).__name__}: {exc}'
        if run is not None:
            try:
                run.finish(exit_code=1)
            except Exception:
                traceback.print_exc()
    if stop[0]:
        code = 75
    attempt.update(elapsed_seconds=time.monotonic()-started, exit_code=code, error=error)
    state.update(status='completed' if code == 0 else 'interrupted' if code == 75 else 'failed', completed_images=len(records))
    summary.update(status=state['status'], exit_code=code, error=error)
    atomic(state_path, state)
    atomic(args.out / 'summary.json', summary)
    return code


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--data', default=str(RAID / 'datasets/DS-F2_v8.1.1'))
    parser.add_argument('--detector', default=str(RUNS / 'E0_yolo26m_baseline/runs/yolo26m.pt/weights/best.pt'))
    parser.add_argument('--classifier-root', default=str(NIGHT / 'crops'))
    parser.add_argument('--model-source', default=str(NIGHT / 'source/fly-det/scripts/architecture_lab/models.py'))
    parser.add_argument('--limit-images', type=int, help='Smoke: first N lexically sorted original VAL images')
    parser.add_argument('--no-wandb', action='store_true', help='Allowed only with --limit-images')
    parser.add_argument('--resume', action='store_true', help='Verify content/source contract and reuse W&B ID + atomic images')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    stop, started = [None], time.monotonic()
    handlers = {s: signal.signal(s, lambda number, _frame: stop.__setitem__(0, number))
                for s in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
    try:
        prepare(args)
        with (args.out / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # A concurrent launcher may have passed the initial empty-directory check.
            require(args.resume or not (args.out / 'state.json').exists(), 'Existing evaluation; use --resume')
            return evaluate(args, stop, started)
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)


if __name__ == '__main__':
    raise SystemExit(main())