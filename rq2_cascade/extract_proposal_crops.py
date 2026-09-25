#!/usr/bin/env python3
"""Extraction only: E0 DETECTOR PROPOSALS (not GT boxes) as an ImageFolder crop dataset.

Builds DS-F2_crops_proposals from the train and val splits of DS-F2_v8.1.1 by running
the E0 YOLO detector per image (same predict semantics as eval_yolo_cascade.PREDICT)
and saving each padded proposal crop under <out>/<split>/<CLASS>/<image_stem>_<NNN>.jpg.
Labels NEVER come from the detector's predicted class: each proposal is matched
greedily (class-agnostic IoU, eval_yolo_cascade.greedy/candidates semantics) against
GT; best IoU >= 0.3 assigns that GT class, otherwise the crop is BG (an explicit 8th
rejection class so a classifier learns to reject false positives). NNN is the
zero-padded (3+ digits) index of the proposal in the YOLO prediction list; skipped
tiny/degenerate crops leave gaps in the numbering.

Run in an externally scheduled, 2-hour h100n2 Apptainer CUDA allocation.
No training, downloads, test split access, or source dataset writes.
Imports at module scope are stdlib only (plus the stdlib-only eval_yolo_cascade
sibling); --help and the pure logic unit tests are CPU-safe.
Runtime dependencies: torch, ultralytics, numpy, Pillow, PyYAML.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback


def _load_cascade():
    source = Path(__file__).resolve().with_name('eval_yolo_cascade.py')
    spec = importlib.util.spec_from_file_location('eval_yolo_cascade', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Stdlib-only module; safe outside Slurm.
    return module


cascade = _load_cascade()
require, digest, sha, atomic, raid_path = (cascade.require, cascade.digest, cascade.sha,
                                           cascade.atomic, cascade.raid_path)
parse_gt, padded_box, candidates, greedy = (cascade.parse_gt, cascade.padded_box,
                                            cascade.candidates, cascade.greedy)
allocation_guard = cascade.allocation_guard
PREDICT = cascade.PREDICT
YOLO_CLASSES = cascade.YOLO_CLASSES

RAID = Path('/raid/user_marcospaulo')
DEFAULT_DATA = RAID / 'datasets/DS-F2_v8.1.1'
DEFAULT_DETECTOR = RAID / 'flydet_runs/E0_yolo26m_baseline/runs/yolo26m.pt/weights/best.pt'
DEFAULT_OUT = RAID / 'datasets/DS-F2_crops_proposals'
CLASSES = ('BG', 'INS', 'MAR', 'MC', 'MD', 'MF', 'MV', 'NOISE')  # Exactly 8, alphabetical.
SPLITS = ('train', 'val')  # Test split is forbidden.
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
PAD = .75
MIN_SIZE = 8
IOU_MATCH = .3
JPEG_QUALITY = 95
WARNINGS = [
    'Crops come from E0 detector PROPOSALS, not GT boxes: per-class counts differ from GT-crop datasets '
    '(DS-F2_crops*). One GT matches at most one proposal; unmatched GT yields no crop; unmatched proposals become BG.',
    'E0 was trained on train only; VAL was used for checkpoint/model selection, so VAL proposal statistics are '
    'not a fully held-out estimate (accepted and documented; no train->test leakage, test is never read).',
    'Labels come from greedy class-agnostic IoU>=0.3 matching against GT; the detector predicted class is never used.',
    'BG is an explicit 8th rejection class; species counting on this dataset differs from GT-crop counting.',
    'Proposal ordinal NNN is the index in the YOLO prediction list (conf>=.001, max_det=300); skipped tiny/degenerate '
    'crops leave gaps in the per-image numbering.',
    'Pad 0.75 per side with int() truncation then clipping (extract_crops.py/eval_yolo_cascade semantics); crops '
    '<8px in either dimension are skipped and counted in skipped_tiny.',
    'Original images and labels are opened read-only; nothing under the source dataset is created or modified.',
    'FP32, no AMP/TF32; deterministic proposal ordering is the order YOLO returns predictions.',
]


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def assign_labels(gt, boxes, threshold=IOU_MATCH):
    """Greedy class-agnostic IoU matching (cascade semantics); unmatched proposals -> 'BG'."""
    predictions = []
    for box in boxes:
        require(len(box) == 4 and all(math.isfinite(v) for v in box), 'Nonfinite/invalid proposal box')
        predictions.append({'box': list(box)})
    labels = ['BG'] * len(boxes)
    for g, p, _iou in greedy(candidates(gt, predictions), threshold):
        labels[p] = YOLO_CLASSES[gt[g]['class']]
    return labels


def plan_crop(box, width=1920, height=1080):
    """Padded box (pad=0.75, int truncation, clip); None for degenerate or <8px crops."""
    crop = padded_box(box, width, height)
    if crop is None:
        return None
    if (crop[2] - crop[0]) < MIN_SIZE or (crop[3] - crop[1]) < MIN_SIZE:
        return None
    return crop


def process_image(image, boxes, gt, out_split_dir, stem, write_crop):
    """Crop every proposal of one image; returns per-image counts including skipped_tiny."""
    require(stem == Path(stem).name and bool(stem), 'Unsafe image stem')
    width, height = image.size
    labels = assign_labels(gt, boxes)
    counts = {name: 0 for name in CLASSES}
    skipped = 0
    for index, (box, label) in enumerate(zip(boxes, labels)):
        crop = plan_crop(box, width, height)
        if crop is None:
            skipped += 1
            continue
        write_crop(image.crop(crop), Path(out_split_dir) / label / f'{stem}_{index:03d}.jpg')
        counts[label] += 1
    return {'proposals': len(boxes), 'crops': sum(counts.values()), 'skipped_tiny': skipped,
            'counts': counts}


def run_image(predict_fn, image, gt, out_split_dir, stem, write_crop):
    """Seam for tests: predict_fn(image) -> proposal boxes in deterministic YOLO order."""
    return process_image(image, predict_fn(image), gt, out_split_dir, stem, write_crop)


def jpeg_writer():
    def write(crop, target):
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f'.{target.name}.{os.getpid()}.tmp')
        crop.save(str(temporary), 'JPEG', quality=JPEG_QUALITY)
        os.replace(temporary, target)
    return write


def contract_hash(script_path):
    """Resume contract: script + extraction/predict semantics (not the source commit)."""
    return digest({'schema': 1, 'script_sha256': sha(script_path), 'classes': CLASSES,
                   'splits': SPLITS, 'pad': PAD, 'min_size': MIN_SIZE, 'iou_match': IOU_MATCH,
                   'jpeg_quality': JPEG_QUALITY, 'predict': PREDICT})


def marker_valid(marker, identity, contract_id):
    return (isinstance(marker, dict) and marker.get('schema') == 1 and
            marker.get('status') == 'complete' and marker.get('contract') == contract_id and
            marker.get('identity') == identity and
            isinstance(marker.get('result'), dict) and
            set(marker['result'].get('counts', {})) == set(CLASSES))


def new_split_stats():
    return {'images': 0, 'proposals': 0, 'crops': 0, 'skipped_tiny': 0,
            'per_class': {name: 0 for name in CLASSES}, 'per_image': {}}


def merge_stats(stats, stem, result):
    require(set(result['counts']) == set(CLASSES), 'Result class keys changed')
    require(result['crops'] == sum(result['counts'].values()), 'Inconsistent result counts')
    require(stem not in stats['per_image'], f'Duplicate image stem: {stem}')
    stats['images'] += 1
    stats['proposals'] += result['proposals']
    stats['crops'] += result['crops']
    stats['skipped_tiny'] += result['skipped_tiny']
    for name, count in result['counts'].items():
        stats['per_class'][name] += count
    stats['per_image'][stem] = result
    return stats


def build_manifest(splits, context):
    return {'schema': 1,
            'created_utc': utcnow(),
            'status': context['status'],
            'error': context.get('error'),
            'source_commit': context['source_commit'],
            'script_sha256': context['script_sha256'],
            'detector': {'path': context['detector_path'], 'sha256': context['detector_sha256']},
            'data': {'path': context['data_path'], 'data_yaml_sha256': context['data_yaml_sha256']},
            'classes': list(CLASSES),
            'class_semantics': 'Labels from greedy class-agnostic IoU>=0.3 GT matching; detector '
                               'predicted class never used; unmatched proposals are BG.',
            'thresholds': {'iou_match': IOU_MATCH, 'pad': PAD, 'min_size': MIN_SIZE,
                           'jpeg_quality': JPEG_QUALITY, 'predict': dict(PREDICT)},
            'splits': splits,
            'warnings': WARNINGS}


def split_entries(data, split, Image):
    """Enumerate one split (images + labels + hashes + parsed GT); never the test split."""
    require(split in SPLITS, 'Only train/val; the test split is forbidden')
    image_dir = raid_path(data / 'images' / split)
    label_dir = raid_path(data / 'labels' / split)
    require(image_dir.is_dir() and label_dir.is_dir(), f'Missing {split} image/label directories')
    images = sorted(p for p in image_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMG_EXTS)
    require(images and len(images) == len({p.stem for p in images}), 'Empty or duplicate stems')
    entries = []
    for image in images:
        require(image.stem == Path(image.stem).name and bool(image.stem), 'Unsafe image stem')
        label = label_dir / f'{image.stem}.txt'
        require(label.is_file(), f'Missing label: {label}')
        gt = parse_gt(label.read_text())
        with Image.open(image) as im:
            require(im.size == (1920, 1080), f'Native dimensions not 1920x1080: {image}')
            im.verify()
        entries.append({'image': str(image), 'label': str(label), 'stem': image.stem,
                        'image_sha256': sha(image), 'label_sha256': sha(label), 'gt': gt})
    return entries


def execute(args, stop, started):
    allocation_guard()  # Also protect direct calls that bypass the CLI.
    import torch  # Strict environment guard already passed; CUDA checked before other heavy imports.
    require(torch.cuda.is_available(), 'CUDA required; CPU fallback forbidden')
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    import numpy as np
    import yaml
    from PIL import Image
    from ultralytics import YOLO

    commit = os.environ.get('FLYDET_SOURCE_COMMIT', 'unknown')
    yaml_path = raid_path(args.data / 'data.yaml')
    config = yaml.safe_load(yaml_path.read_text())
    cascade.class_names(config['names'])
    yaml_sha = sha(yaml_path)
    detector_sha = sha(args.detector)
    contract_id = contract_hash(Path(__file__).resolve())
    script_sha = sha(Path(__file__).resolve())
    detector = YOLO(str(args.detector))
    detector.model.float().eval()
    cascade.class_names(detector.names)
    marker_root = args.out / '.markers'
    splits, code, error = {}, 1, None
    try:
        with torch.inference_mode():
            for _ in range(2):  # Warmup outside any per-image accounting.
                detector.predict(source=np.zeros((1080, 1920, 3), dtype=np.uint8), **PREDICT)
            interrupted = False
            for split in SPLITS:  # train first, then val.
                entries = split_entries(args.data, split, Image)
                if args.limit_images is not None:
                    entries = entries[:args.limit_images]
                stats = new_split_stats()
                marker_dir = marker_root / split
                marker_dir.mkdir(parents=True, exist_ok=True)
                for entry in entries:
                    if stop[0] is not None:
                        interrupted = True
                        break
                    identity = {'image_sha256': entry['image_sha256'],
                                'label_sha256': entry['label_sha256'],
                                'detector_sha256': detector_sha}
                    marker_path = marker_dir / f"{entry['stem']}.json"
                    if marker_path.exists():  # Resume: a completed image is the transaction boundary.
                        marker = cascade.read_json(marker_path)
                        if marker_valid(marker, identity, contract_id):
                            merge_stats(stats, entry['stem'], marker['result'])
                            continue
                        require(False, f'Stale/corrupt marker: {marker_path}; remove .markers to rebuild')
                    require(sha(entry['image']) == entry['image_sha256'] and
                            sha(entry['label']) == entry['label_sha256'], 'Input changed mid-run')
                    with Image.open(entry['image']) as raw:
                        require(raw.size == (1920, 1080), 'Native dimensions changed')
                        image = raw.convert('RGB')
                    bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])

                    def predict(_image, _bgr=bgr):
                        outputs = detector.predict(source=_bgr, **PREDICT)
                        require(len(outputs) == 1 and tuple(outputs[0].orig_shape) == (1080, 1920),
                                'Unexpected detector output')
                        boxes = outputs[0].boxes
                        require(boxes is not None and len(boxes) <= PREDICT['max_det'],
                                'Invalid detector box count')
                        return boxes.xyxy.cpu().tolist()

                    result = run_image(predict, image, entry['gt'], args.out / split,
                                       entry['stem'], jpeg_writer())
                    atomic(marker_path, {'schema': 1, 'contract': contract_id, 'identity': identity,
                                         'status': 'complete', 'result': result,
                                         'job_id': os.environ['SLURM_JOB_ID'],
                                         'finished_utc': utcnow()})
                    merge_stats(stats, entry['stem'], result)
                splits[split] = stats
                if interrupted:
                    break
            code = 75 if (interrupted or stop[0] is not None) else 0
    except Exception as exc:
        error, code = f'{type(exc).__name__}: {exc}', 1
        traceback.print_exc()
    if stop[0] is not None and code == 0:
        code = 75
    status = 'completed' if code == 0 else 'interrupted' if code == 75 else 'failed'
    manifest = build_manifest(splits, {'status': status, 'error': error, 'source_commit': commit,
                                       'script_sha256': script_sha, 'detector_path': str(args.detector),
                                       'detector_sha256': detector_sha, 'data_path': str(args.data),
                                       'data_yaml_sha256': yaml_sha})
    manifest['elapsed_seconds'] = time.monotonic() - started
    atomic(args.out / 'manifest.json', manifest)
    return code


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=os.environ.get('FLYDET_PROP_CROPS_ROOT', str(DEFAULT_OUT)))
    parser.add_argument('--data', default=str(DEFAULT_DATA))
    parser.add_argument('--detector', default=str(DEFAULT_DETECTOR))
    parser.add_argument('--limit-images', type=int, help='Smoke: first N lexically sorted images per split')
    return parser


def prepare(args):
    allocation_guard()  # Before heavy imports, cache creation, or output writes.
    require(args.limit_images is None or args.limit_images > 0, '--limit-images must be positive')
    args.data = raid_path(args.data)
    args.detector = raid_path(args.detector)
    args.out = raid_path(args.out)
    require(args.detector.is_file(), f'Missing detector: {args.detector}')
    require(not args.out.is_relative_to(args.data) and not args.data.is_relative_to(args.out),
            'Output must be separate from the source dataset tree')
    args.out.mkdir(parents=True, exist_ok=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    stop, started = [None], time.monotonic()
    handlers = {s: signal.signal(s, lambda number, _frame: stop.__setitem__(0, number))
                for s in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
    try:
        prepare(args)
        return execute(args, stop, started)
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)


if __name__ == '__main__':
    raise SystemExit(main())
