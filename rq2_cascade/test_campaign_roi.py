"""Original phase-1 frozen GT-ROI adapter for consolidated TEST evaluation.

evaluate_roi(entries, out, stop_requested) -> JSON-serializable dict with id,
group, records, seconds and fingerprints. Each record contains image, true,
pred and probabilities (seven columns: MD/MV/MC/MF/INS/NOISE/MAR).

``out`` is the campaign directory; this adapter owns only its ``roi/`` child.
Resume is automatic, single-writer, atomic per image, and fails closed on any
changed contract/source/input or corrupt record. ``seconds`` sums persisted
per-image decode/full-detector/pool/head times across invocations, excluding
initialization, provenance verification and JSON persistence; it is NOT pure
detector latency. ``invocation_seconds`` includes this invocation's overhead.
The caller supplies a callable stop flag and owns signal handlers: cancellation
raises InterruptedError at image boundaries, never marks a partial image done.

Only stdlib imports at module scope. Runtime requires one visible CUDA GPU in
Slurm h100n2 + Apptainer, and the original torch/torchvision/Ultralytics versions.
Existing trusted local pickle checkpoints are opened only after these guards.
Original snapshot helpers are hash-checked and imported in an isolated package.
No training, submission, download, image resize or original dataset/cache write.

GT boxes are oracle inputs, unfiltered by object size: separate diagnostic group,
not end-to-end detection and not comparable to size-filtered GT-crop populations.
Probabilities are uncalibrated. Cache creation did not record CUDA numerical
flags: reproduce the pinned Torch 2.6 cache defaults (including cuDNN TF32=True),
then the explicit deterministic FP32/TF32-off head evaluation settings. This is
not a claim of bitwise identity across hardware/driver versions. Pinning library
versions and recording runtime provenance cannot establish split provenance.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import types
import uuid


RAID = Path('/raid/user_marcospaulo')
CAMPAIGN = RAID / 'flydet_runs/architecture-night-v1-20260907'
YOLO_CLASSES = ('MD', 'MV', 'MC', 'MF', 'INS', 'NOISE', 'MAR')
FOLDER_CLASSES = tuple(sorted(YOLO_CLASSES))
CLASS_MAP = [FOLDER_CLASSES.index(name) for name in YOLO_CLASSES]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _stop(callback):
    if callback():
        raise InterruptedError('ROI TEST evaluation interrupted at image boundary')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _trusted(path):
    path = Path(path)
    _require(path.is_absolute(), f'Absolute RAID path required: {path}')
    path = path.resolve(strict=True)
    _require(path.is_relative_to(RAID), f'Path outside trusted RAID: {path}')
    return path


def _sha(path):
    digest = hashlib.sha256()
    with _trusted(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path):
    path = _trusted(path)
    return {'path': str(path), 'sha256': _sha(path), 'bytes': path.stat().st_size}


def _read(path):
    _require(not Path(path).is_symlink(), f'Symlinked JSON forbidden: {path}')
    return json.loads(Path(path).read_text())


def _atomic(path, payload):
    """Durable replace; orphan temporary files are never treated as results."""
    _require(not path.is_symlink(), f'Symlinked output forbidden: {path}')
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x') as handle:
            json.dump(payload, handle, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _guard():
    """Environment checks precede even importing torch; CUDA checks follow it."""
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'h100n2':
        raise RuntimeError('ROI requires a Slurm GPU allocation on h100n2')
    if not (os.environ.get('APPTAINER_CONTAINER') or os.environ.get('APPTAINER_NAME')):
        raise RuntimeError('ROI requires Apptainer')
    if (os.environ.get('WORLD_SIZE', '1') != '1'
            or os.environ.get('SLURM_NTASKS', '1') != '1'
            or os.environ.get('SLURM_PROCID', '0') != '0'
            or os.environ.get('LOCAL_RANK', '0') != '0'):
        raise RuntimeError('ROI requires one process and one GPU')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(visible) != 1 or visible[0].strip() in ('', '-1', 'NoDevFiles', 'void', 'none'):
        raise RuntimeError('ROI requires exactly one CUDA_VISIBLE_DEVICES entry')


def _artifacts():
    summary_path = CAMPAIGN / 'roi/head/summary.json'
    summary = _read(summary_path)
    metadata_path = CAMPAIGN / 'roi/cache/metadata.json'
    metadata = _read(metadata_path)
    c, m = metadata['contract'], summary['metadata']
    _require(metadata['complete'] is True and metadata['status'] == 'complete'
             and metadata['digest'] == _digest(c), 'Invalid/incomplete ROI cache manifest')
    _require(summary['cache_metadata'] == metadata and m['cache_digest'] == metadata['digest'],
             'Head/cache provenance mismatch')
    _require(summary['exit_code'] == 0 and summary['error'] is None
             and summary['status'] in ('completed', 'early_stop', 'budget')
             and type(summary['best_epoch']) is int and summary['best_epoch'] > 0
             and summary['metrics'] is not None, 'No learned, successfully exported ROI head')
    _require(c['schema'] == 1 and c['layers'] == [4, 6] and c['strides'] == [8, 16]
             and c['classes'] == list(FOLDER_CLASSES) and c['class_map'] == CLASS_MAP
             and c['limit_images_per_split'] is None
             and c['input'] == 'RGB float32 /255; native 1920x1080; zero pad right/bottom to multiple32'
             and c['roi'] == {'size': [3, 3], 'aligned': True, 'sampling_ratio': -1, 'dtype': 'float16'},
             'Unsupported phase-1 ROI protocol')
    _require(metadata['channels'] == m['channels'] == 1024
             and metadata['feature_shapes'] == [[1, 512, 136, 240], [1, 512, 68, 120]]
             and m['head'] == 'Conv1x1->128 GELU Flatten Dropout0.2 Linear7'
             and m['args']['cache'] == str(metadata_path.parent)
             and m['torch'] == c['versions'][0], 'Head/cache architecture mismatch')
    _require({e['split'] for e in c['entries']} == {'train', 'val'}, 'Unexpected cache splits')
    source = CAMPAIGN / 'source/fly-det/scripts/architecture_lab'
    fingerprints = {name: _fingerprint(path) for name, path in {
        'checkpoint': CAMPAIGN / 'roi/head/best.pt', 'detector': c['detector']['path'],
        'summary': summary_path, 'cache_metadata': metadata_path,
        'roi_source': source / 'roi.py', 'train_source': source / 'train.py',
        'evaluator': Path(__file__).resolve(), 'dataset_yaml': c['yaml']['path'],
    }.items()}
    _require(fingerprints['roi_source']['sha256'] == c['roi_source_sha256'] == m['roi_sha256']
             and fingerprints['train_source']['sha256'] == m['metrics_sha256'],
             'Pinned original source SHA mismatch')
    for key, original in (('detector', c['detector']), ('dataset_yaml', c['yaml'])):
        _require(fingerprints[key]['sha256'] == original['sha256'], f'{key} SHA mismatch')
    return summary, metadata, fingerprints


def _entries(entries, dataset):
    """Verify original TEST files and the supplied GT order, without writing."""
    dataset = _trusted(dataset)
    image_root = _trusted(dataset / 'images/test')
    label_root = _trusted(dataset / 'labels/test')
    _require(image_root.is_relative_to(dataset) and label_root.is_relative_to(dataset),
             'TEST trees must remain inside the original dataset')
    result, seen = [], set()
    for entry in entries:
        image = _trusted(entry['image'])
        _require(image.is_relative_to(image_root) and str(image) not in seen,
                 f'Non-TEST or duplicate image: {image}')
        seen.add(str(image))
        label = _trusted((label_root / image.relative_to(image_root)).with_suffix('.txt'))
        _require(label.is_relative_to(label_root), 'Label escapes TEST')
        for key, path in (('image_sha256', image), ('label_sha256', label)):
            _require(entry[key] == _sha(path), f'Input fingerprint mismatch: {path}')
        expected = []
        for line in label.read_text().splitlines():
            if not line.strip():
                continue
            fields = line.split()
            _require(len(fields) == 5, 'Expected YOLO detection labels')
            cls = int(fields[0])
            cx, cy, w, h = map(float, fields[1:])
            _require(0 <= cls < 7 and all(math.isfinite(v) and 0 <= v <= 1
                                        for v in (cx, cy, w, h)) and min(w, h) > 0,
                     'Invalid original YOLO label')
            box = [(cx-w/2)*1920, (cy-h/2)*1080, (cx+w/2)*1920, (cy+h/2)*1080]
            _require(box[0] >= -.01 and box[1] >= -.01 and box[2] <= 1920.01
                     and box[3] <= 1080.01, 'Original ROI outside image')
            expected.append({'class': cls, 'box': [max(0., box[0]), max(0., box[1]),
                                                  min(1920., box[2]), min(1080., box[3])]})
        _require(len(entry['gt']) == len(expected), 'Missing/filtered TEST GT ROIs')
        targets = []
        for gt, original in zip(entry['gt'], expected):
            _require(type(gt['class']) is int and gt['class'] == original['class']
                     and len(gt['box']) == 4, 'GT class/order mismatch')
            box = [float(v) for v in gt['box']]
            _require(all(math.isfinite(a) and abs(a-b) <= 1e-4
                         for a, b in zip(box, original['box'])), 'GT geometry/order mismatch')
            _require(0 <= box[0] < box[2] <= 1920 and 0 <= box[1] < box[3] <= 1080,
                     'Expected clipped native pixel xyxy GT boxes')
            targets.append({'class': gt['class'], 'box': box})
        result.append({'image': str(image), 'label': str(label), 'gt': targets,
                       'image_sha256': entry['image_sha256'], 'label_sha256': entry['label_sha256']})
    return result


def _unchanged(entry):
    for key in ('image', 'label'):
        _require(_sha(entry[key]) == entry[f'{key}_sha256'], f'Changed TEST {key}: {entry[key]}')


@contextmanager
def _numeric(torch, head=False):
    old = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
           torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
           torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    try:
        torch.use_deterministic_algorithms(head)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = not head
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
            yield
    finally:
        torch.use_deterministic_algorithms(old[0], warn_only=old[1])
        (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
         torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) = old[2:]


@contextmanager
def _pinned(fingerprints):
    """No checkout imports, package __init__ execution, or sys.path changes."""
    name = '_test_roi_snapshot_' + uuid.uuid4().hex
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
    try:
        for leaf, key in (('train', 'train_source'), ('roi', 'roi_source')):
            fp = fingerprints[key]
            _require(_sha(fp['path']) == fp['sha256'], 'Snapshot changed before import')
            spec = importlib.util.spec_from_file_location(f'{name}.{leaf}', fp['path'])
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            # Compile in memory: never write bytecode into the original snapshot.
            exec(compile(Path(fp['path']).read_bytes(), fp['path'], 'exec'), module.__dict__)
        yield module
    finally:
        for key in (name + '.roi', name + '.train', name):
            sys.modules.pop(key, None)


def _runtime():
    _guard()
    # Set all potentially used third-party caches before importing libraries.
    for key, suffix in {
        'XDG_CACHE_HOME': 'xdg', 'TORCH_HOME': 'torch', 'HF_HOME': 'hf',
        'HF_HUB_CACHE': 'hf/hub', 'HUGGINGFACE_HUB_CACHE': 'hf/hub',
        'PIP_CACHE_DIR': 'pip', 'YOLO_CONFIG_DIR': 'ultralytics',
        'WANDB_DIR': 'wandb', 'WANDB_CACHE_DIR': 'wandb/cache',
        'WANDB_CONFIG_DIR': 'wandb/config', 'WANDB_DATA_DIR': 'wandb/data',
        'MPLCONFIGDIR': 'matplotlib', 'CUDA_CACHE_PATH': 'cuda',
        'TRITON_CACHE_DIR': 'triton', 'TORCHINDUCTOR_CACHE_DIR': 'inductor', 'TMPDIR': 'tmp',
    }.items():
        target = RAID / 'cache/test_campaign_roi' / suffix
        _require(target.resolve().is_relative_to(RAID), 'Cache symlink escapes RAID')
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true', WANDB_MODE='disabled',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA GPU required; CPU fallback forbidden')
    torch.cuda.set_device(0)
    import torchvision
    import ultralytics
    import numpy as np
    import PIL
    from PIL import Image
    versions = [str(torch.__version__), str(torchvision.__version__), ultralytics.__version__]
    runtime = {'versions': versions, 'numpy': np.__version__, 'pillow': PIL.__version__,
               'python': sys.version, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
               'gpu': torch.cuda.get_device_name(0),
               'container': os.environ.get('APPTAINER_CONTAINER') or os.environ.get('APPTAINER_NAME')}
    return torch, ultralytics, np, Image, runtime


def _validate_record(payload, contract_hash, entry):
    checksum = payload['sha256']
    _require(checksum == _digest({k: v for k, v in payload.items() if k != 'sha256'}),
             'Corrupt ROI result checksum')
    _require(payload['contract_sha256'] == contract_hash and payload['entry_sha256'] == _digest(entry),
             'ROI result contract/input mismatch')
    record = payload['record']
    _require(record['image'] == entry['image'] and record['true'] == [g['class'] for g in entry['gt']]
             and len(record['pred']) == len(record['probabilities']) == len(record['true']),
             'Invalid ROI result population')
    for predicted, row in zip(record['pred'], record['probabilities']):
        _require(type(predicted) is int and 0 <= predicted < 7 and len(row) == 7
                 and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in row)
                 and abs(sum(row)-1) < 1e-5 and row[predicted] == max(row),
                 'Invalid ROI probabilities/prediction')
    _require(type(payload['seconds']) in (int, float) and math.isfinite(payload['seconds'])
             and payload['seconds'] >= 0, 'Invalid ROI timing')
    return record


def evaluate_roi(entries, out, stop_requested) -> dict:
    """Evaluate supplied original TEST entries; no detector proposals or training.

    entries: iterable of {image: absolute path, gt: [{class: YOLO int,
    box: native pixel xyxy}], image_sha256: str, label_sha256: str}.
    out: campaign output directory on RAID, disjoint from source artifacts.
    stop_requested: callable returning truthy when cancellation is requested.
    """
    started = time.perf_counter()
    _guard()
    _require(callable(stop_requested), 'stop_requested must be callable')
    _stop(stop_requested)
    summary, metadata, fingerprints = _artifacts()
    dataset = _trusted(metadata['contract']['root'])
    entries = _entries(entries, dataset)
    _stop(stop_requested)
    root = Path(out)
    _require(root.is_absolute() and root.resolve().is_relative_to(RAID), 'Output must be on RAID')
    root = root.resolve() / 'roi'
    protected = [dataset, CAMPAIGN, Path(fingerprints['detector']['path']).parent]
    for source in protected:
        _require(not root.resolve().is_relative_to(source) and not source.is_relative_to(root.resolve()),
                 'ROI outputs must be disjoint from original dataset and artifacts')
    _require(not root.is_symlink(), 'Symlinked ROI output forbidden')
    root.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(root / '.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    environment = dict(os.environ)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another ROI evaluator owns this output') from exc
        torch, ultralytics, np, Image, runtime = _runtime()
        _require(runtime['versions'] == metadata['contract']['versions'],
                 'Original torch/torchvision/Ultralytics versions required')
        contract = {'schema': 1, 'id': 'roi', 'group': 'gt_roi', 'fingerprints': fingerprints,
                    'entries': entries, 'runtime': runtime, 'cache_digest': metadata['digest'],
                    'class_order': list(YOLO_CLASSES), 'head_batch': summary['metadata']['args']['batch'],
                    'protocol': 'native RGB /255; pad1088; P3/P4 aligned3x3; CPU half->float->CUDA; original head',
                    'numeric': 'cache Torch2.6 defaults; head deterministic TF32-off FP32; AMP off'}
        contract_hash = _digest(contract)
        manifest = root / 'contract.json'
        if manifest.exists():
            _require(_read(manifest) == {'sha256': contract_hash, 'contract': contract},
                     'ROI contract changed: choose a new campaign output')
        else:
            _require(not any(p.name != '.lock' and not p.name.endswith('.tmp') for p in root.iterdir()),
                     'Nonempty ROI output without a contract')
            _atomic(manifest, {'sha256': contract_hash, 'contract': contract})
        records_dir = root / 'images'
        _require(not records_dir.is_symlink(), 'Symlinked result directory forbidden')
        records_dir.mkdir(exist_ok=True)
        _stop(stop_requested)
        with _pinned(fingerprints) as roi:
            result = _evaluate(entries, records_dir, contract_hash, summary, metadata,
                               fingerprints, stop_requested, roi, torch, ultralytics, np, Image)
        _stop(stop_requested)
        result.update(id='roi', group='gt_roi', fingerprints={**fingerprints,
                      'contract_sha256': contract_hash, 'entries_sha256': _digest(entries)},
                      invocation_seconds=time.perf_counter()-started)
        return result
    finally:
        # Do not disable the enclosing campaign's W&B tracking/offline policy.
        for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'YOLO_AUTOINSTALL',
                    'YOLO_OFFLINE', 'WANDB_MODE', 'CUBLAS_WORKSPACE_CONFIG'):
            if key in environment:
                os.environ[key] = environment[key]
            else:
                os.environ.pop(key, None)
        os.close(lock_fd)


def _evaluate(entries, directory, digest, summary, metadata, fingerprints, stop,
              roi, torch, ultralytics, np, Image):
    detector = head = saved = tensor = features = pooled = logits = None
    captured, handles, records = {}, [], []
    seconds = 0.
    try:
        for index, entry in enumerate(entries):
            _stop(stop)
            _unchanged(entry)
            path = directory / f'{index:06d}.json'
            if path.exists():
                payload = _read(path)
                records.append(_validate_record(payload, digest, entry))
                seconds += payload['seconds']
                _stop(stop)
                continue
            if detector is None:
                for key in ('checkpoint', 'detector'):
                    _require(_sha(fingerprints[key]['path']) == fingerprints[key]['sha256'],
                             'Weights changed before loading')
                # Trusted original best.pt embeds state=best (see pinned persist()).
                saved = torch.load(fingerprints['checkpoint']['path'], map_location='cpu', weights_only=False)
                _require(saved['contract'] == summary['metadata']
                         and saved['state']['epoch'] == saved['best']['epoch'] == summary['best_epoch']
                         and saved['state']['metrics'] == summary['metrics']
                         and saved['wandb_run_id'] == summary['wandb_run_id'], 'Checkpoint/export mismatch')
                _require(saved['state']['model'].keys() == saved['best']['model'].keys()
                         and all(torch.equal(v, saved['best']['model'][k])
                                 for k, v in saved['state']['model'].items()), 'best.pt is not best state')
                head = roi.build_head(metadata['channels']).float().cuda().eval()
                head.load_state_dict(saved['state']['model'], strict=True)
                head.requires_grad_(False)
                saved = None
                detector = ultralytics.YOLO(fingerprints['detector']['path']).model.eval().float().cuda()
                detector.requires_grad_(False)
                _require(tuple(detector.names[i] for i in range(7)) == YOLO_CLASSES,
                         'Detector class order mismatch')
                for layer in metadata['contract']['layers']:
                    def hook(_module, _inputs, output, layer=layer):
                        _require(isinstance(output, torch.Tensor) and output.ndim == 4, 'Invalid P3/P4 hook')
                        captured[layer] = output.detach().clone()
                    handles.append(detector.model[layer].register_forward_hook(hook))
                _stop(stop)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with Image.open(entry['image']) as image:
                pw, ph = roi.padded_size(*image.size)
                array = np.array(image.convert('RGB'), copy=True)
            with _numeric(torch):
                tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).cuda().float().div_(255)
                tensor = torch.nn.functional.pad(tensor, (0, pw-1920, 0, ph-1080), value=0)
                boxes = torch.tensor([gt['box'] for gt in entry['gt']], device='cuda',
                                     dtype=torch.float32).reshape(-1, 4)
                captured.clear()
                detector(tensor)  # Full original detector forward, including head overhead.
                _require(set(captured) == set(metadata['contract']['layers']), 'Missing feature hooks')
                features = [captured[i] for i in metadata['contract']['layers']]
                _require([list(f.shape) for f in features] == metadata['feature_shapes'], 'Feature shape mismatch')
                # EXACT training cache path: GPU float32 pool -> CPU float16;
                # ROIDataset.__getitem__ casts to CPU float32 before .cuda().
                pooled = roi.pool_features(features, boxes).cpu().half()
                _require(torch.isfinite(pooled).all().item(), 'Nonfinite quantized ROI features')
            probabilities, predicted = [], []
            batch = summary['metadata']['args']['batch']
            with _numeric(torch, head=True):
                for start in range(0, len(pooled), batch):
                    logits = head(pooled[start:start+batch].float().cuda())
                    _require(torch.isfinite(logits).all().item(), 'Nonfinite ROI logits')
                    # Preserve alphabetical argmax tie-breaking, then remap IDs.
                    probs = logits.softmax(1)
                    predicted.extend(CLASS_MAP.index(i) for i in probs.argmax(1).tolist())
                    probabilities.extend(probs[:, CLASS_MAP].cpu().tolist())
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-begin
            record = {'image': entry['image'], 'true': [gt['class'] for gt in entry['gt']],
                      'pred': predicted, 'probabilities': probabilities}
            payload = {'contract_sha256': digest, 'entry_sha256': _digest(entry),
                       'record': record, 'seconds': elapsed}
            payload['sha256'] = _digest(payload)
            _validate_record(payload, digest, entry)
            _unchanged(entry)
            _stop(stop)  # A signalled image is discarded, not marked complete.
            _atomic(path, payload)
            records.append(record)
            seconds += elapsed
            captured.clear()
            tensor = features = pooled = logits = None
            _stop(stop)
        for fp in fingerprints.values():
            _require(_fingerprint(fp['path']) == fp, 'Artifact changed during ROI evaluation')
        return {'records': records, 'seconds': seconds}
    finally:
        for handle in handles:
            handle.remove()
        captured.clear()
        detector = head = saved = tensor = features = pooled = logits = None
        gc.collect()
        torch.cuda.empty_cache()