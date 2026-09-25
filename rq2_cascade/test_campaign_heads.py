"""Read-only classifier adapters for the full TEST campaign.

Public interface (import/discovery require only the standard library):
    discover_heads() -> list of JSON-serializable specs
    load_head(spec) -> adapter
    adapter.predict(list[PIL.Image.Image]) -> list[list[float]]
    adapter.gradcam(image) -> (float32 ndarray[original H,W], YOLO class index)
    adapter.close()  # also supports `with load_head(spec) as head:`

Probability columns are ALWAYS MD, MV, MC, MF, INS, NOISE, MAR. Scores are
uncalibrated softmax probabilities, not detection confidences. No evaluation,
dataset extraction, training, submission or downloads happen here. The caller
owns splits, crop extraction/padding, metrics and the final manifest.

Discovery reads actual summaries, hashes checkpoints (streaming; no unpickling),
and verifies original architecture snapshots, not the current checkout. Each
spec exposes spec_hash, checkpoint_sources, sources, summary, metadata (for
architecture), crop_root, pad, size, variant, batch_size and supports_cascade.
`checkpoint` is the first member for ensembles; `checkpoints` and
`checkpoint_sources` are authoritative for ALL members. Ensemble input is
pad75 -> PIL bilinear 384 -> tensor -> normalization -> per-model normalized
tensor bilinear resize (antialias=False) -> optional six views -> mean softmax.

Legacy summaries lack crop/size provenance: these fields are cross-checked
against the actual Slurm scripts and then the checkpoint at load time. Their
source hashes identify the current legacy scripts, NOT an original snapshot.
Architecture sources are the original, SHA-validated models.py/train.py pair.
Discovery deliberately excludes smoke, ROI and duplicated W&B summaries.
The dinov2-20260909 campaign (seed42/seed84 runs, backbone dinov2_vitb14,
nominal output stride 14, no conv stem) is part of discovery; while its
directory or summaries are absent it is skipped only when the caller passes
allow_missing_campaigns=True, otherwise discovery fails closed.

Loading and all CUDA work require Slurm h100n2 + Apptainer. Only trusted,
resolved files under TRUSTED_ROOT may be unpickled (weights_only=False).
FP32, disabled AMP/TF32, eval-only; batch 8 for hires mechanisms, 32 otherwise.
No initial/fallback exports are accepted. close() drops models and empties the
CUDA cache, but cannot release tensors retained by external callers.

SAM has pad=None and supports_cascade=False. For SAM predict/gradcam, pass
original PIL Image.open objects retaining `filename`, located in crop_root/test;
ordinary ImageFolder's RGB conversion drops filename and is NOT sufficient.
Never pass uncleaned detector crops to SAM. Other adapters accept arbitrary PIL
images, which the caller must already have cropped with the specified padding.
Grad-CAM uses raw architecture feature_map or legacy final features; Swin's
BHWC activations AND gradients are converted to BCHW. Ensemble Grad-CAM is
explicitly unsupported. CAM is qualitative, not a causal explanation.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import types


TRUSTED_ROOT = Path('/raid/user_marcospaulo')
RUNS_ROOT = TRUSTED_ROOT / 'flydet_runs'
REPO_ROOT = Path(__file__).resolve().parents[2]
YOLO_CLASSES = ('MD', 'MV', 'MC', 'MF', 'INS', 'NOISE', 'MAR')
FOLDER_CLASSES = tuple(sorted(YOLO_CLASSES))
NORMALIZATION = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
_LEGACY = {
    'E3_swin': ('E3_swin_classifier.sh', 'swin_t', 224, 'DS-F2_crops', 0.15),
    'E3b_convnext': ('E3b_convnext.sh', 'convnext_t', 384, 'DS-F2_crops_pad75', 0.75),
    'E3c_focal': ('E3c_focal.sh', 'convnext_t', 384, 'DS-F2_crops_pad75', 0.75),
    'E3d_convnext_s': ('E3d_convnext_s.sh', 'convnext_s', 384, 'DS-F2_crops_pad75', 0.75),
    'E4_convnext_sam': ('E4_convnext_sam.sh', 'convnext_t', 384, 'DS-F2_crops_sam', None),
    'E5_convnext_512': ('E5_convnext_512.sh', 'convnext_t', 512, 'DS-F2_crops_pad75', 0.75),
}
_VARIANTS = {'baseline', 'supcon', 'arcface', 'parts', 'hires',
             'parts_supcon', 'hires_parts', 'hires_supcon', 'dinov2'}
_LEGACY_WITHOUT_ARCH = frozenset({'E3_swin', 'E3b_convnext', 'E3c_focal'})
_DINOV2_CAMPAIGN = 'dinov2-20260909'
_DINOV2_SEEDS = ('seed42', 'seed84')


def _canonical_metadata(metadata):
    """Normalize tuple/list serialization without dropping provenance fields."""
    return json.loads(json.dumps(metadata, sort_keys=True, allow_nan=False))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _trusted(path):
    path = Path(path)
    _require(path.is_absolute(), f'Absolute path required: {path}')
    path = path.resolve(strict=True)
    _require(path.is_relative_to(TRUSTED_ROOT.resolve()), f'Untrusted path: {path}')
    return path


def _json(path):
    return json.loads(_trusted(path).read_text())


def _sha(path):
    digest = hashlib.sha256()
    with _trusted(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def spec_hash(spec):
    """Canonical SHA256 of the entire spec except its own spec_hash field."""
    payload = {k: v for k, v in spec.items() if k != 'spec_hash'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _classes(classes):
    _require(isinstance(classes, (list, tuple)) and len(classes) == 7
             and all(isinstance(c, str) for c in classes)
             and set(classes) == set(YOLO_CLASSES), f'Invalid class order: {classes}')
    return [classes.index(name) for name in YOLO_CLASSES]


def _script_options(path):
    """Read simple literal assignments/options; never execute a shell script."""
    text = _trusted(path).read_text()
    variables = {}
    for key, value in re.findall(r'^([A-Z_]+)="([^"\n]*)"$', text, re.MULTILINE):
        variables[key] = re.sub(r'\$\{([A-Z_]+)\}',
                                lambda m: variables.get(m[1], m[0]), value)
    tokens = shlex.split(text.replace('\\\n', ' '), comments=True)
    tokens = [re.sub(r'\$\{([A-Z_]+)\}', lambda m: variables.get(m[1], m[0]), t)
              for t in tokens]
    result = {}
    for option in ('--data', '--model', '--img-size', '--ckpts'):
        if option not in tokens:
            continue
        start = tokens.index(option) + 1
        end = start + 1
        if option == '--ckpts':
            while end < len(tokens) and not tokens[end].startswith('--'):
                end += 1
        result[option] = tokens[start:end]
    return result


def _validate_arch_summary(summary):
    _require(summary.get('schema_version') == 1, 'Unknown architecture summary schema')
    _require(summary.get('export_complete') is True, 'Incomplete architecture export')
    epoch = summary.get('exported_epoch')
    _require(type(epoch) is int and epoch > 0, 'Missing/initial exported epoch')
    _require(summary.get('initial_checkpoint_fallback') is False, 'Initial fallback export')
    m = summary['metadata']
    _classes(m['class_order'])
    _require(tuple(m['class_order']) == FOLDER_CLASSES, 'Unexpected architecture class order')
    _require(m['class_to_idx'] == {c: i for i, c in enumerate(m['class_order'])},
             'class_to_idx mismatch')
    args, prep, arch = m['args'], m['preprocessing'], m['architecture']
    _require(m.get('smoke') is False and not args.get('limit_train')
             and not args.get('limit_val'), 'Smoke/limited run is not a TEST head')
    variant, size = args['variant'], args['img_size']
    _require(variant in _VARIANTS and arch['variant'] == variant, 'Variant mismatch')
    _require(type(size) is int and size > 0, 'Invalid image size')
    _require(prep['resize'] == [size, size] and prep['interpolation'] == 'bilinear'
             and prep['antialias'] is True and prep['normalization'] == NORMALIZATION
             and prep['dataset_files_modified'] is False
             and 'pad75 RGB crops' in prep['input']
             and prep['val'] == 'resize + tensor + ImageNet normalization; no augmentation',
             'Unsupported architecture preprocessing')
    _require(m['dataset_root'] == args['data'], 'Dataset metadata mismatch')
    _require(Path(args['data']) == TRUSTED_ROOT / 'datasets/DS-F2_crops_pad75',
             'Unsupported architecture crop root')
    if variant == 'dinov2':
        # ViT-B/14 patch embedding, no conv stem: stem fields are null by contract.
        _require(arch['backbone'] == 'dinov2_vitb14'
                 and arch['stem_stride'] is None
                 and arch['stem_original_stride'] is None
                 and arch['stem_weights_preserved'] is None
                 and arch['nominal_output_stride'] == 14, 'Invalid architecture metadata')
    else:
        hires = 'hires' in variant.split('_')
        _require(arch['backbone'] == 'torchvision.convnext_tiny'
                 and arch['stem_stride'] == ([2, 2] if hires else [4, 4])
                 and arch['stem_original_stride'] == [4, 4]
                 and arch['nominal_output_stride'] == (16 if hires else 32)
                 and arch['stem_weights_preserved'] is True, 'Invalid architecture metadata')
    return m


def discover_heads(runs_root=RUNS_ROOT, repo_root=REPO_ROOT, campaign_roots=None,
                   allow_missing_campaigns=False):
    """Discover complete real exports, fail closed on missing/invalid artifacts.

    Checkpoint hashes are computed without torch or pickle. Alternative roots
    must still be trusted RAID paths; script provenance must agree with them.
    No checkpoint/model cache, files or directories are created. campaign_roots
    optionally restricts which campaign directories are scanned (default: all
    architecture-night-*/architecture-followup-* plus dinov2-20260909). Spec
    order is deterministic: legacy, campaigns sorted by directory name, then
    ensembles. A missing dinov2 campaign (absent directory or seed summaries)
    is skipped only with allow_missing_campaigns=True; otherwise it is an error.
    """
    runs, repo = _trusted(runs_root), _trusted(repo_root)
    scripts, slurm = repo / 'fly-det/scripts', repo / 'fly-det/slurm'
    fingerprints = {}

    def record(path):
        path = _trusted(path)
        key = str(path)
        if key not in fingerprints:
            fingerprints[key] = {'path': key, 'sha256': _sha(path),
                                 'bytes': path.stat().st_size}
        return dict(fingerprints[key])

    def finish(spec, checkpoints, sources):
        spec['checkpoint_sources'] = [record(p) for p in checkpoints]
        spec['checkpoints'] = [r['path'] for r in spec['checkpoint_sources']]
        spec['checkpoint'] = spec['checkpoints'][0]
        spec['sources'] = [record(p) for p in sources]
        spec['source'] = spec['sources'][0]['path']
        spec['crop_root'] = str(_trusted(spec['crop_root']))
        test = _trusted(Path(spec['crop_root']) / 'test')
        _require(all((test / name).is_dir() for name in FOLDER_CLASSES),
                 f'Missing TEST class directories: {test}')
        spec['class_order_yolo'] = list(YOLO_CLASSES)
        spec['batch_size'] = 8 if 'hires' in spec['variant'].split('_') else 32
        spec['spec_hash'] = spec_hash(spec)
        return spec

    specs = []
    for name, (script_name, variant, size, crop_name, pad) in _LEGACY.items():
        summary_path = runs / name / 'summary.json'
        summary = _json(summary_path)
        _classes(summary['classes'])
        _require(summary['model'] == variant, f'Legacy model mismatch: {name}')
        script = slurm / script_name
        options = _script_options(script)
        crop_root = TRUSTED_ROOT / 'datasets' / crop_name
        _require(options['--data'] == [str(crop_root)]
                 and options['--model'] == [variant]
                 and options['--img-size'] == [str(size)], f'Legacy script changed: {script}')
        specs.append(finish({
            'id': name, 'kind': 'legacy', 'variant': variant, 'size': size,
            'crop_root': str(crop_root), 'pad': pad, 'supports_cascade': pad is not None,
            'summary': str(summary_path), 'classes': summary['classes'],
            'source_provenance': 'current legacy scripts; no historical source snapshot',
        }, [runs / name / 'best.pt'], [scripts / 'train_classifier.py', script, summary_path]))

    def architecture_spec(campaign, path):
        summary = _json(path)
        m = _validate_arch_summary(summary)
        _require(_trusted(m['args']['out']) == path.parent.resolve(),
                 f'Export output path mismatch: {path}')
        source_dir = campaign / 'source/fly-det/scripts/architecture_lab'
        source_files = [source_dir / n for n in ('models.py', 'train.py')]
        for source in source_files:
            _require(record(source)['sha256'] == m['source']['file_sha256'][source.name],
                     f'Original snapshot SHA mismatch: {source}')
        return finish({
            'id': '/'.join(path.parent.relative_to(runs).parts),
            'kind': 'architecture', 'variant': m['args']['variant'],
            'size': m['args']['img_size'], 'seed': m['args']['seed'],
            'crop_root': m['dataset_root'], 'pad': 0.75, 'supports_cascade': True,
            'summary': str(path), 'exported_epoch': summary['exported_epoch'],
            'metadata': m, 'classes': m['class_order'],
            'source_provenance': 'original per-run SHA-verified snapshot',
        }, [path.parent / 'best.pt'], [*source_files, path])

    def campaign_summaries(campaign):
        if campaign.name == _DINOV2_CAMPAIGN:
            # Explicit seed runs; the smoke export sibling is never scanned.
            paths = [campaign / seed / 'summary.json' for seed in _DINOV2_SEEDS]
            if not all(p.is_file() for p in paths):
                _require(allow_missing_campaigns,
                         f'Missing dinov2 campaign summaries: {campaign}')
                return []
            return paths
        return [path for group in ('crops', 'combos', 'seed84')
                for path in sorted((campaign / group).glob('*/summary.json'))]

    if campaign_roots is None:
        campaigns = sorted(p for p in runs.iterdir() if p.is_dir() and
                           (p.name.startswith('architecture-night-') or
                            p.name.startswith('architecture-followup-')))
        _require(bool(campaigns), 'No architecture campaigns found')
        dinov2 = runs / _DINOV2_CAMPAIGN
        if dinov2.is_dir():
            campaigns.append(dinov2)  # 'architecture-*' sorts before 'dinov2-*'
        else:
            _require(allow_missing_campaigns,
                     f'Missing architecture campaign: {dinov2}')
    else:
        campaigns = []
        for root in campaign_roots:
            path = Path(root)
            if not (path.is_absolute() and path.is_dir()):
                _require(allow_missing_campaigns,
                         f'Missing architecture campaign: {root}')
                continue
            campaigns.append(_trusted(path))
        _require(campaigns or allow_missing_campaigns, 'No architecture campaigns found')
        campaigns.sort()
    for campaign in campaigns:
        for path in campaign_summaries(campaign):
            specs.append(architecture_spec(campaign, path))

    ensemble_script = slurm / 'E3_ensemble_tta.sh'
    options = _script_options(ensemble_script)
    members = [next(s for s in specs if s['id'] == name)
               for name in ('E3_swin', 'E3b_convnext', 'E3c_focal')]
    checkpoints = [s['checkpoint'] for s in members]
    _require(options['--ckpts'] == checkpoints, 'Original ensemble members changed')
    summary_path = runs / 'E3_ensemble_tta/summary.json'
    summary = _json(summary_path)
    _classes(summary['classes'])
    _require(summary['tta'] is True and summary['base_size'] == 384
             and summary['models'] == [[s['variant'], s['size']] for s in members]
             and all(s['classes'] == summary['classes'] for s in members),
             'Ensemble summary mismatch')
    crop_root = str(TRUSTED_ROOT / 'datasets/DS-F2_crops_pad75')
    _require(options['--data'] == [crop_root] and options['--img-size'] == ['384'],
             'Ensemble preprocessing script mismatch')
    for tta in (False, True):
        specs.append(finish({
            'id': 'E3_ensemble_tta' if tta else 'E3_ensemble', 'kind': 'ensemble',
            'variant': 'ensemble_tta' if tta else 'ensemble', 'size': 384,
            'crop_root': crop_root, 'pad': 0.75, 'supports_cascade': True,
            'summary': str(summary_path), 'classes': summary['classes'],
            'tta': tta, 'members': members, 'views': 6 if tta else 1,
            'source_provenance': 'original ensemble membership from Slurm script',
        }, checkpoints, [scripts / 'ensemble_tta.py', ensemble_script, summary_path]))
    _require(len({s['id'] for s in specs}) == len(specs), 'Duplicate head IDs')
    return specs


def _guard():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Head inference requires SLURM_JOB_ID; never run on login nodes')
    if os.environ.get('SLURM_JOB_PARTITION') != 'h100n2':
        raise RuntimeError('Head inference requires SLURM_JOB_PARTITION=h100n2')
    if not (os.environ.get('APPTAINER_CONTAINER') or os.environ.get('APPTAINER_NAME')):
        raise RuntimeError('Head inference requires Apptainer')


def _runtime():
    _guard()  # Must precede ALL optional/compute imports, including snapshot execution.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required inside the Slurm/Apptainer allocation')
    from torchvision import models, transforms
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch, models, transforms


def _validate_spec(spec):
    _require(spec.get('spec_hash') == spec_hash(spec), 'Spec hash mismatch')
    _require(spec['kind'] in {'legacy', 'architecture', 'ensemble'}, 'Unknown head kind')
    _classes(spec['classes'])
    _require(spec['class_order_yolo'] == list(YOLO_CLASSES), 'Invalid output class order')
    _require(spec['checkpoint'] == spec['checkpoints'][0]
             and spec['checkpoints'] == [r['path'] for r in spec['checkpoint_sources']],
             'Checkpoint source list mismatch')
    _require(spec['source'] == spec['sources'][0]['path'], 'Model source mismatch')
    for record in spec['checkpoint_sources'] + spec['sources']:
        path = _trusted(record['path'])
        _require(path.stat().st_size == record['bytes'] and _sha(path) == record['sha256'],
                 f'Artifact changed since discovery: {path}')
    _require(spec['batch_size'] == (8 if 'hires' in spec['variant'].split('_') else 32),
             'Invalid inference batch size')
    _trusted(spec['crop_root'])
    if spec['kind'] == 'ensemble':
        _require(spec['checkpoints'] == [m['checkpoint'] for m in spec['members']]
                 and len(spec['members']) == 3, 'Invalid ensemble members')
        for member in spec['members']:
            _validate_spec(member)


def _validate_checkpoint(spec, ck):
    """Schema-only validation; also testable with ordinary dictionaries, no torch."""
    _require(isinstance(ck, dict) and isinstance(ck.get('model'), dict),
             'Expected checkpoint model state_dict')
    if spec['kind'] == 'architecture':
        summary = _json(spec['summary'])
        metadata = _validate_arch_summary(summary)
        _require(_canonical_metadata(ck.get('metadata')) == _canonical_metadata(metadata)
                 == _canonical_metadata(spec['metadata']),
                 'Checkpoint/summary metadata mismatch')
        _require(ck.get('is_initial_fallback') is False, 'Initial checkpoint fallback')
        _require(type(ck.get('epoch')) is int
                 and ck['epoch'] == spec['exported_epoch'] == summary['exported_epoch'],
                 'Checkpoint/exported epoch mismatch')
        _require(metadata['args']['variant'] == spec['variant']
                 and metadata['args']['img_size'] == spec['size']
                 and metadata['class_order'] == spec['classes']
                 and metadata['dataset_root'] == spec['crop_root']
                 and spec['pad'] == 0.75 and spec['supports_cascade'] is True,
                 'Architecture spec metadata mismatch')
    else:
        _classes(ck.get('classes'))
        expected = _LEGACY[spec['id']]
        # These three inspected exports omit arch. Discovery validates their
        # explicit variant against BOTH the Slurm script and summary; load_head
        # rechecks source hashes, and _build_single always loads weights strictly.
        missing_arch = 'arch' not in ck and spec['id'] in _LEGACY_WITHOUT_ARCH
        _require((ck.get('arch') == spec['variant'] or missing_arch)
                 and spec['variant'] == expected[1] and spec['size'] == expected[2]
                 and type(ck.get('img_size')) is int
                 and ck['img_size'] == spec['size'] and ck['classes'] == spec['classes'],
                 'Legacy checkpoint arch/img_size/classes mismatch (no guessing)')
        _require(spec['pad'] == expected[4]
                 and spec['supports_cascade'] is (expected[4] is not None)
                 and Path(spec['crop_root']) == TRUSTED_ROOT / 'datasets' / expected[3],
                 'Legacy crop/padding contract mismatch')


def _build_single(spec, torch, models):
    checkpoint = _trusted(spec['checkpoint'])
    ck = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    _validate_checkpoint(spec, ck)
    if spec['kind'] == 'architecture':
        source = _trusted(spec['source'])
        code = source.read_bytes()
        _require(hashlib.sha256(code).hexdigest() ==
                 ck['metadata']['source']['file_sha256']['models.py'], 'Snapshot SHA mismatch')
        # No sys.path mutation, module-name collision or __pycache__ writes.
        module = types.ModuleType('_test_head_' + spec['spec_hash'])
        module.__file__ = str(source)
        exec(compile(code, str(source), 'exec'), module.__dict__)
        model = module.build_model(spec['variant'], 7, pretrained=False)
        reconstructed = _canonical_metadata(model.architecture_metadata)
        architecture = ck['metadata']['architecture']
        # Training initialization provenance is not a reconstruction parameter
        # (dinov2 also records the pretrained checkpoint path/hash there).
        for key in ('pretrained', 'checkpoint_path', 'checkpoint_sha256'):
            if key in reconstructed or key in architecture:
                reconstructed[key] = architecture.get(key)
        _require(reconstructed == _canonical_metadata(architecture),
                 'Reconstructed architecture differs from checkpoint metadata')
    else:
        factories = {'swin_t': models.swin_t, 'convnext_t': models.convnext_tiny,
                     'convnext_s': models.convnext_small}
        model = factories[spec['variant']](weights=None)
        if spec['variant'].startswith('swin'):
            model.head = torch.nn.Linear(model.head.in_features, 7)
        else:
            model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 7)
    model.load_state_dict(ck['model'], strict=True)
    return model.float().eval().requires_grad_(False).to('cuda')


def load_head(spec):
    """Load a discovered spec only in Slurm h100n2 / Apptainer / CUDA.

    Verifies manifest content hashes before trusted checkpoint deserialization.
    No GPU job is submitted automatically. Caller must close the returned head.
    """
    _guard()
    spec = copy.deepcopy(spec)
    _validate_spec(spec)
    torch, models, transforms = _runtime()
    adapter = _Head(spec, torch, transforms)
    try:
        members = spec['members'] if spec['kind'] == 'ensemble' else [spec]
        for member in members:
            adapter.models.append(_build_single(member, torch, models))
        return adapter
    except BaseException:
        adapter.close()
        raise


class _Head:
    def __init__(self, spec, torch, transforms):
        self.spec = spec
        self.models = []
        self.closed = False
        self.torch = torch
        self.supports_cascade = spec['supports_cascade']
        self.batch_size = spec['batch_size']
        self.order = _classes(spec['classes'])
        norm = (spec['metadata']['preprocessing']['normalization']
                if spec['kind'] == 'architecture' else NORMALIZATION)
        self.transform = transforms.Compose([
            transforms.Resize((spec['size'], spec['size']),
                              interpolation=transforms.InterpolationMode.BILINEAR,
                              antialias=True),
            transforms.ToTensor(), transforms.Normalize(norm['mean'], norm['std']),
        ])

    def _ready(self):
        _guard()
        if self.closed:
            raise RuntimeError('Head is closed')

    def _image(self, image):
        if not self.supports_cascade:
            filename = getattr(image, 'filename', None)
            _require(bool(filename), 'SAM requires an existing cleaned TEST crop filename')
            path = _trusted(filename)
            test = _trusted(Path(self.spec['crop_root']) / 'test')
            _require(path.is_file() and path.is_relative_to(test)
                     and path.relative_to(test).parts[0] in FOLDER_CLASSES,
                     'SAM only accepts existing cleaned GT TEST crops, not cascade crops')
        return self.transform(image.convert('RGB'))

    @staticmethod
    def _logits(output):
        return output['logits'] if isinstance(output, dict) else output

    def predict(self, images):
        """Return N x 7 FP32 softmax probabilities reordered to YOLO_CLASSES."""
        self._ready()
        torch = self.torch
        result = []
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
            for start in range(0, len(images), self.batch_size):
                x = torch.stack([self._image(i) for i in images[start:start + self.batch_size]])
                x = x.to(device='cuda', dtype=torch.float32)
                if self.spec['kind'] != 'ensemble':
                    probs = self._logits(self.models[0](x)).softmax(dim=1)
                else:
                    probs = None
                    for model, member in zip(self.models, self.spec['members']):
                        size = member['size']
                        resized = (torch.nn.functional.interpolate(
                            x, size=(size, size), mode='bilinear', align_corners=False,
                            antialias=False) if size != self.spec['size'] else x)
                        views = [resized]
                        if self.spec['tta']:
                            views += [torch.flip(resized, [3]), torch.flip(resized, [2])]
                            views += [torch.rot90(resized, k, [2, 3]) for k in (1, 2, 3)]
                        for view in views:
                            p = self._logits(model(view)).softmax(dim=1)
                            probs = p if probs is None else probs + p
                    probs = probs / (len(self.models) * self.spec['views'])
                result.extend(probs[:, self.order].cpu().tolist())
        return result

    def gradcam(self, image):
        """Return normalized original-size CAM and predicted YOLO class index."""
        self._ready()
        if self.spec['kind'] == 'ensemble':
            raise NotImplementedError('Grad-CAM is not defined for ensemble adapters')
        torch, model = self.torch, self.models[0]
        captured = []
        hook = None
        try:
            if self.spec['kind'] == 'legacy':
                hook = model.features.register_forward_hook(
                    lambda module, inputs, output: captured.append(output))
            with torch.inference_mode(False), torch.enable_grad(), \
                    torch.autocast(device_type='cuda', enabled=False):
                x = self._image(image).unsqueeze(0).to('cuda', dtype=torch.float32)
                x.requires_grad_(True)
                output = model(x)
                logits = self._logits(output)
                index = int(logits[0].argmax().item())
                features = output['feature_map'] if isinstance(output, dict) else captured[0]
                gradients, = torch.autograd.grad(logits[0, index], features)
                if self.spec['kind'] == 'legacy' and self.spec['variant'].startswith('swin'):
                    features = features.permute(0, 3, 1, 2)
                    gradients = gradients.permute(0, 3, 1, 2)
                cam = (gradients.mean(dim=(2, 3), keepdim=True) * features).sum(1).relu()
                cam = torch.nn.functional.interpolate(
                    cam.unsqueeze(1), size=(image.height, image.width),
                    mode='bilinear', align_corners=False)[0, 0]
                cam = cam - cam.min()
                cam = cam / cam.max().clamp_min(1e-12)
                return cam.detach().float().cpu().numpy(), self.order.index(index)
        finally:
            if hook is not None:
                hook.remove()
            captured.clear()
            model.zero_grad(set_to_none=True)

    def close(self):
        """Idempotently release model references and unused CUDA allocator blocks."""
        if not self.closed:
            self.models.clear()
            self.closed = True
            gc.collect()
            self.torch.cuda.empty_cache()

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()