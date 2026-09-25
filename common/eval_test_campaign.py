#!/usr/bin/env python3
"""Frozen TEST campaign; evaluation only, stdlib-only import and --help.

Execution belongs to the external h100n2/Apptainer launcher (4 hours). This
process checkpoints at image/batch boundaries and returns 75 on a signal or
its 3h50 soft deadline. --resume requires identical code, inputs, exports and
runtime. No training, downloads, threshold selection or dataset writes.

Only the public head/ROI adapters perform classification. The VAL evaluator
is imported in memory, never edited; its tested matching/AP implementation is
reused with a temporary single-pipeline configuration. Native TEST is always
verified in full, including during smoke runs. Existing crop filenames must
identify the exact TEST image and original label ordinal, not just a prefix.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import datetime as dt
import fcntl
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time
import traceback
import types
import uuid


RAID = Path('/raid/user_marcospaulo')
REPO = RAID / 'fly-det'
RUNS = RAID / 'flydet_runs'
DATA = RAID / 'datasets/DS-F2_v8.1.1'
CLASSES = ('MD', 'MV', 'MC', 'MF', 'INS', 'NOISE', 'MAR')
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
POLICIES = ('label_only', 'product')
SOFT_SECONDS = 3 * 3600 + 50 * 60
PREDICT = dict(imgsz=1920, conf=.001, iou=.5, max_det=300, rect=True,
               device=0, half=False, verbose=False, augment=False, save=False)
DETECTORS = {
    'E0': RUNS / 'E0_yolo26m_baseline/runs/yolo26m.pt/weights/best.pt',
    'E1': RUNS / 'E1_yolo26m_p2/runs/yolo26m-p2.yaml/weights/best.pt',
    'E2': RUNS / 'E2_yolo26m_sahi/runs/yolo26m.pt/weights/best.pt',
}
POPULATIONS = {'DS-F2_crops': 2707, 'DS-F2_crops_pad75': 2840, 'DS-F2_crops_sam': 2840}
WARNINGS = [
    'TEST já foi exposto: não é uma avaliação cega; não se afirma significância estatística.',
    'Sem tuning no TEST: confusão em score >= 0.25 e IoU >= 0.50, fixados antecipadamente.',
    'Espécies são somente MD/MV/MC/MF (índices 0–3). INS NÃO é espécie-alvo.',
    'Precisão/recall/F1 micro somam TP/FP/FN por classe, incluindo erros para outras classes e BG.',
    'Acurácia condicional = diagonal / GT pareados (linhas verdadeiras do grupo, colunas 0–6); '
    'não é acurácia padrão de detecção. Recall inclui objetos perdidos. Não há TN de detecção.',
    'GT-crops e GT-RoI são diagnósticos oracle, não detecção ponta a ponta. Populações filtradas '
    '2707 (pad15) e 2840 (pad75/SAM) diferem dos 2864 GT nativos; não usar ranking conjunto.',
    'label_only preserva score YOLO da classe original (não objectness); product multiplica '
    'por max softmax, sem calibração. Todas as propostas E0, inclusive FP, são classificadas.',
    'Crops de cascata são RGB nativos em memória, sem recompressão JPEG; padding int/clip '
    'por lado, sem descarte por tamanho. Crop inválido preserva classe/score YOLO.',
    'E2_sliced: grade manual 512, stride 384, último início ajustado à borda (overlap final '
    'pode ser maior que 25%); offset global, torchvision batched_nms class-aware IoU .5, '
    'FP32 CPU, cap global 300. Não se afirma equivalência bit a bit ao Supervision.',
    'E0/E1/E2 usam imgsz1920; E2_sliced usa imgsz512 por tile, sem alterar arquivos nativos. '
    'YOLO26 end2end pode ignorar o parâmetro NMS IoU na predição por imagem/tile.',
    'mAP50 e mAP50:95 apenas para detecção/cascata, média das classes com GT; '
    'o piso .001 e cap300 limitam propostas. Matching é o do avaliador existente.',
    'Tempos observacionais, não benchmark repetido: warmup, carregamento, SHA, persistência, '
    'métricas, CAM e rede excluídos. Detector inclui preprocessamento/pós-processamento; '
    'GT batch exclui decode; cascata inclui crop/transform/inferência e soma tempo E0.',
    'RoI inclui decode/detector/pool/head e primeira imagem sem warmup separado, conforme '
    'adaptador original; não comparar sua latência diretamente com GT-crops.',
    'FP32 sem AMP/TF32 para detectores/crops; RoI reproduz flags específicas do cache original. '
    'Ensemble mantém seus três membros simultaneamente, nunca cabeças independentes juntas.',
    'CAM é qualitativo, não explicação causal; primeiros até 3 erros e 3 acertos em ordem fixa. '
    'Ensembles não oferecem Grad-CAM. SAM não recebe propostas não limpas.',
    'Não selecionar implantação somente por TEST; não declarar meta 0.95 atingida sem valores reais.',
]


def sibling(name):
    """Load trusted local stdlib-only helpers without creating __pycache__."""
    path = Path(__file__).resolve().with_name(name + '.py')
    module = types.ModuleType('_campaign_' + name)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


EV = sibling('eval_yolo_cascade')
atomic, digest, sha, require = EV.atomic, EV.digest, EV.sha, EV.require


def group_metrics(matrix, indices):
    """Rows=GT, columns=prediction, optional BG at index 7; no true negatives.

    Micro is the sum of *classwise* TP/FP/FN, not a binary collapse of species.
    Conditional denominator excludes only predicted BG, never other classes.
    Zero denominators produce 0; macro includes all requested classes.
    """
    require(len(matrix) in (7, 8) and all(len(r) == len(matrix) for r in matrix),
            'Expected square 7x7 or 8x8 confusion')
    require(all(type(v) is int and v >= 0 for r in matrix for v in r), 'Invalid confusion counts')
    indices = list(indices)
    require(indices and len(set(indices)) == len(indices)
            and all(type(i) is int and 0 <= i < 7 for i in indices), 'Invalid target indices')
    per = {}
    for i in indices:
        tp = matrix[i][i]
        support = sum(matrix[i])
        fp = sum(row[i] for row in matrix) - tp
        fn = support - tp
        matched = sum(matrix[i][:7])
        per[CLASSES[i]] = dict(tp=tp, fp=fp, fn=fn, support=support, matched=matched,
                              precision=tp/(tp+fp) if tp+fp else 0.,
                              recall=tp/support if support else 0.,
                              f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                              classification_accuracy_conditional_matched=tp/matched if matched else 0.)
    tp, fp, fn, support, matched = (sum(r[k] for r in per.values())
                                   for k in ('tp', 'fp', 'fn', 'support', 'matched'))
    return dict(indices=indices, tp=tp, fp=fp, fn=fn, support=support, matched=matched,
                micro=dict(precision=tp/(tp+fp) if tp+fp else 0.,
                           recall=tp/(tp+fn) if tp+fn else 0.,
                           f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.),
                macro={k: sum(r[k] for r in per.values())/len(per) for k in ('precision', 'recall', 'f1')},
                classification_accuracy_conditional_matched=tp/matched if matched else 0.,
                per_class=per)


def classification_metrics(records):
    matrix = [[0] * 8 for _ in range(8)]
    for row in records:
        require(type(row['true']) is int and 0 <= row['true'] < 7, 'Invalid crop target')
        validate_probs(row['probabilities'])
        require(type(row['pred']) is int and 0 <= row['pred'] < 7
                and row['probabilities'][row['pred']] == max(row['probabilities']), 'Invalid class prediction')
        matrix[row['true']][row['pred']] += 1
    result = metric_groups(matrix)
    result['crop_accuracy_all7'] = result['all7']['micro']['recall']
    result['crop_accuracy_species_true_rows'] = result['species']['micro']['recall']
    result['confusion7'] = [row[:7] for row in matrix[:7]]
    return result


def metric_groups(matrix):
    return {'confusion': matrix, 'species': group_metrics(matrix, range(4)),
            'all7': group_metrics(matrix, range(7))}


def validate_probs(probabilities):
    require(len(probabilities) == 7 and all(type(v) in (int, float) and math.isfinite(v)
            and 0 <= v <= 1 for v in probabilities) and abs(sum(probabilities)-1) < 1e-5,
            'Expected seven YOLO-order probabilities')


def padded_box(box, pad, width=1920, height=1080):
    require(len(box) == 4 and all(math.isfinite(v) for v in box)
            and pad in (.15, .75), 'Invalid geometry/padding')
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return None
    pw, ph = (x1-x0)*pad, (y1-y0)*pad
    crop = (max(0, int(x0-pw)), max(0, int(y0-ph)),
            min(width, int(x1+pw)), min(height, int(y1+ph)))
    return crop if crop[2] > crop[0] and crop[3] > crop[1] else None


def score_policies(proposals, probabilities):
    """Already YOLO-order adapter outputs; never apply alphabetical remap twice."""
    require(len(proposals) == len(probabilities), 'Proposal count changed')
    result = {key: [] for key in POLICIES}
    for proposal, probs in zip(proposals, probabilities):
        if probs is not None:
            validate_probs(probs)
        cls = proposal['class'] if probs is None else max(range(7), key=probs.__getitem__)
        for key in POLICIES:
            score = proposal['yolo_confidence'] * (max(probs) if key == 'product' and probs is not None else 1.)
            result[key].append(dict(proposal, **{'class': cls, 'confidence': score,
                              'probabilities': probs, 'fallback': 'invalid_crop' if probs is None else None}))
    validate_geometry(proposals, result)
    return result


def validate_geometry(proposals, policies):
    require(set(policies) == set(POLICIES), 'Missing score policy')
    for key, rows in policies.items():
        require(len(rows) == len(proposals), 'Cascade lost proposals')
        for base, row in zip(proposals, rows):
            require(row['box'] == base['box'] and row['yolo_confidence'] == base['yolo_confidence'],
                    'Cascade changed geometry or detector score')
            probs = row['probabilities']
            if probs is not None:
                validate_probs(probs)
            require(row['class'] == (base['class'] if probs is None else max(range(7), key=probs.__getitem__)),
                    'Cascade changed class mapping')
            expected = base['yolo_confidence'] * (max(probs) if key == 'product' and probs is not None else 1.)
            require(row['confidence'] == expected, 'Incorrect score policy')


def tile_grid(width=1920, height=1080, size=512, stride=384):
    require(width > 0 and height > 0 and 0 < stride <= size, 'Invalid tile grid')
    def starts(length):
        last = max(0, length-size)
        return sorted(set([*range(0, last+1, stride), last]))
    return [(x, y, min(x+size, width), min(y+size, height))
            for y in starts(height) for x in starts(width)]


def crop_membership(path, entries_by_stem):
    path = Path(path)
    match = re.fullmatch(r'(.+)_(\d{3,})', path.stem)
    require(match is not None and match[1] in entries_by_stem, f'Crop is not an exact TEST member: {path}')
    entry, index = entries_by_stem[match[1]], int(match[2])
    require(path.parent.name in CLASSES and index < len(entry['gt']), f'Invalid crop class/ordinal: {path}')
    target = CLASSES.index(path.parent.name)
    require(entry['gt'][index]['class'] == target, f'Crop label differs from original GT: {path}')
    return {'image': entry['image'], 'gt_index': index, 'true': target}


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(RAID), f'Input outside RAID: {path}')
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def unchanged(fp):
    require(fingerprint(fp['path']) == fp, f'Input changed: {fp["path"]}')


def source_unchanged(contract):
    for fp in contract['sources'] + list(contract['detectors'].values()):
        unchanged(fp)
    for spec in contract['heads']:
        for fp in spec['sources'] + spec['checkpoint_sources']:
            unchanged(fp)
    for fp in contract['roi']['fingerprints'].values():
        unchanged(fp)


def data_unchanged(contract, stop):
    for entry in contract['all_entries']:
        stop.check()
        for key in ('image', 'label'):
            require(sha(entry[key]) == entry[key+'_sha256'], f'Changed final TEST {key}')
    for rows in contract['crops'].values():
        for row in rows:
            stop.check()
            unchanged({k: row[k] for k in ('path', 'sha256', 'bytes')})


def native_entries(Image, yaml):
    config = yaml.safe_load((DATA / 'data.yaml').read_text())
    EV.class_names(config['names'])
    require(Path(config['path']).resolve() == DATA and config['test'] == 'images/test', 'Unexpected TEST YAML')
    entries = []
    for path in sorted((DATA / 'images/test').rglob('*')):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        image = path.resolve(strict=True)
        label = (DATA / 'labels/test' / path.relative_to(DATA / 'images/test')).with_suffix('.txt').resolve(strict=True)
        require(image.is_relative_to(DATA / 'images/test') and label.is_relative_to(DATA / 'labels/test'),
                'TEST input escaped native split')
        text = label.read_text()
        require(all(line.strip() for line in text.splitlines()), 'Blank GT lines break original crop ordinals')
        gt = EV.parse_gt(text)
        for row in gt:
            row['box'] = [max(0., row['box'][0]), max(0., row['box'][1]),
                          min(1920., row['box'][2]), min(1080., row['box'][3])]
        with Image.open(image) as im:
            require(im.size == (1920, 1080), f'Non-native TEST size: {image}')
            im.verify()
        entries.append(dict(image=str(image), label=str(label), gt=gt,
                            image_sha256=sha(image), label_sha256=sha(label)))
    require(len(entries) == 55 and sum(len(e['gt']) for e in entries) == 2864, 'Expected native TEST 55 images / 2864 GT')
    require(len({e['image'] for e in entries}) == 55 and len({e['label'] for e in entries}) == 55,
            'Duplicate TEST images/labels')
    require(len({Path(e['image']).stem for e in entries}) == 55, 'Ambiguous native image stems')
    return entries


def crop_inventory(specs, entries, stop):
    by_stem = {Path(e['image']).stem: e for e in entries}
    result = {}
    for root in sorted({s['crop_root'] for s in specs}):
        test = Path(root).resolve(strict=True) / 'test'
        require(Path(root).name in POPULATIONS and test.resolve().is_relative_to(Path(root).resolve()),
                'Unexpected crop population')
        require({p.name for p in test.iterdir() if p.is_dir()} == set(CLASSES), 'Unexpected ImageFolder classes')
        rows, seen = [], set()
        for path in sorted(test.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            stop.check()
            require(path.parent.parent == test and path.resolve().is_relative_to(test), 'Crop outside TEST class folder')
            row = crop_membership(path, by_stem)
            key = (row['image'], row['gt_index'])
            require(key not in seen, 'Duplicate crop instance')
            seen.add(key)
            rows.append({**fingerprint(path), **row})
        require(len(rows) == POPULATIONS[Path(root).name], f'Unexpected TEST crop population: {root}: {len(rows)}')
        result[root] = rows
    return result


def statistics(values, unit, exclusions):
    values = sorted(float(x) for x in values)
    require(all(math.isfinite(x) and x >= 0 for x in values), 'Invalid timing')
    def percentile(q):
        if not values:
            return None
        offset = (len(values)-1)*q
        lo, hi = math.floor(offset), math.ceil(offset)
        return values[lo] + (values[hi]-values[lo])*(offset-lo)
    return dict(n=len(values), seconds=sum(values), mean=sum(values)/len(values) if values else None,
                p50=percentile(.5), p95=percentile(.95), unit=unit, exclusions=exclusions)


def detection_metrics(entries, predictions, np, ap_per_class, stop):
    require(len(entries) == len(predictions), 'Detection population changed')
    records = [{'gt': e['gt'], 'pipelines': {'yolo': p}} for e, p in zip(entries, predictions)]
    previous, original_candidates = EV.PIPELINES, EV.candidates

    def fast_candidates(gt, pred, class_aware=False):
        # Same double-precision xyxy arithmetic and exact tested greedy matcher;
        # zero-IoU pairs cannot pass any of the fixed positive thresholds.
        stop.check()
        if not gt or not pred:
            return []
        a, b = np.asarray([g['box'] for g in gt], dtype=np.float64), np.asarray([p['box'] for p in pred], dtype=np.float64)
        wh = np.maximum(0., np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]))
        inter = wh[:, :, 0]*wh[:, :, 1]
        area_a, area_b = np.maximum(0., a[:, 2:]-a[:, :2]), np.maximum(0., b[:, 2:]-b[:, :2])
        union = (area_a[:, 0]*area_a[:, 1])[:, None] + (area_b[:, 0]*area_b[:, 1])[None, :] - inter
        overlaps = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        mask = overlaps > 0
        if class_aware:
            mask &= np.asarray([g['class'] for g in gt])[:, None] == np.asarray([p['class'] for p in pred])[None, :]
        gs, ps = np.nonzero(mask)
        return [(int(g), int(p), float(overlaps[g, p])) for g, p in zip(gs, ps)]

    try:
        EV.PIPELINES, EV.candidates = ('yolo',), fast_candidates
        result = EV.metrics(records, np, ap_per_class)
    finally:
        EV.PIPELINES, EV.candidates = previous, original_candidates
    values = result['pipelines']['yolo']
    return {**values, **metric_groups(values['confusion']), 'evaluation_details': {
        k: v for k, v in result.items() if k != 'pipelines'}}


class Stop:
    def __init__(self):
        self.deadline = time.monotonic() + SOFT_SECONDS
        self.reason = None

    def signal(self, number, _frame):
        self.reason = f'signal {number}'

    def __call__(self):
        if time.monotonic() >= self.deadline:
            self.reason = self.reason or 'soft deadline 3h50'
        return self.reason is not None

    def check(self):
        if self():
            raise InterruptedError(self.reason)


class Store:
    """Single-writer campaign; each checkpoint binds identity, input and content."""
    def __init__(self, out, contract, resume):
        self.out, self.identity = Path(out), digest(contract)
        manifest = self.out / 'manifest.json'
        # JSON persists tuples (e.g. tile coordinates) as lists. Compare the
        # canonical representation without relaxing any content/hash checks.
        expected = json.loads(json.dumps({'identity': self.identity, 'contract': contract}, allow_nan=False))
        if resume:
            require(self.read(manifest) == expected, 'Resume identity mismatch: use a NEW output directory')
            self.state = self.read(self.out / 'state.json')
            require(self.state['identity'] == self.identity and re.fullmatch('[0-9a-f]{32}', self.state['wandb_run_id']),
                    'Invalid persisted campaign state')
        else:
            require(not manifest.exists(), 'Existing manifest; explicit --resume required')
            atomic(manifest, expected)
            self.state = dict(identity=self.identity, wandb_run_id=uuid.uuid4().hex, stages={},
                              status='initialized', wandb={'status': 'not_started'}, errors=[])
            self.save()

    @staticmethod
    def read(path):
        path = Path(path)
        require(not path.is_symlink(), f'Symlinked output: {path}')
        return json.loads(path.read_text())

    def path(self, relative):
        path = self.out / relative
        require(path.resolve().is_relative_to(self.out.resolve()) and not path.is_symlink(), 'Unsafe output path')
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save(self):
        self.state['updated_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic(self.path('state.json'), self.state)

    def get(self, relative, input_value):
        path = self.path(relative)
        if not path.exists():
            return None
        payload = self.read(path)
        checksum = payload.pop('sha256')
        require(checksum == digest(payload) and payload['identity'] == self.identity
                and payload['input_sha256'] == digest(input_value), f'Corrupt/stale checkpoint: {relative}')
        return payload['value']

    def put(self, relative, input_value, value):
        payload = dict(identity=self.identity, input_sha256=digest(input_value), value=value)
        payload['sha256'] = digest(payload)
        atomic(self.path(relative), payload)

    def error(self, stage, exception):
        message = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        secret = os.environ.get('WANDB_API_KEY')
        if secret:
            message = message.replace(secret, '[REDACTED]')
        error = dict(stage=stage, message=message, utc=dt.datetime.now(dt.timezone.utc).isoformat())
        self.state['errors'].append(error)
        atomic(self.path('errors.json'), self.state['errors'])
        print(f'ERROR {stage}: {message}', file=sys.stderr, flush=True)
        self.save()
        return message


def stage_key(name):
    return re.sub('[^A-Za-z0-9_.-]', '_', name)[:100] + '-' + digest(name)[:10]


class Campaign:
    def __init__(self, store, entries, runtime, stop):
        self.store, self.entries, self.runtime, self.stop = store, entries, runtime, stop
        self.tracker = None

    def run_stage(self, name, group, function):
        """Failures are independent; partial checkpoints survive; signals propagate."""
        key = stage_key(name)
        relative = f'stages/{key}/result.json'
        record = self.store.state['stages'].setdefault(name, {'group': group})
        try:
            self.stop.check()
            cached = self.store.get(relative, name)
            if cached is not None:
                for ref in cached['checkpoint_refs']:
                    require(sha(self.store.path(ref['path'])) == ref['sha256'], 'Changed stage checkpoint')
                result = cached['result']
            else:
                record.update(status='running', error=None)
                self.store.save()
                result = function(f'stages/{key}')
                self.stop.check()
                refs = [{'path': str(p.relative_to(self.store.out)), 'sha256': sha(p)}
                    for p in sorted(self.store.path(f'stages/{key}').rglob('*'))
                    if p.is_file() and p.suffix in {'.json', '.png'} and p.name != 'result.json']
                # ROI owns a separate image-checkpoint namespace.
                if group == 'gt_roi':
                    refs += [{'path': str(p.relative_to(self.store.out)), 'sha256': sha(p)}
                             for p in sorted((self.store.out / 'roi').rglob('*.json'))]
                self.store.put(relative, name, {'result': result, 'checkpoint_refs': refs})
            record.update(status=result.get('status', 'completed'), error=None, result=relative,
                          metrics=result.get('metrics'), timing=result.get('timing'),
                          population=result.get('population'), reason=result.get('reason'))
            self.store.save()
            print(f'{name}: {record["status"]}', flush=True)
            if self.tracker:
                self.tracker.stage(name, result)
            self.stop.check()
            return result
        except (InterruptedError, KeyboardInterrupt):
            record['status'] = 'interrupted'
            self.store.save()
            raise
        except Exception as exc:
            record.update(status='failed', error=self.store.error(name, exc))
            self.store.save()
            return None

    def checkpoint(self, path, inputs, compute):
        self.stop.check()
        value = self.store.get(path, inputs)
        if value is None:
            value = compute()
            # Whole image/batch is committed, even if signalled during inference.
            self.store.put(path, inputs, value)
        self.stop.check()
        return value

    def timed(self, function):
        torch = self.runtime['torch']
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = function()
        torch.cuda.synchronize()
        return value, time.perf_counter()-start

    def verify_entry(self, entry):
        for field in ('image', 'label'):
            require(sha(entry[field]) == entry[field + '_sha256'], f'Changed native {field}')

    def detector(self, name, prefix, checkpoint):
        torch, Image, YOLO = (self.runtime[k] for k in ('torch', 'Image', 'YOLO'))
        model, rows = None, []
        sliced = name == 'E2_sliced'
        try:
            for index, entry in enumerate(self.entries):
                self.verify_entry(entry)
                def compute(entry=entry):
                    nonlocal model
                    if model is None:
                        unchanged(checkpoint)
                        model = YOLO(checkpoint['path'])
                        EV.class_names(model.names)
                        model.model.float().eval().requires_grad_(False)
                        with Image.open(entry['image']) as image:
                            rgb = image.convert('RGB')
                        try:
                            self.predict_detector(model, rgb, sliced)  # Excluded warmup.
                        finally:
                            rgb.close()
                    with Image.open(entry['image']) as image:
                        rgb = image.convert('RGB')
                    try:
                        predictions, seconds = self.timed(lambda: self.predict_detector(model, rgb, sliced))
                    finally:
                        rgb.close()
                    self.verify_entry(entry)
                    return dict(image=entry['image'], predictions=predictions, seconds=seconds)
                row = self.checkpoint(f'{prefix}/images/{index:06d}.json', entry, compute)
                require(row['image'] == entry['image'] and len(row['predictions']) <= 300, 'Invalid detector checkpoint')
                rows.append(row)
        finally:
            model = None
            gc.collect()
            torch.cuda.empty_cache()
        metrics = self.checkpoint(f'{prefix}/metrics.json', digest(rows), lambda: detection_metrics(
            self.entries, [r['predictions'] for r in rows], self.runtime['np'], self.runtime['ap_per_class'], self.stop))
        return dict(records=rows, metrics=metrics, population=len(self.entries),
                    timing=statistics([r['seconds'] for r in rows], 'seconds/image', 'decode, warmup, load, SHA, JSON, metrics, network'))

    def predict_detector(self, model, image, sliced):
        def predict(crop, offset=(0, 0)):
            self.stop.check()
            options = dict(PREDICT, imgsz=512 if sliced else 1920)
            torch = self.runtime['torch']
            with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
                result = model.predict(source=crop, **options)[0]
            rows = []
            for box, cls, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                require(int(cls) == cls and 0 <= cls < 7 and math.isfinite(score) and .001 <= score <= 1., 'Invalid detector output')
                box = [box[0]+offset[0], box[1]+offset[1], box[2]+offset[0], box[3]+offset[1]]
                require(all(math.isfinite(v) for v in box), 'Nonfinite detector box')
                rows.append(dict(box=box, **{'class': int(cls), 'confidence': score, 'yolo_confidence': score}))
            return rows
        if not sliced:
            return predict(image)
        proposals = []
        for box in tile_grid(*image.size):
            with image.crop(box) as crop:
                proposals.extend(predict(crop, box[:2]))
        if not proposals:
            return []
        torch, batched_nms = self.runtime['torch'], self.runtime['batched_nms']
        # Stable input ordering; torchvision may choose a different winner for
        # exact score ties across versions, hence pinned runtime in manifest.
        proposals.sort(key=lambda p: (-p['confidence'], p['class'], *p['box']))
        keep = batched_nms(torch.tensor([p['box'] for p in proposals], dtype=torch.float32),
                           torch.tensor([p['confidence'] for p in proposals], dtype=torch.float32),
                           torch.tensor([p['class'] for p in proposals], dtype=torch.int64), .5).tolist()[:300]
        return [proposals[i] for i in keep]

    def gt_crops(self, session, crops, prefix):
        Image, rows, times = self.runtime['Image'], [], []
        warmed = False
        for start in range(0, len(crops), session.spec['batch_size']):
            batch = crops[start:start+session.spec['batch_size']]
            for fp in batch:
                unchanged({k: fp[k] for k in ('path', 'sha256', 'bytes')})
            def compute(batch=batch):
                nonlocal warmed
                adapter = session.get()
                with ExitStack() as stack:
                    # Keep native Image.open objects: SAM requires .filename.
                    images = [stack.enter_context(Image.open(r['path'])) for r in batch]
                    for image in images:
                        image.load()
                    if not warmed:
                        adapter.predict(images)
                        warmed = True
                    probabilities, seconds = self.timed(lambda: adapter.predict(images))
                require(len(probabilities) == len(batch), 'Classifier batch length changed')
                output = []
                for target, probs in zip(batch, probabilities):
                    validate_probs(probs)
                    output.append(dict(target, pred=max(range(7), key=probs.__getitem__), probabilities=probs))
                    unchanged({k: target[k] for k in ('path', 'sha256', 'bytes')})
                return dict(records=output, seconds=seconds)
            value = self.checkpoint(f'{prefix}/batches/{start:06d}.json', batch, compute)
            require([{k: r[k] for k in b} for r, b in zip(value['records'], batch)] == batch
                    and len(value['records']) == len(batch), 'GT batch population changed')
            rows.extend(value['records'])
            times.append(value['seconds'])
        return dict(records=rows, metrics=classification_metrics(rows), population=len(rows),
                    crop_root=session.spec['crop_root'],
                    timing=statistics(times, 'seconds/batch (not image)', 'decode, warmup, load, SHA, JSON, metrics, network'),
                    seconds_per_crop=sum(times)/len(rows) if rows else None)

    def cascade(self, session, e0, prefix):
        require(session.spec['supports_cascade'], 'SAM cannot classify uncleaned proposals')
        require(e0 is not None, 'E0 failed/unavailable; cascade blocked, GT diagnostic remains independent')
        rows, warmed = [], False
        Image = self.runtime['Image']
        for index, (entry, base) in enumerate(zip(self.entries, e0['records'])):
            require(entry['image'] == base['image'], 'E0/native image order mismatch')
            self.verify_entry(entry)
            inputs = {'entry': entry, 'E0': base, 'head': session.spec['spec_hash']}
            def compute(entry=entry, base=base):
                nonlocal warmed
                adapter = session.get()
                with Image.open(entry['image']) as source:
                    image = source.convert('RGB')
                try:
                    if not warmed:
                        warm_boxes = [padded_box(p['box'], session.spec['pad']) for p in base['predictions']]
                        warm_boxes = [b for b in warm_boxes if b is not None][:session.spec['batch_size']]
                        with ExitStack() as stack:
                            warm_images = [stack.enter_context(image.crop(b)) for b in warm_boxes]
                            if warm_images:
                                adapter.predict(warm_images)
                        warmed = True
                    def classify():
                        probabilities = [None] * len(base['predictions'])
                        valid = [(i, padded_box(p['box'], session.spec['pad'])) for i, p in enumerate(base['predictions'])]
                        valid = [(i, box) for i, box in valid if box is not None]
                        for start in range(0, len(valid), session.spec['batch_size']):
                            self.stop.check()
                            batch = valid[start:start+session.spec['batch_size']]
                            with ExitStack() as stack:
                                images = [stack.enter_context(image.crop(box)) for _, box in batch]
                                probs = adapter.predict(images)
                            require(len(probs) == len(batch), 'Cascade batch length changed')
                            for (i, _), prob in zip(batch, probs):
                                probabilities[i] = prob
                        return score_policies(base['predictions'], probabilities)
                    policies, seconds = self.timed(classify)
                finally:
                    image.close()
                self.verify_entry(entry)
                return dict(image=entry['image'], policies=policies, classifier_seconds=seconds,
                            seconds=seconds+base['seconds'],
                            invalid_crops=sum(p['fallback'] is not None for p in policies['label_only']))
            row = self.checkpoint(f'{prefix}/images/{index:06d}.json', inputs, compute)
            require(row['image'] == entry['image'], 'Cascade image mismatch')
            validate_geometry(base['predictions'], row['policies'])
            rows.append(row)
        require(len(rows) == len(self.entries), 'Cascade lost TEST images')
        metrics = {}
        for policy in POLICIES:
            metrics[policy] = self.checkpoint(f'{prefix}/metrics_{policy}.json', digest(rows),
                lambda policy=policy: detection_metrics(self.entries, [r['policies'][policy] for r in rows],
                                                        self.runtime['np'], self.runtime['ap_per_class'], self.stop))
        return dict(records=rows, metrics=metrics, population=len(rows),
                    proposal_count=sum(len(r['policies']['label_only']) for r in rows),
                    invalid_crops=sum(r['invalid_crops'] for r in rows),
                    timing=statistics([r['seconds'] for r in rows], 'seconds/image (E0 + classifier)', 'decode, warmup, load, SHA, JSON, metrics, network'),
                    classifier_timing=statistics([r['classifier_seconds'] for r in rows], 'seconds/image (classifier only)', 'decode, E0, warmup, load, SHA, JSON, metrics, network'))

    def roi(self, adapter, roi_contract, _prefix):
        require(roi_contract['error'] is None, f'ROI provenance unavailable: {roi_contract["error"]}')
        result = adapter.evaluate_roi(self.entries, self.store.out, self.stop)
        require(len(result['records']) == len(self.entries), 'ROI image population changed')
        rows = []
        for entry, record in zip(self.entries, result['records']):
            require(record['image'] == entry['image'] and record['true'] == [g['class'] for g in entry['gt']]
                    and len(record['true']) == len(record['pred']) == len(record['probabilities']), 'ROI GT order changed')
            rows.extend(dict(image=record['image'], gt_index=i, true=t, pred=p, probabilities=prob)
                        for i, (t, p, prob) in enumerate(zip(record['true'], record['pred'], record['probabilities'])))
        timings = [Store.read(self.store.out / 'roi/images' / f'{i:06d}.json')['seconds'] for i in range(len(self.entries))]
        return dict(records=rows, metrics=classification_metrics(rows), population=len(rows),
                    fingerprints=result['fingerprints'],
                    timing=statistics(timings, 'seconds/image (native RoI)', 'load, SHA, JSON; includes decode and cold first image'))

    def cam(self, session, gt_result, prefix):
        if session.spec['kind'] == 'ensemble':
            return {'status': 'skipped', 'reason': 'Adapter ensemble sem Grad-CAM'}
        require(gt_result is not None, 'GT predictions unavailable for deterministic CAM selection')
        chosen = []
        for correct in (False, True):
            chosen.extend([r for r in gt_result['records'] if (r['true'] == r['pred']) == correct][:3])
        plt, images = self.runtime['plt'], []
        for i, row in enumerate(chosen):
            self.stop.check()
            unchanged({k: row[k] for k in ('path', 'sha256', 'bytes')})
            with self.runtime['Image'].open(row['path']) as image:
                cam, pred = session.get().gradcam(image)
                require(pred == row['pred'], 'CAM prediction differs from saved classifier prediction')
                figure, axis = plt.subplots(figsize=(5, 4))
                try:
                    axis.imshow(image.convert('RGB'))
                    axis.imshow(cam, cmap='jet', alpha=.4, vmin=0, vmax=1)
                    axis.set_title(f'GT {CLASSES[row["true"]]} / pred {CLASSES[pred]}')
                    axis.axis('off')
                    relative = f'{prefix}/cam_{i:02d}.png'
                    figure.savefig(self.store.path(relative), dpi=120, bbox_inches='tight')
                finally:
                    plt.close(figure)
            images.append(dict(path=relative, sha256=sha(self.store.path(relative)), crop=row['path'], true=row['true'], pred=pred))
        self.store.put(f'{prefix}/selection.json', session.spec['spec_hash'], images)
        return {'images': images, 'population': len(images)}


class HeadSession:
    def __init__(self, module, spec):
        self.module, self.spec, self.adapter = module, spec, None

    def get(self):
        if self.adapter is None:
            self.adapter = self.module.load_head(self.spec)
        return self.adapter

    def close(self):
        if self.adapter is not None:
            self.adapter.close()
            self.adapter = None


class Tracking:
    """Tracking failures never masquerade as uploaded results or kill metrics."""
    def __init__(self, store, disabled):
        self.store, self.run, self.wandb = store, None, None
        store.state['wandb'] = dict(status='disabled_smoke' if disabled else 'initializing',
                                   run_id=store.state['wandb_run_id'], entity='pestline', project='fly-species',
                                   stages={})
        store.save()
        if disabled:
            return
        try:
            require(os.environ.get('WANDB_MODE', 'online') == 'online', 'W&B must be online')
            require(bool(os.environ.get('WANDB_API_KEY')), 'Inherited WANDB_API_KEY missing')
            import wandb
            self.wandb = wandb
            run = wandb.init(entity='pestline', project='fly-species', id=store.state['wandb_run_id'],
                             resume='allow', job_type='test-evaluation', name='test-campaign-'+store.state['wandb_run_id'][:8],
                             dir=str(store.out), mode='online',
                             config={'identity': store.identity, 'split': 'test', 'confidence': .25, 'iou': .5},
                             settings=wandb.Settings(init_timeout=60))
            require(run is not None and run.id == store.state['wandb_run_id']
                    and run.entity == 'pestline' and run.project == 'fly-species', 'W&B run identity mismatch')
            self.run = run  # Never attach logging to an unverified/old training run.
            store.state['wandb'].update(status='active', url=self.run.url)
            store.save()
        except Exception as exc:
            self.fail('init', exc)

    def fail(self, operation, exc):
        self.store.state['wandb'].update(status='failed', failed_operation=operation)
        self.store.error('wandb/' + operation, exc)

    def stage(self, name, result):
        if self.run is None:
            return
        try:
            values = {}
            def flatten(prefix, obj):
                if isinstance(obj, dict):
                    for key, val in obj.items():
                        flatten(prefix+'/'+key, val)
                elif type(obj) in (int, float) and math.isfinite(obj):
                    values[prefix] = obj
            flatten(name, {'metrics': result.get('metrics'), 'timing': result.get('timing')})
            for i, image in enumerate(result.get('images', [])):
                path = self.store.path(image['path'])
                require(sha(path) == image['sha256'], 'Changed diagnostic image')
                values[f'{name}/image_{i}'] = self.wandb.Image(str(path))
            if values:
                self.run.log(values)
            self.store.state['wandb']['stages'][name] = 'submitted_to_sdk_not_yet_confirmed'
            self.store.save()
        except Exception as exc:
            self.store.state['wandb']['stages'][name] = 'failed'
            self.fail('stage/' + name, exc)

    def finish(self, contract, stop):
        stop.check()
        if self.run is None:
            return
        try:
            artifact = self.wandb.Artifact('test-campaign-'+self.store.state['wandb_run_id'], type='evaluation',
                                           metadata={'identity': self.store.identity})
            paths = sorted({*self.store.out.glob('*.json'), *self.store.out.glob('*.md'),
                            *(self.store.out / 'stages').rglob('*.json'), *(self.store.out / 'stages').rglob('*.png'),
                            *(self.store.out / 'roi').rglob('*.json')})
            paths = [p for p in paths if p.name != 'upload_hashes.json']
            file_hashes = {str(p.relative_to(self.store.out)): sha(p) for p in paths}
            atomic(self.store.path('upload_hashes.json'), file_hashes)
            for path in [*paths, self.store.path('upload_hashes.json')]:
                artifact.add_file(str(path), name=str(path.relative_to(self.store.out)))
            logged = self.run.log_artifact(artifact)
            logged.wait(timeout=180)
            stop.check()
            self.store.state['wandb'].update(results_artifact='confirmed', artifact_id=logged.id)
            self.store.save()
            write_report(self.store, contract)
            final_report = self.wandb.Artifact('test-report-'+self.store.state['wandb_run_id'], type='report')
            for name in ('report.md', 'summary.json', 'state.json', 'manifest.json'):
                final_report.add_file(str(self.store.path(name)), name=name)
            self.run.log_artifact(final_report).wait(timeout=180)
            stop.check()
            self.store.state['wandb']['report_artifact'] = 'confirmed'
        except (InterruptedError, KeyboardInterrupt):
            raise
        except Exception as exc:
            self.fail('artifact', exc)
        finally:
            try:
                # Do not start a potentially blocking network flush after a signal.
                if not stop():
                    self.run.finish(exit_code=0 if self.store.state['status'] == 'completed' else 1)
                    if self.store.state['wandb']['status'] != 'failed':
                        self.store.state['wandb']['status'] = 'finished_sdk'
            except Exception as exc:
                self.fail('finish', exc)
            self.store.save()
        stop.check()


def confusion_plots(campaign, name, metrics, prefix):
    plt, images = campaign.runtime['plt'], []
    variants = metrics if name.startswith('cascade/') else {'main': metrics}
    for policy, values in variants.items():
        campaign.stop.check()
        matrix = values['confusion']
        figure, axis = plt.subplots(figsize=(8, 7))
        try:
            drawing = axis.imshow(matrix, cmap='Blues')
            labels = [*CLASSES, 'BG']
            axis.set(xticks=range(8), yticks=range(8), xticklabels=labels, yticklabels=labels,
                     xlabel='Predição', ylabel='GT', title=f'{name} / {policy}')
            for i in range(8):
                for j in range(8):
                    axis.text(j, i, str(matrix[i][j]), ha='center', va='center', fontsize=8)
            figure.colorbar(drawing, ax=axis)
            relative = f'{prefix}/{policy}.png'
            figure.savefig(campaign.store.path(relative), dpi=120, bbox_inches='tight')
            images.append({'path': relative, 'sha256': sha(campaign.store.path(relative))})
        finally:
            plt.close(figure)
    campaign.store.put(f'{prefix}/images.json', name, images)
    return {'images': images}


def table_rows(state, groups):
    result = []
    for name, stage in state['stages'].items():
        if stage['group'] not in groups or stage.get('status') != 'completed' or not stage.get('metrics'):
            continue
        variants = stage['metrics'] if stage['group'] == 'cascade' else {'': stage['metrics']}
        for policy, metrics in variants.items():
            result.append((name + ('/'+policy if policy else ''), metrics, stage))
    return result


def write_report(store, contract):
    state = store.state
    summary = {'identity': store.identity, 'status': state['status'], 'wandb': state['wandb'],
               'native_available_images': 55, 'native_available_gt': 2864,
               'evaluated_images': len(contract['selected_entries']),
               'evaluated_gt': sum(len(e['gt']) for e in contract['selected_entries']),
               'crop_populations': {k: len(v) for k, v in contract['crops'].items()},
               'stages': state['stages'], 'warnings': WARNINGS}
    atomic(store.path('summary.json'), summary)
    lines = ['# Avaliação TEST — campanha consolidada', '',
             f'**Estado:** {state["status"]} | **Identidade:** {store.identity}', '',
             f'TEST nativo: **55 imagens / 2864 GT**. Nesta execução: '
             f'**{summary["evaluated_images"]} imagens / {summary["evaluated_gt"]} GT**.', '',
             '**SMOKE — não representa o TEST completo.**' if contract['limit_images'] is not None else 'Avaliação integral do TEST nativo.', '',
             '## Tracking e reprodutibilidade', '',
             f'W&B: **{state["wandb"]["status"]}**, pestline/fly-species, run `{state["wandb_run_id"]}`.',
             f'Artefato de resultados: {state["wandb"].get("results_artifact", "não confirmado")}; '
             f'relatório: {state["wandb"].get("report_artifact", "não confirmado")}.',
             'Estado local é atualizado após confirmação/finalização do SDK; o artefato de relatório '
             'registra o estado anterior à sua própria confirmação. Erros de tracking ficam explícitos em errors.json.',
             'manifest.json contém fontes, 18 specs, checkpoints, versões e SHA256 de cada entrada/crop. '
             'stages/ e roi/ contêm predições e checkpoints; upload_hashes.json identifica os bytes enviados.', '',
             '## Inventário e estados', '', '| Experimento | Grupo | Estado | População | Motivo/erro |', '|---|---|---|---:|---|']
    for name, s in state['stages'].items():
        reason = (s.get('reason') or s.get('error') or '').replace('\n', ' ').replace('|', '/')
        lines.append(f'| {name} | {s["group"]} | {s.get("status", "pending")} | {s.get("population") or "—"} | {reason[:350]} |')
    def fmt(value):
        return '—' if value is None else f'{value:.4f}'
    for title, groups in [('Detecção + cascata (mesmas imagens TEST)', {'detector', 'cascade'}),
                           ('Diagnósticos GT-crop / GT-RoI (populações distintas)', {'gt_crop', 'gt_roi'})]:
        accuracy_notes = []
        lines += ['', '## '+title, '',
                  '| Experimento | GT esp./7 | Pµ esp. | Rµ esp. | F1µ esp. | Pµ 7 | Rµ 7 | F1µ 7 | Acc cond. esp./7 | mAP50/50:95 | Tempo média/p50/p95 (s) |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        for name, metrics, stage in table_rows(state, groups):
            species, all7 = metrics['species'], metrics['all7']
            rates = [group['micro'][key] for group in (species, all7) for key in ('precision', 'recall', 'f1')]
            timing = stage.get('timing') or {}
            lines.append(f'| {name} | {species["support"]}/{all7["support"]} | '
                         + ' | '.join(fmt(v) for v in rates) + ' | '
                         + '/'.join(fmt(g['classification_accuracy_conditional_matched']) for g in (species, all7)) + ' | '
                         + '/'.join(fmt(metrics.get(k)) for k in ('mAP50', 'mAP50_95')) + ' | '
                         + '/'.join(fmt(timing.get(k)) for k in ('mean', 'p50', 'p95')) + ' '+timing.get('unit', '')+' |')
            if stage['group'] in {'gt_crop', 'gt_roi'}:
                accuracy_notes.append(f'- Acurácia GT {name}: geral={fmt(metrics["crop_accuracy_all7"])}, '
                                      f'espécies por linha verdadeira={fmt(metrics["crop_accuracy_species_true_rows"])}.')
        lines += ['', *accuracy_notes]
        lines += ['', '### Macro e métricas por classe (todas as 7)', '',
                  '| Experimento / classe | Suporte | Precisão | Recall | F1 |', '|---|---:|---:|---:|---:|']
        for name, metrics, _ in table_rows(state, groups):
            for group in ('species', 'all7'):
                macro = metrics[group]['macro']
                lines.append(f'| {name} / macro {group} | {metrics[group]["support"]} | '
                             + ' | '.join(fmt(macro[k]) for k in ('precision', 'recall', 'f1')) + ' |')
            for cls, row in metrics['all7']['per_class'].items():
                lines.append(f'| {name} / {cls} | {row["support"]} | '
                             + ' | '.join(fmt(row[k]) for k in ('precision', 'recall', 'f1')) + ' |')
    lines += ['', '## Mudanças observadas contra E0 (sem seleção de implantação)', '',
              '| Experimento | ΔP esp. | ΔR esp. | ΔF1 esp. | ΔP 7 | ΔR 7 | ΔF1 7 | ΔAcc cond. esp. | ΔAcc cond. 7 | Melhoras observadas |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    det_rows = table_rows(state, {'detector', 'cascade'})
    baseline = next((m for n, m, _ in det_rows if n == 'detector/E0'), None)
    if baseline:
        for name, metrics, _ in det_rows:
            if name == 'detector/E0':
                continue
            deltas = [(g+'/'+k, metrics[g]['micro'][k]-baseline[g]['micro'][k])
                      for g in ('species', 'all7') for k in ('precision', 'recall', 'f1')]
            deltas += [(g+'/acc_conditional', metrics[g]['classification_accuracy_conditional_matched']
                        - baseline[g]['classification_accuracy_conditional_matched']) for g in ('species', 'all7')]
            lines.append('| '+name+' | '+' | '.join(f'{v:+.4f}' for _, v in deltas)+' | '
                         + (', '.join(k for k, v in deltas if v > 0) or 'nenhuma nestas métricas')+' |')
    else:
        lines.append('\nE0 indisponível: nenhuma comparação contra baseline foi fabricada.')
    lines += ['', '## Limitações e interpretação', ''] + ['- '+warning for warning in WARNINGS]
    lines += ['- Interseção comum entre populações de crops não calculada; suportes separados são obrigatórios.',
              '- Proveniência por nome/ordinal/classe é verificada, mas não prova ausência de leakage de placa/local/período no treinamento.',
              '', '## Tempos: exclusões por etapa', '']
    for name, stage in state['stages'].items():
        if stage.get('timing'):
            lines.append(f'- {name}: {stage["timing"]["exclusions"]}.')
    atomic(store.path('report.md'), '\n'.join(lines)+'\n', text=True)


def prepare_environment(args):
    EV.allocation_guard()  # Must precede all heavy imports and ANY mkdir.
    require(args.limit_images is None or 0 < args.limit_images < 55, '--limit-images smoke must be between 1 and 54')
    require(not args.no_wandb or args.limit_images is not None, '--no-wandb is smoke-only')
    require(os.environ.get('SLURM_NTASKS', '1') == '1' and os.environ.get('LOCAL_RANK', '0') == '0', 'Single process required')
    require(args.out.is_absolute(), '--out must be absolute')
    args.out = args.out.resolve()
    require(args.out.is_relative_to(RAID), '--out must be on RAID')
    protected = [RAID / 'datasets', REPO, RAID / 'secrets', RAID / 'containers',
                 RUNS / 'architecture-night-v1-20260907', RUNS / 'architecture-followup-v1-20260908',
                 *[p.parents[3] for p in DETECTORS.values()]]
    # Reject writing anywhere inside any original classifier export directory.
    protected += [RUNS / n for n in ('E3_swin', 'E3b_convnext', 'E3c_focal', 'E3d_convnext_s',
                                    'E4_convnext_sam', 'E5_convnext_512', 'E3_ensemble_tta')]
    for path in protected:
        path = path.resolve()
        require(not args.out.is_relative_to(path) and not path.is_relative_to(args.out), 'Output overlaps original data/source/artifacts')
    if args.resume:
        require((args.out / 'manifest.json').is_file() and (args.out / 'state.json').is_file(), '--resume requires manifest and state')
    else:
        # A preflight interrupted before creating a manifest may leave only its
        # lock file; acquiring flock below still prevents concurrent writers.
        require(not args.out.exists() or not any(p.name != '.lock' for p in args.out.iterdir()),
                'Nonempty output: use --resume')
    if args.out.exists():
        require(not any(p.is_symlink() for p in args.out.rglob('*')), 'Symlinked campaign output')
    for key, suffix in {
        'XDG_CACHE_HOME': 'xdg', 'XDG_CONFIG_HOME': 'config', 'TORCH_HOME': 'torch',
        'HF_HOME': 'hf', 'HF_HUB_CACHE': 'hf/hub', 'HUGGINGFACE_HUB_CACHE': 'hf/hub',
        'TRANSFORMERS_CACHE': 'hf/transformers', 'PIP_CACHE_DIR': 'pip',
        'YOLO_CONFIG_DIR': 'ultralytics', 'WANDB_DIR': 'wandb', 'WANDB_CACHE_DIR': 'wandb/cache',
        'WANDB_CONFIG_DIR': 'wandb/config', 'WANDB_DATA_DIR': 'wandb/data',
        'MPLCONFIGDIR': 'matplotlib', 'CUDA_CACHE_PATH': 'cuda', 'TRITON_CACHE_DIR': 'triton',
        'TORCHINDUCTOR_CACHE_DIR': 'inductor', 'TMPDIR': 'tmp',
    }.items():
        value = Path(os.environ.get(key, str(RAID / 'cache/test_campaign' / suffix)))
        require(value.is_absolute() and value.resolve().is_relative_to(RAID)
                and not value.resolve().is_relative_to(RAID / 'datasets'), f'{key} must be a RAID cache outside datasets')
        value.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(value.resolve())
    import tempfile
    tempfile.tempdir = os.environ['TMPDIR']
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', YOLO_AUTOINSTALL='false',
                      YOLO_OFFLINE='true', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    sys.dont_write_bytecode = True
    args.out.mkdir(parents=True, exist_ok=True)


def runtime():
    EV.allocation_guard()
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one allocated CUDA GPU required')
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    import numpy as np
    import yaml
    from PIL import Image
    from ultralytics import YOLO
    from ultralytics.utils.metrics import ap_per_class
    from torchvision.ops import batched_nms
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    versions = {}
    for package in ('torch', 'torchvision', 'ultralytics', 'numpy', 'Pillow', 'PyYAML', 'matplotlib', 'wandb'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    provenance = dict(versions=versions, python=sys.version, cuda=torch.version.cuda,
                      cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
                      container=os.environ.get('APPTAINER_CONTAINER') or os.environ.get('APPTAINER_NAME'))
    return dict(torch=torch, np=np, yaml=yaml, Image=Image, YOLO=YOLO, ap_per_class=ap_per_class,
                batched_nms=batched_nms, plt=plt, provenance=provenance)


def build_contract(args, rt, heads, roi, stop):
    specs = heads.discover_heads(repo_root=REPO)
    require(len(specs) == 18, f'Expected 18 classifier specs, discovered {len(specs)}')
    entries = native_entries(rt['Image'], rt['yaml'])
    crops = crop_inventory(specs, entries, stop)
    roi_info = {'fingerprints': {}, 'error': None}
    try:
        # Provenance-only helper: no torch, deserialization or CUDA. Evaluation
        # itself uses the public evaluate_roi interface exclusively.
        _, _, roi_info['fingerprints'] = roi._artifacts()
    except Exception as exc:
        roi_info['error'] = f'{type(exc).__name__}: {exc}'
    sources = [Path(__file__).resolve(), Path(EV.__file__), Path(heads.__file__), Path(roi.__file__),
               DATA / 'data.yaml', REPO / 'fly-det/fly_det/sahi_inference.py']
    return dict(schema=1, sources=[fingerprint(p) for p in sources],
                heads=specs, detectors={name: fingerprint(path) for name, path in DETECTORS.items()},
                roi=roi_info, all_entries=entries, selected_entries=entries[:args.limit_images], crops=crops,
                runtime=rt['provenance'], limit_images=args.limit_images, no_wandb=args.no_wandb,
                protocol=dict(classes=list(CLASSES), species=[0, 1, 2, 3], confusion_conf=.25,
                              confusion_iou=.5, predict=PREDICT, tiles=tile_grid(), tile_imgsz=512,
                              merge='torchvision.ops.batched_nms CPU float32, class-aware .5, max300',
                              policies=list(POLICIES), soft_seconds=SOFT_SECONDS))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--out', type=Path, required=True)
    result.add_argument('--limit-images', type=int, help='Smoke: first N native TEST images and only their existing crops')
    result.add_argument('--no-wandb', action='store_true', help='Allowed only with --limit-images')
    result.add_argument('--resume', action='store_true')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    stop = Stop()
    previous = {sig: signal.signal(sig, stop.signal) for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
    store = tracker = contract = None
    lock_fd = None
    try:
        prepare_environment(args)
        lock_fd = os.open(args.out / '.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stop.check()
        rt = runtime()
        heads, roi = sibling('test_campaign_heads'), sibling('test_campaign_roi')
        contract = build_contract(args, rt, heads, roi, stop)
        store = Store(args.out, contract, args.resume)
        campaign = Campaign(store, contract['selected_entries'], rt, stop)
        tracker = campaign.tracker = Tracking(store, args.no_wandb)
        inventory = [('detector/'+name, 'detector') for name in (*DETECTORS, 'E2_sliced')]
        for spec in contract['heads']:
            inventory += [('gt_crop/'+spec['id'], 'gt_crop'), ('cascade/'+spec['id'], 'cascade'), ('cam/'+spec['id'], 'cam')]
        inventory.append(('gt_roi/original', 'gt_roi'))
        for name, group in inventory:
            store.state['stages'].setdefault(name, dict(group=group, status='pending'))
        store.state['status'] = 'running'
        store.save()
        e0 = None
        for name in (*DETECTORS, 'E2_sliced'):
            result = campaign.run_stage('detector/'+name, 'detector',
                                        lambda prefix, name=name: campaign.detector(name, prefix, contract['detectors']['E2' if name == 'E2_sliced' else name]))
            if name == 'E0':
                e0 = result
            write_report(store, contract)
        selected = {e['image'] for e in campaign.entries}
        for spec in contract['heads']:
            session = HeadSession(heads, spec)
            try:
                crops = [r for r in contract['crops'][spec['crop_root']] if r['image'] in selected]
                gt = campaign.run_stage('gt_crop/'+spec['id'], 'gt_crop', lambda prefix: campaign.gt_crops(session, crops, prefix))
                if gt is None:
                    session.close()
                if spec['supports_cascade']:
                    casc = campaign.run_stage('cascade/'+spec['id'], 'cascade', lambda prefix: campaign.cascade(session, e0, prefix))
                    if casc is None:
                        session.close()
                else:
                    campaign.run_stage('cascade/'+spec['id'], 'cascade', lambda _: {
                        'status': 'skipped', 'reason': 'SAM GT-only: propostas E0 não passaram pela limpeza SAM original'})
                campaign.run_stage('cam/'+spec['id'], 'cam', lambda prefix: campaign.cam(session, gt, prefix))
            finally:
                session.close()
            write_report(store, contract)
        campaign.run_stage('gt_roi/original', 'gt_roi', lambda prefix: campaign.roi(roi, contract['roi'], prefix))
        for name, stage in list(store.state['stages'].items()):
            if stage.get('metrics') and stage.get('status') == 'completed':
                campaign.run_stage('confusion/'+name, 'diagnostic',
                                    lambda prefix, name=name, metrics=stage['metrics']: confusion_plots(campaign, name, metrics, prefix))
        stop.check()
        source_unchanged(contract)
        data_unchanged(contract, stop)
        store.state['status'] = 'completed_with_errors' if any(s.get('status') == 'failed' for s in store.state['stages'].values()) else 'completed'
        store.save()
        write_report(store, contract)
        tracker.finish(contract, stop)
        if store.state['wandb']['status'] == 'failed':
            store.state['status'] = 'completed_with_tracking_errors' if store.state['status'] == 'completed' else 'completed_with_errors'
        store.save()
        write_report(store, contract)
        return 0 if store.state['status'] == 'completed' else 1
    except (InterruptedError, KeyboardInterrupt) as exc:
        if store is not None:
            store.state.update(status='interrupted', interruption=stop.reason or str(exc))
            if store.state['wandb']['status'] not in {'disabled_smoke', 'failed', 'finished_sdk'}:
                store.state['wandb']['status'] = 'interrupted_sync_unconfirmed'
            store.save()
            write_report(store, contract)
        print('Campaign interrupted; resume the identical manifest with --resume.', file=sys.stderr)
        return 75
    except Exception as exc:
        if store is not None:
            store.state['status'] = 'failed'
            store.error('campaign', exc)
            write_report(store, contract)
        else:
            print(f'Campaign preflight failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())