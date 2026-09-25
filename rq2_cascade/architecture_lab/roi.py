"""Frozen GT-ROI feasibility ONLY, not detection evaluation or mAP.

Use python -m architecture_lab.roi {cache,train} --help. Heavy imports occur
only after Slurm + Apptainer guards. Cache reads images/{train,val} and existing
labels/{train,val}, never TEST. Caller must guarantee board/trap/time split
provenance and that E0 was trained without validation leakage.

Native RGB 1920x1080 /255, right/bottom zero padding to 1920x1088, no resize.
Full frozen detector forward (INCLUDING detector head overhead), hooks P3/P4,
aligned 3x3 RoIAlign, channel concatenation, CPU float16 cache per image.
Only the small classifier is trained; no augmentation or detector fine-tuning.
Per-split CPU feature cache is bounded to 2 GiB (or one oversized image blob).
Cache restart automatically validates/skips complete images. Immutable inputs
are identified by paths/size/mtime_ns (detector and YAML additionally SHA256).
Training budget is per invocation, checked BETWEEN epochs including validation;
one complete epoch and checkpoint/export IO can exceed the deadline. Signals
discard an incomplete epoch, return 75, and preserve last complete epoch/zero.
CUDA calls/IO cannot be preempted safely. Resume ONLY trusted local checkpoints
(pickle). One writer per output directory; no concurrent cache/train writers.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import uuid

from .train import (
    EXPECTED_CLASSES, SELECTION_METRIC, RAID, compute_metrics, Control,
    SignalInterrupted, BudgetReached, _atomic_write, _write_json, _digest,
    _positive_int, _positive_float, _capture_rng, _restore_rng,
    _save_predictions, _log_metrics, _check_split_tree,
)

YOLO_CLASSES = ("MD", "MV", "MC", "MF", "INS", "NOISE", "MAR")
CLASS_MAP = tuple(EXPECTED_CLASSES.index(name) for name in YOLO_CLASSES)
STRIDES = (8, 16)
SCHEMA = 1


def padded_size(width=1920, height=1080):
    if (width, height) != (1920, 1080):
        raise ValueError(f"Expected native 1920x1080, got {width}x{height}; resize forbidden")
    return ((width + 31) // 32 * 32, (height + 31) // 32 * 32)


def parse_label(line, width=1920, height=1080):
    """YOLO class/cx/cy/w/h -> alphabetical class and native xyxy, no pad shift."""
    padded_size(width, height)
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"Expected exactly five YOLO detection fields: {line!r}")
    label = int(fields[0])
    if not 0 <= label < len(YOLO_CLASSES):
        raise ValueError(f"Invalid YOLO class: {label}")
    cx, cy, bw, bh = map(float, fields[1:])
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (cx, cy, bw, bh)) or min(bw, bh) <= 0:
        raise ValueError(f"Invalid normalized bbox: {line!r}")
    box = ((cx - bw / 2) * width, (cy - bh / 2) * height,
           (cx + bw / 2) * width, (cy + bh / 2) * height)
    # Tolerate only annotation decimal-rounding error, not substantive clipping.
    if box[0] < -0.01 or box[1] < -0.01 or box[2] > width + 0.01 or box[3] > height + 0.01:
        raise ValueError(f"BBox extends outside native image: {line!r}")
    return CLASS_MAP[label], [max(0., box[0]), max(0., box[1]), min(float(width), box[2]), min(float(height), box[3])]


def verify_feature_shape(shape, stride, padded=(1920, 1088)):
    w, h = padded
    if (len(shape) != 4 or shape[0] != 1 or shape[1] < 1
            or shape[2] * stride != h or shape[3] * stride != w):
        raise ValueError(f"P3/P4 hook mismatch: shape={tuple(shape)}, expected NCHW "
                         f"[1,C,{h // stride},{w // stride}], isotropic stride={stride}; "
                         "inspect YOLO.model layers; refusing automatic fallback")


def _fingerprint(path, sha=False):
    path = Path(path)
    stat = path.stat()
    result = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if sha:
        with path.open("rb") as handle:
            result["sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    return result


def _sources_unchanged(contract):
    sources = [contract["detector"], contract["yaml"]]
    sources += [entry[key] for entry in contract["entries"] for key in ("image", "label")]
    for item in sources:
        if _fingerprint(item["path"], "sha256" in item) != item:
            raise ValueError(f"Source changed since cache manifest: {item['path']}")


def _manifest(args):
    import yaml

    root = Path(args.data)
    config = root / "data.yaml"
    if config.is_symlink():
        raise ValueError("Dataset YAML must not be a symlink")
    document = yaml.safe_load(config.read_text())
    names = document.get("names")
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    if tuple(names or ()) != YOLO_CLASSES:
        raise ValueError(f"YOLO class order must be {YOLO_CLASSES}, got {names}")
    entries = []
    for split in ("train", "val"):
        if document.get(split) != f"images/{split}":
            raise ValueError(f"Expected YAML {split}: images/{split}; refusing ambiguous split")
        for tree in ("images", "labels"):
            if (root / tree).is_symlink():
                raise ValueError(f"Symlinked {tree} tree forbidden; must not redirect into TEST")
            _check_split_tree(root, f"{tree}/{split}")
        images = sorted(p for p in (root / "images" / split).rglob("*")
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        if args.limit_images:
            images = images[:args.limit_images]  # Deterministic cap PER split.
        if not images:
            raise ValueError(f"Empty {split} image selection")
        for image in images:
            relative = image.relative_to(root / "images" / split)
            label = (root / "labels" / split / relative).with_suffix(".txt")
            parsed = [parse_label(line) for line in label.read_text().splitlines() if line.strip()]
            entries.append({"split": split, "image": _fingerprint(image), "label": _fingerprint(label),
                            "targets": [v[0] for v in parsed], "boxes": [v[1] for v in parsed]})
    return {"schema": SCHEMA, "root": str(root), "yaml": _fingerprint(config, True),
            "detector": _fingerprint(args.detector, True), "layers": args.layers,
            "strides": list(STRIDES), "entries": entries, "limit_images_per_split": args.limit_images,
            "classes": list(EXPECTED_CLASSES), "class_map": list(CLASS_MAP),
            "input": "RGB float32 /255; native 1920x1080; zero pad right/bottom to multiple32",
            "roi": {"size": [3, 3], "aligned": True, "sampling_ratio": -1, "dtype": "float16"}}


def pool_features(features, boxes):
    import torch
    from torchvision.ops import roi_align

    pooled = []
    for feature, stride in zip(features, STRIDES):
        verify_feature_shape(feature.shape, stride)
        pooled.append(roi_align(feature.float(), [boxes.float()], (3, 3),
                                spatial_scale=1.0 / stride, sampling_ratio=-1, aligned=True))
    if len(pooled) != 2 or len(features) != 2:
        raise ValueError("Exactly two P3/P4 tensors required")
    return torch.cat(pooled, dim=1)


def build_head(channels):
    from torch import nn

    return nn.Sequential(nn.Conv2d(channels, 128, 1), nn.GELU(), nn.Flatten(),
                         nn.Dropout(0.2), nn.Linear(128 * 3 * 3, len(EXPECTED_CLASSES)))


def _save_tensor(path, state):
    import torch

    _atomic_write(path, lambda handle: torch.save(state, handle), binary=True)


def _blob(root, index, entry, digest):
    import torch

    value = torch.load(root / f"{index:06d}.pt", map_location="cpu", weights_only=True)
    features = value["features"]
    if value["manifest"] != digest or value["entry"] != entry:
        raise ValueError(f"Cache entry {index} manifest mismatch")
    shapes = value["feature_shapes"]
    if len(shapes) != 2:
        raise ValueError("Missing P3/P4 shapes")
    for shape, stride in zip(shapes, STRIDES):
        verify_feature_shape(shape, stride)
    if (features.dtype != torch.float16 or tuple(features.shape) !=
            (len(entry["targets"]), sum(s[1] for s in shapes), 3, 3)
            or not torch.isfinite(features).all().item()):
        raise ValueError(f"Invalid cached ROI tensor: {index}")
    return value


def cache(args, control):
    import torch
    import torchvision
    import ultralytics
    from PIL import Image
    import numpy as np

    out = Path(args.out)
    contract = _manifest(args)
    contract["versions"] = [str(torch.__version__), str(torchvision.__version__), ultralytics.__version__]
    contract["roi_source_sha256"] = _fingerprint(Path(__file__), True)["sha256"]
    digest = _digest(contract)
    meta_path = out / "metadata.json"
    if meta_path.exists() and json.loads(meta_path.read_text())["contract"] != contract:
        raise ValueError("Cache configuration/source changed; use a new --out")
    if not meta_path.exists() and any(out.iterdir()):
        raise FileExistsError("Nonempty cache without metadata; use a new --out")
    metadata = {"contract": contract, "digest": digest, "status": "building", "complete": False}
    _write_json(meta_path, metadata)
    detector = ultralytics.YOLO(args.detector).model.eval().float().cuda()
    detector.requires_grad_(False)
    layers = detector.model
    captured, handles = {}, []
    for index in args.layers:
        if index >= len(layers):
            raise ValueError(f"Layer {index} absent; YOLO.model has {len(layers)} layers")
        def hook(_module, _inputs, output, index=index):
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                raise ValueError(f"Layer {index} ({type(layers[index]).__name__}) is not a 4D tensor")
            captured[index] = output.detach().clone()  # Protect against later in-place operations.
        handles.append(layers[index].register_forward_hook(hook))
    channels, shapes, done, skipped = None, None, 0, 0
    try:
        for index, entry in enumerate(contract["entries"]):
            control.check(budget=False)
            path = out / f"{index:06d}.pt"
            if path.exists():
                value = _blob(out, index, entry, digest)  # Invalid/corrupt blobs abort, never silently skip.
                skipped += 1
            else:
                with Image.open(entry["image"]["path"]) as image:
                    pw, ph = padded_size(*image.size)
                    array = np.array(image.convert("RGB"), copy=True)
                tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).cuda().float().div_(255)
                tensor = torch.nn.functional.pad(tensor, (0, pw - 1920, 0, ph - 1080), value=0)
                boxes = torch.tensor(entry["boxes"], device="cuda", dtype=torch.float32).reshape(-1, 4)
                captured.clear()
                with torch.inference_mode():
                    detector(tensor)  # Full forward, NOT a backbone-only latency estimate.
                    if set(captured) != set(args.layers):
                        raise ValueError(f"Missing hooks: requested {args.layers}, captured {list(captured)}")
                    features = [captured[i] for i in args.layers]
                    for layer, feature, stride in zip(args.layers, features, STRIDES):
                        try:
                            verify_feature_shape(feature.shape, stride)
                        except ValueError as exc:
                            raise ValueError(f"YOLO.model[{layer}] ({type(layers[layer]).__name__}): {exc}") from exc
                    value = {"manifest": digest, "entry": entry,
                             "feature_shapes": [list(f.shape) for f in features],
                             "features": pool_features(features, boxes).cpu().half()}
                if not torch.isfinite(value["features"]).all().item():
                    raise FloatingPointError("Nonfinite cached features")
                control.check(budget=False)
                _save_tensor(path, value)
            if shapes is not None and shapes != value["feature_shapes"]:
                raise ValueError(f"P3/P4 channels/shapes changed at image {index}")
            shapes = value["feature_shapes"]
            channels = value["features"].shape[1]
            done += 1
        _sources_unchanged(contract)
        control.check(budget=False)
        metadata.update(status="complete", complete=True)
    except SignalInterrupted:
        metadata.update(status="interrupted")
    except Exception as exc:
        metadata.update(status="failed", error=str(exc))
        raise
    finally:
        for handle in handles:
            handle.remove()
        if control.signal_number is not None:
            metadata.update(status="interrupted", complete=False)
        metadata.update(channels=channels, feature_shapes=shapes, images_checked=done,
                        images_skipped=skipped, elapsed_seconds=control.elapsed(),
                        signal=control.signal_number,
                        detector_forward="full model including detector head overhead")
        _write_json(meta_path, metadata)
    return 75 if control.signal_number is not None else 0


class ROIDataset:
    """Object index and bounded 2-GiB CPU feature LRU, zero workers.

    Random object sampling otherwise repeatedly reloads an entire image blob.
    A byte budget avoids that IO bottleneck without unbounded dataset residency.
    """

    def __init__(self, root, metadata, split):
        self.root, self.metadata, self.lru = Path(root), metadata, OrderedDict()
        self.cache_bytes = 0
        self.max_cache_bytes = 2 * 1024 ** 3
        self.indices = [(i, j) for i, e in enumerate(metadata["contract"]["entries"])
                        if e["split"] == split for j in range(len(e["targets"]))]
        if not self.indices:
            raise ValueError(f"No labeled ROIs in {split}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i, j = self.indices[index]
        entry = self.metadata["contract"]["entries"][i]
        if i not in self.lru:
            value = _blob(self.root, i, entry, self.metadata["digest"])
            if value["feature_shapes"] != self.metadata["feature_shapes"]:
                raise ValueError("Cached channels/shapes differ from metadata")
            self.lru[i] = value["features"]
            self.cache_bytes += value["features"].numel() * value["features"].element_size()
            while self.cache_bytes > self.max_cache_bytes and len(self.lru) > 1:
                _, evicted = self.lru.popitem(last=False)
                self.cache_bytes -= evicted.numel() * evicted.element_size()
        self.lru.move_to_end(i)
        return self.lru[i][j].float(), entry["targets"][j], f"{entry['image']['path']}#roi={j}"


def _epoch(model, loader, optimizer, control):
    import torch

    model.train(optimizer is not None)
    total, count, rows, targets, predictions = 0., 0, [], [], []
    with torch.set_grad_enabled(optimizer is not None):
        for features, labels, paths in loader:
            control.check(budget=False)  # Signal checks per batch, budget BETWEEN epochs.
            features, labels = features.cuda(), labels.cuda()
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=0.1)
            if not torch.isfinite(loss).item() or not torch.isfinite(logits).all().item():
                raise FloatingPointError("Nonfinite head loss/logits")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            else:
                confidence, predicted = logits.softmax(1).max(1)
                true, pred = labels.tolist(), predicted.tolist()
                targets.extend(true)
                predictions.extend(pred)
                rows.extend({"path": path, "true": EXPECTED_CLASSES[t], "pred": EXPECTED_CLASSES[p],
                             "confidence": c} for path, t, p, c in zip(paths, true, pred, confidence.tolist()))
            total += loss.item() * len(labels)
            count += len(labels)
            control.check(budget=False)
    return total / count, compute_metrics(targets, predictions), rows


def train(args, control):
    import torch
    import numpy as np

    root, out = Path(args.cache), Path(args.out)
    metadata = json.loads((root / "metadata.json").read_text())
    if not metadata["complete"] or metadata["digest"] != _digest(metadata["contract"]):
        raise ValueError("Cache incomplete or manifest digest invalid")
    if metadata["contract"]["schema"] != SCHEMA or metadata["contract"]["classes"] != list(EXPECTED_CLASSES):
        raise ValueError("Incompatible cache schema/classes")
    for source in (Path(metadata["contract"]["root"]), Path(metadata["contract"]["detector"]["path"])):
        if out.is_relative_to(source) or source.is_relative_to(out):
            raise ValueError("Head outputs must remain separate from original dataset/detector")
    _sources_unchanged(metadata["contract"])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    generators = {s: torch.Generator().manual_seed(args.seed + i) for i, s in enumerate(("train", "val"))}
    loaders = {s: torch.utils.data.DataLoader(ROIDataset(root, metadata, s), batch_size=args.batch,
               shuffle=s == "train", num_workers=0, generator=generators[s]) for s in generators}
    contract = {"cache_digest": metadata["digest"], "channels": metadata["channels"],
                "args": {k: v for k, v in vars(args).items() if k not in {"out", "resume", "max_hours", "epochs"}},
                "torch": str(torch.__version__), "roi_sha256": _fingerprint(Path(__file__), True)["sha256"],
                "metrics_sha256": _fingerprint(Path(__file__).with_name("train.py"), True)["sha256"],
                "head": "Conv1x1->128 GELU Flatten Dropout0.2 Linear7",
                "optimizer": "AdamW weight_decay=0.05; CE smoothing=0.1; fp32; clip_norm=1",
                "selection_metric": SELECTION_METRIC, "save_period": 1,
                "scope": "frozen GT ROI feasibility; not end-to-end detection; no mAP"}
    model = build_head(metadata["channels"]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    saved = torch.load(out / "last.pt", map_location="cpu", weights_only=False) if args.resume else None
    if saved and saved["contract"] != contract:
        raise ValueError("Resume contract changed; use a new --out")
    run_id = saved["wandb_run_id"] if saved else uuid.uuid4().hex[:12]
    state = saved["state"] if saved else None
    best = saved["best"] if saved else None
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
    run = None
    if args.wandb:
        import wandb

        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=args.wandb_run_name,
                         id=run_id, resume="must" if saved else "never", dir=str(out), config=contract)
        if run is None or getattr(run, "disabled", False):
            raise RuntimeError("W&B initialization failed")
    epoch, stale = (state["epoch"], state["patience_counter"]) if state else (0, 0)

    def snapshot(metrics=None, rows=None):
        return {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "optimizer": copy.deepcopy(optimizer.state_dict()), "epoch": epoch,
                "patience_counter": stale, "metrics": metrics, "predictions": rows or [],
                "rng": _capture_rng(generators)}

    def persist():
        # Embed best complete state: last.pt is authoritative if interrupted between renames.
        payload = {"contract": contract, "wandb_run_id": run_id, "state": state, "best": best}
        _save_tensor(out / "last.pt", payload)
        _save_tensor(out / "best.pt", {**payload, "state": best})

    status, code, error = "completed", 0, None
    try:
        if state:
            _restore_rng(state["rng"], generators)
        else:
            state = best = snapshot()  # Complete initial epoch-0 fallback, before any batch.
        persist()  # Also repairs best.pt from last.pt on resume.
        while epoch < args.epochs and stale < args.patience:
            control.check()
            loss, _, _ = _epoch(model, loaders["train"], optimizer, control)
            val_loss, metrics, rows = _epoch(model, loaders["val"], None, control)
            metrics["loss"] = val_loss
            control.check(budget=False)
            improved = best["metrics"] is None or metrics[SELECTION_METRIC] > best["metrics"][SELECTION_METRIC]
            epoch, stale = epoch + 1, 0 if improved else stale + 1
            state = snapshot(metrics, rows)
            if improved:
                best = state
            persist()
            _log_metrics(run, epoch, loss, metrics, args.lr)
            print(json.dumps({"epoch": epoch, "species_macro_f1": metrics[SELECTION_METRIC]}), flush=True)
        if stale >= args.patience:
            status = "early_stop"
    except BudgetReached:
        status = "budget"
    except SignalInterrupted:
        status, code = "interrupted", 75
    except Exception as exc:
        status, code, error = "failed", 1, f"{type(exc).__name__}: {exc}"
    try:
        completed = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        state, best = completed["state"], completed["best"]
        control.check(budget=False)
        if code == 0 and best["metrics"] is not None:
            _save_predictions(out / "val_predictions.csv", best["predictions"])
            _write_json(out / "confusion.json", {"split": "val", "epoch": best["epoch"], "metrics": best["metrics"]})
            if run is not None:
                run.log({"best/epoch": best["epoch"], "val/confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=[EXPECTED_CLASSES.index(r["true"]) for r in best["predictions"]],
                    preds=[EXPECTED_CLASSES.index(r["pred"]) for r in best["predictions"]],
                    class_names=list(EXPECTED_CLASSES))})
                for name in ("val_predictions.csv", "confusion.json"):
                    run.save(str(out / name), base_path=str(out), policy="now")
                run.summary["best/metrics"] = best["metrics"]
    except SignalInterrupted:
        status, code = "interrupted", 75
    except Exception as exc:
        status, code, error = "failed", 1, f"Export: {exc}"
    summary = {"metadata": contract, "cache_metadata": metadata, "status": status, "exit_code": code,
               "error": error, "epoch": state["epoch"] if state else 0, "best_epoch": best["epoch"] if best else None,
               "metrics": best["metrics"] if best else None, "wandb_run_id": run_id,
               "max_hours_per_invocation": args.max_hours, "elapsed_seconds": control.elapsed(),
               "budget": "between epochs; one epoch + checkpoint/export IO may overrun", "signal": control.signal_number}
    _write_json(out / "summary.json", summary)
    try:
        if run is not None:
            run.summary.update({"status": status, "complete_epoch": summary["epoch"]})
            run.save(str(out / "summary.json"), base_path=str(out), policy="now")
            run.finish(exit_code=75 if control.signal_number else code)
    except Exception as exc:
        code, status, error = 1, "failed", f"W&B finalization: {exc}"
    if control.signal_number is not None:
        code, status = 75, "interrupted"
    summary.update(status=status, exit_code=code, error=error, signal=control.signal_number)
    _write_json(out / "summary.json", summary)
    return code


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cache", help="Cache train/val only; restart automatically verifies/skips images")
    c.add_argument("--data", required=True)
    c.add_argument("--detector", required=True, help="Existing local E0 best.pt; no downloads")
    c.add_argument("--out", required=True)
    c.add_argument("--limit-images", type=_positive_int, help="First N sorted images PER split (smoke only)")
    c.add_argument("--layers", default="4,6", help="Exactly two layer indices, P3 stride8 then P4 stride16")
    t = sub.add_parser("train", help="Train frozen-GT-ROI head only")
    t.add_argument("--cache", required=True)
    t.add_argument("--out", required=True)
    for name, default in (("epochs", 30), ("patience", 5), ("batch", 128)):
        t.add_argument(f"--{name}", type=_positive_int, default=default)
    t.add_argument("--max-hours", type=_positive_float, default=1.)
    t.add_argument("--lr", type=_positive_float, default=1e-3)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--resume", action="store_true", help="Trusted last.pt; epochs may extend, budget resets per invocation")
    t.add_argument("--wandb", action="store_true")
    t.add_argument("--wandb-entity", default="pestline")
    t.add_argument("--wandb-project", default="fly-species")
    t.add_argument("--wandb-run-name")
    return parser


def _guard(args):
    if not os.environ.get("SLURM_JOB_ID") or not any(os.environ.get(k) for k in (
            "APPTAINER_CONTAINER", "APPTAINER_NAME", "SINGULARITY_CONTAINER", "SINGULARITY_NAME")):
        raise RuntimeError("Requires Slurm GPU allocation + Apptainer; no torch import on login")
    if (int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("SLURM_PROCID", "0")) != 0
            or os.environ.get("SLURM_JOB_PARTITION", "h100n2") != "h100n2"):
        raise RuntimeError("Single-process GPU job on h100n2 required")
    keys = ("data", "detector", "out") if args.command == "cache" else ("cache", "out")
    for key in keys:
        path = Path(getattr(args, key)).resolve()
        if not path.is_relative_to(RAID):
            raise ValueError(f"{key} must reside under {RAID}")
        setattr(args, key, str(path))
    out = Path(args.out)
    for key in keys[:-1]:
        source = Path(getattr(args, key))
        if out.is_relative_to(source) or source.is_relative_to(out):
            raise ValueError("Inputs and output trees must be separate")
    if args.command == "cache":
        args.layers = [int(i) for i in args.layers.split(",")]
        if len(args.layers) != 2 or min(args.layers) < 0 or len(set(args.layers)) != 2:
            raise ValueError("--layers requires two distinct nonnegative indices, P3 then P4")
        if not Path(args.detector).is_file():
            raise FileNotFoundError(args.detector)
    else:
        if not 0 <= args.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if args.resume and not (out / "last.pt").is_file():
            raise FileNotFoundError("--resume requires last.pt")
        if not args.resume and out.exists() and any(out.iterdir()):
            raise FileExistsError("Nonempty output; use --resume or a new --out")
    for key, suffix in {
        "XDG_CACHE_HOME": "xdg", "TORCH_HOME": "torch", "HF_HOME": "hf", "HF_HUB_CACHE": "hf/hub",
        "HUGGINGFACE_HUB_CACHE": "hf/hub", "PIP_CACHE_DIR": "pip", "YOLO_CONFIG_DIR": "ultralytics",
        "WANDB_CACHE_DIR": "wandb/cache", "WANDB_CONFIG_DIR": "wandb/config", "WANDB_DATA_DIR": "wandb/data",
        "MPLCONFIGDIR": "matplotlib", "CUDA_CACHE_PATH": "cuda", "TMPDIR": "tmp",
    }.items():
        target = RAID / "cache" / "architecture_roi" / suffix
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    os.environ["WANDB_DIR"] = str(out)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    out.mkdir(parents=True, exist_ok=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    _guard(args)  # Must precede ALL third-party imports.
    control = Control(getattr(args, "max_hours", 1.))
    previous = {s: signal.signal(s, control.handle_signal) for s in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required; CPU fallback forbidden")
        torch.cuda.set_device(0)
        return cache(args, control) if args.command == "cache" else train(args, control)
    except Exception as exc:
        code = 75 if control.signal_number is not None else 1
        _write_json(Path(args.out) / "summary.json", {
            "metadata": vars(args), "status": "interrupted" if code == 75 else "failed",
            "exit_code": code, "metrics": None, "error": f"{type(exc).__name__}: {exc}",
            "signal": control.signal_number, "elapsed_seconds": control.elapsed(),
            "checkpoint_policy": "Only trust the last atomic complete epoch; initialization may precede epoch0",
        })
        print(f"{type(exc).__name__}: {exc}", flush=True)
        return code
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())