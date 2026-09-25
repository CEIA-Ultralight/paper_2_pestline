"""Controlled ConvNeXt ablations; launch only in an Apptainer Slurm GPU job.

CLI: python -m architecture_lab.train (put fly-det/scripts on PYTHONPATH).
Only the pre-existing train/val ImageFolders are opened; crops must already be
pad75, with a leakage-free board/trap/site/time split. Padding and split provenance
cannot be inferred from pixels. No source images are modified or TEST accessed.

Metrics and CLI parsing use only the standard library. Torch/torchvision/numpy
are imported lazily, AFTER the allocation/container guards. Runtime requires
Python 3.12, torch 2.6, torchvision, numpy, Pillow, and optionally wandb.

Budget is per invocation, includes initialization/training, and is checked at
batch boundaries. An unfinished epoch is discarded, not marked completed. Final
best-checkpoint validation/export is outside that budget (but signal-aware).
Signals return 75 without evaluation, retaining the last COMPLETE epoch, or the
initial epoch-0 checkpoint. No mid-epoch optimizer/RNG state is checkpointed.
An in-flight CUDA operation, image read, initialization, or atomic checkpoint
cannot be interrupted safely; SIGKILL can only recover the last atomic save.

Manifest SHA256 hashes canonical sorted [relative path, target, byte size]
records, NOT image contents. Use the same immutable pad75 root for all variants.
Checkpoints use pickle: resume ONLY trusted local last.pt files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import tempfile
import time
import traceback
import uuid


EXPECTED_CLASSES = ("INS", "MAR", "MC", "MD", "MF", "MV", "NOISE")
BG_CLASSES = ("BG", "INS", "MAR", "MC", "MD", "MF", "MV", "NOISE")
SPECIES = ("MD", "MV", "MC", "MF")
SELECTION_METRIC = "species_macro_f1"


def classes_for(args):
    """Seven GT classes by default; proposal datasets opt in to an extra BG class."""
    return BG_CLASSES if getattr(args, "with_background", False) else EXPECTED_CLASSES
RAID = Path("/raid/user_marcospaulo")
SCHEMA_VERSION = 1
VARIANTS = ("baseline", "supcon", "arcface", "parts", "hires",
            "parts_supcon", "hires_supcon", "hires_parts", "dinov2")
SUPCON_VARIANTS = ("supcon", "parts_supcon", "hires_supcon")
PARTS_VARIANTS = ("parts", "parts_supcon", "hires_parts")


def has_supcon(variant):
    """Whether this supported variant uses two views and the SupCon loss."""
    return variant in SUPCON_VARIANTS


def has_parts(variant):
    """Whether this supported variant uses four parts and diversity loss."""
    return variant in PARTS_VARIANTS


def validate_classes(classes, expected=EXPECTED_CLASSES):
    """Reject reordered, missing, extra, or differently named classes."""
    if tuple(classes) != expected:
        raise ValueError(f"Expected alphabetical classes {expected}, got {classes}")


def compute_metrics(targets, predictions, classes=EXPECTED_CLASSES):
    """Metrics with zero_division=0 and fixed-class macro averages.

    Rows of the confusion matrix are true labels; columns are predictions.
    Species accuracy selects TRUE species rows, but keeps all prediction
    columns. Species F1/recall are derived from the FULL matrix, so non-species
    predicted as species contribute false positives. Absent species count as
    zero in the four-species macro average, rather than disappearing.
    With the BG variant, BG behaves like any other class in the matrix.
    """
    validate_classes(classes)
    targets, predictions = list(targets), list(predictions)
    if len(targets) != len(predictions):
        raise ValueError("Targets and predictions must have equal lengths")
    count = len(classes)
    matrix = [[0] * count for _ in range(count)]
    for true, pred in zip(targets, predictions):
        if any(
            isinstance(value, bool) or not isinstance(value, numbers.Integral)
            or not 0 <= value < count for value in (true, pred)
        ):
            raise ValueError("Labels must be integer indices in [0, 7)")
        matrix[int(true)][int(pred)] += 1
    per_class = {}
    for index, name in enumerate(classes):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        per_class[name] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": support, "predicted": predicted,
            "tp": tp, "fp": predicted - tp, "fn": support - tp,
        }
    species_support = sum(per_class[name]["support"] for name in SPECIES)
    species_correct = sum(per_class[name]["tp"] for name in SPECIES)
    total = len(targets)
    return {
        "num_samples": total,
        "overall_accuracy": sum(matrix[i][i] for i in range(count)) / total if total else 0.0,
        "species_accuracy": species_correct / species_support if species_support else 0.0,
        "species_support": species_support,
        "species_macro_recall": sum(per_class[n]["recall"] for n in SPECIES) / 4,
        "species_macro_f1": sum(per_class[n]["f1"] for n in SPECIES) / 4,
        "macro_f1": sum(item["f1"] for item in per_class.values()) / count,
        "per_class": per_class,
        "class_order": list(classes),
        "confusion": matrix,
        "confusion_axes": {"rows": "true", "columns": "predicted"},
    }


def is_improvement(metrics, best_metrics):
    """Strict greater-than, no accuracy/loss/epoch tiebreak."""
    return best_metrics is None or metrics[SELECTION_METRIC] > best_metrics[SELECTION_METRIC]


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return parsed


def _positive_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("Must be positive and finite")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Immutable pad75 ImageFolder root with train/val")
    parser.add_argument("--out", required=True, help="Dedicated output directory under /raid/user_marcospaulo")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--epochs", type=_positive_int, default=24)
    parser.add_argument("--patience", type=_positive_int, default=5)
    parser.add_argument("--batch", type=_positive_int, default=32)
    parser.add_argument("--img-size", type=_positive_int, default=384)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=_positive_float, default=8e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hours", type=_positive_float, default=1.5,
                        help="Per-invocation training budget; final evaluation/export is extra")
    parser.add_argument("--resume", action="store_true", help="Resume trusted --out/last.pt, not a weights-only restart")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--limit-train", type=_positive_int)
    parser.add_argument("--limit-val", type=_positive_int)
    parser.add_argument("--wandb", action="store_true", help="Required unless BOTH smoke limits are provided")
    parser.add_argument("--wandb-entity", default="pestline")
    parser.add_argument("--wandb-project", default="fly-species")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--with-background", action="store_true",
                        help="Proposal-crop datasets: 8 classes with extra BG (alphabetical first)")
    return parser


def _atomic_write(path, writer, binary=False):
    """Same-filesystem replace; failed writes never replace the previous file."""
    path = Path(path)
    mode = "wb" if binary else "w"
    kwargs = {} if binary else {"encoding": "utf-8", "newline": ""}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode=mode, dir=path.parent,
                                         prefix=f".{path.name}.", delete=False, **kwargs) as handle:
            temporary = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_json(path, value):
    _atomic_write(path, lambda handle: json.dump(value, handle, indent=2, allow_nan=False))


def _digest(value):
    payload = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


class SignalInterrupted(Exception):
    pass


class BudgetReached(Exception):
    pass


class Control:
    def __init__(self, max_hours):
        self.started = time.monotonic()
        self.deadline = self.started + max_hours * 3600
        self.signal_number = None

    def handle_signal(self, number, _frame):
        # No CUDA, IO, checkpointing, logging, or exceptions in signal handlers.
        self.signal_number = number

    def check(self, budget=True):
        if self.signal_number is not None:
            raise SignalInterrupted
        if budget and time.monotonic() >= self.deadline:
            raise BudgetReached

    def elapsed(self):
        return time.monotonic() - self.started


def _runtime_guard(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU training/evaluation requires SLURM_JOB_ID; no login-node/CPU fallback")
    if not any(os.environ.get(key) for key in (
        "APPTAINER_CONTAINER", "APPTAINER_NAME", "SINGULARITY_CONTAINER", "SINGULARITY_NAME"
    )):
        raise RuntimeError("Run inside the training Apptainer container in Slurm")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("SLURM_PROCID", "0")) != 0:
        raise RuntimeError("This trainer is single-process, single-GPU; do not launch multiple ranks")
    if args.workers < 0 or not 0 <= args.seed < 2**32:
        raise ValueError("workers must be >= 0 and seed must be in [0, 2**32)")
    smoke = args.limit_train is not None and args.limit_val is not None
    if not smoke and not args.wandb:
        raise ValueError("Real jobs require --wandb; no-W&B smoke requires both --limit-train and --limit-val")
    if args.wandb and not smoke and os.environ.get("WANDB_MODE", "online") != "online":
        raise ValueError("Real jobs require online W&B tracking")
    data, out = Path(args.data).resolve(), Path(args.out).resolve()
    if not all(path.is_relative_to(RAID) for path in (data, out)):
        raise ValueError("Data and outputs must remain under /raid/user_marcospaulo")
    if out == data or out.is_relative_to(data) or data.is_relative_to(out):
        raise ValueError("Output and dataset trees must be separate")
    args.data, args.out = str(data), str(out)
    if args.resume:
        if not (out / "last.pt").is_file():
            raise FileNotFoundError("--resume requires trusted --out/last.pt")
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty output directory; use --resume")
    cache = RAID / "cache" / "architecture_lab"
    for key, suffix in {
        "XDG_CACHE_HOME": "xdg", "TORCH_HOME": "torch", "HF_HOME": "huggingface",
        "HUGGINGFACE_HUB_CACHE": "huggingface/hub", "PIP_CACHE_DIR": "pip",
        "WANDB_CACHE_DIR": "wandb/cache", "WANDB_CONFIG_DIR": "wandb/config",
        "WANDB_DATA_DIR": "wandb/data", "TMPDIR": "tmp",
        "MPLCONFIGDIR": "matplotlib", "CUDA_CACHE_PATH": "cuda",
        "YOLO_CONFIG_DIR": "ultralytics",
    }.items():
        target = cache / suffix
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    os.environ["WANDB_DIR"] = str(out)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    out.mkdir(parents=True, exist_ok=True)
    return data, out, smoke


def _check_split_tree(root, split):
    """Do not follow symlinks that might accidentally lead into another split."""
    folder = root / split
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError(f"{split} must be a real directory, not a symlink")
    for base, directories, files in os.walk(folder, followlinks=False):
        for name in directories + files:
            if (Path(base) / name).is_symlink():
                raise ValueError(f"Symlinks are disallowed inside {split}")


def _manifest(dataset, root, split, limit, seed, classes=EXPECTED_CLASSES):
    indices = list(range(len(dataset)))
    if limit is not None and limit < len(indices):
        # Local RNG: neither variant nor global training RNG affects the subset.
        indices = sorted(random.Random(seed).sample(indices, limit))
    entries = sorted([
        [Path(dataset.samples[i][0]).relative_to(root).as_posix(),
         int(dataset.samples[i][1]), Path(dataset.samples[i][0]).stat().st_size]
        for i in indices
    ])
    return indices, {
        "schema_version": SCHEMA_VERSION, "split": split,
        "hash_definition": "SHA256 of canonical JSON sorted [relative_path,target,byte_size] records; NOT content hash",
        "sha256": _digest(entries), "class_order": list(classes),
        "class_to_idx": dataset.class_to_idx,
        "available_count": len(dataset), "selected_count": len(indices), "entries": entries,
    }


class AugmentedFolder:
    """Picklable, map-style dataset; only SupCon requests two independent views."""
    def __init__(self, folder, indices, transform, root, two_views=False):
        self.folder = folder
        self.indices = indices
        self.transform = transform
        self.root = root
        self.two_views = two_views

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample_index = self.indices[index]
        image, target = self.folder[sample_index]
        views = self.transform(image)
        if self.two_views:
            views = (views, self.transform(image))
        path = Path(self.folder.samples[sample_index][0]).relative_to(self.root).as_posix()
        return views, target, path


def _seed_worker(_worker_id):
    import numpy as np
    import torch

    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def _capture_rng(generators):
    import numpy as np
    import torch

    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
        "loaders": {key: value.get_state() for key, value in generators.items()},
    }


def _restore_rng(state, generators):
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if len(state["cuda"]) != torch.cuda.device_count():
        raise ValueError("Resume requires the same number of visible CUDA devices for RNG restoration")
    torch.cuda.set_rng_state_all(state["cuda"])
    for key, generator in generators.items():
        generator.set_state(state["loaders"][key])


def _source_metadata():
    folder = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(folder), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(folder), "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "commit": os.environ.get("FLYDET_SOURCE_COMMIT", commit), "working_tree_dirty": dirty,
        "file_sha256": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                        for name in ("train.py", "models.py")},
    }


def _preprocessing(args):
    return {
        "input": "pre-extracted pad75 RGB crops; caller guarantees padding and leakage-free split",
        "dataset_files_modified": False,
        "resize": [args.img_size, args.img_size], "interpolation": "bilinear", "antialias": True,
        "train": {"hflip_probability": 0.5, "vflip_probability": 0.5,
                  "rotation_degrees": 30, "rotation_fill": 0,
                  "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05}},
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "val": "resize + tensor + ImageNet normalization; no augmentation",
        "supcon_views": (
            "two independent calls of the same train transform, concatenated view-wise"
            if has_supcon(args.variant) else "not used; single train view"
        ),
        "sampler": "unweighted shuffled permutation, no replacement, no class balancing, drop_last=False",
    }


def _make_loaders(args, data, preprocessing):
    import torch
    from torchvision import datasets, transforms
    from torchvision.transforms import InterpolationMode

    normalize = preprocessing["normalization"]
    resize = transforms.Resize((args.img_size, args.img_size),
                               interpolation=InterpolationMode.BILINEAR, antialias=True)
    tail = [transforms.ToTensor(), transforms.Normalize(normalize["mean"], normalize["std"])]
    train_transform = transforms.Compose([
        resize, transforms.RandomHorizontalFlip(0.5), transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(30, interpolation=InterpolationMode.BILINEAR, fill=0),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05), *tail,
    ])
    val_transform = transforms.Compose([resize, *tail])
    loaders, generators, manifests = {}, {}, {}
    for split, limit, transform in (
        ("train", args.limit_train, train_transform), ("val", args.limit_val, val_transform)
    ):
        _check_split_tree(data, split)
        classes = classes_for(args)
        folder = datasets.ImageFolder(str(data / split))
        validate_classes(folder.classes, classes)
        if folder.class_to_idx != {name: i for i, name in enumerate(classes)}:
            raise ValueError(f"{split} class mapping differs from the expected mapping")
        indices, manifests[split] = _manifest(folder, data, split, limit, args.seed, classes)
        wrapped = AugmentedFolder(folder, indices, transform, data,
                                  two_views=split == "train" and has_supcon(args.variant))
        generator = torch.Generator().manual_seed(args.seed + (split == "val"))
        generators[split] = generator
        extra = {"multiprocessing_context": "spawn"} if args.workers else {}
        loaders[split] = torch.utils.data.DataLoader(
            wrapped, batch_size=args.batch, shuffle=split == "train", num_workers=args.workers,
            pin_memory=True, drop_last=False, generator=generator, worker_init_fn=_seed_worker,
            persistent_workers=False, **extra,
        )
    return loaders, generators, manifests


def _train_epoch(model, loader, optimizer, scaler, args, control):
    import torch
    from torch.nn import functional as F
    from .models import arcface_logits, attention_diversity, supcon_loss

    model.train()
    loss_sum, examples = 0.0, 0
    control.check()
    for images, labels, _paths in loader:
        control.check()
        labels = labels.to("cuda", non_blocking=True)
        if has_supcon(args.variant):
            images = torch.cat(images, dim=0)
            labels = torch.cat((labels, labels), dim=0)
        images = images.to("cuda", non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(images)
            logits = outputs["logits"]
            if args.variant == "arcface":
                logits = arcface_logits(outputs["embedding"], model.metric_weight, labels,
                                        scale=model.scale, margin=0.3)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            if has_supcon(args.variant):
                loss = loss + 0.1 * supcon_loss(outputs["embedding"], labels, temperature=0.1)
            if has_parts(args.variant):
                loss = loss + 0.05 * attention_diversity(outputs["attention"])
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Nonfinite training loss; retaining last complete checkpoint")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Fail rather than silently checkpoint a run with skipped/nonfinite updates.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if not torch.isfinite(norm).item():
            raise FloatingPointError("Nonfinite gradient norm")
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.detach().item() * labels.numel()
        examples += labels.numel()
        control.check()
    return loss_sum / examples


def _evaluate(model, loader, control, *, budget, collect_rows=False, classes=EXPECTED_CLASSES):
    import torch
    from torch.nn import functional as F

    model.eval()
    targets, predictions, rows = [], [], []
    loss_sum = 0.0
    control.check(budget=budget)
    with torch.inference_mode():
        for images, labels, paths in loader:
            control.check(budget=budget)
            images, gpu_labels = images.to("cuda", non_blocking=True), labels.to("cuda", non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # No true-label input: ArcFace inference is scaled cosine, never target-margin logits.
                logits = model(images)["logits"]
                loss = F.cross_entropy(logits, gpu_labels, label_smoothing=0.1)
            if not torch.isfinite(logits).all().item() or not torch.isfinite(loss).item():
                raise FloatingPointError("Nonfinite validation output")
            confidence, predicted = logits.float().softmax(dim=1).max(dim=1)
            true_values, pred_values = labels.tolist(), predicted.cpu().tolist()
            targets.extend(true_values)
            predictions.extend(pred_values)
            loss_sum += loss.item() * len(true_values)
            if collect_rows:
                rows.extend({"path": path, "true": classes[true],
                             "pred": classes[pred], "confidence": probability}
                            for path, true, pred, probability in
                            zip(paths, true_values, pred_values, confidence.cpu().tolist()))
            control.check(budget=budget)
    metrics = compute_metrics(targets, predictions, classes)
    metrics["loss"] = loss_sum / len(targets)
    return metrics, rows


def _log_metrics(run, epoch, train_loss, metrics, lr):
    if run is None:
        return
    values = {"epoch": epoch, "train/loss": train_loss, "train/lr": lr}
    values.update({f"val/{key}": value for key, value in metrics.items()
                   if isinstance(value, (int, float))})
    for name, measures in metrics["per_class"].items():
        values.update({f"val/{name}/{key}": value for key, value in measures.items()})
    run.log(values)


def _save_predictions(path, rows):
    def write(handle):
        writer = csv.DictWriter(handle, fieldnames=("path", "true", "pred", "confidence"))
        writer.writeheader()
        writer.writerows(rows)
    _atomic_write(path, write)


def _run(args, data, out, smoke, control):
    import numpy as np
    import torch
    import torchvision
    from .models import build_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required inside Slurm; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    classes = classes_for(args)

    preprocessing = _preprocessing(args)
    loaders, generators, manifests = _make_loaders(args, data, preprocessing)
    source = _source_metadata()
    training_spec = {
        "optimizer": "AdamW", "weight_decay": 0.05, "betas": [0.9, 0.999],
        "scheduler": "CosineAnnealingLR per completed epoch", "eta_min": args.lr * 0.01,
        "label_smoothing": 0.1, "supcon_weight": 0.1, "supcon_temperature": 0.1,
        "parts_diversity_weight": 0.05, "arcface_scale": 30.0, "arcface_margin": 0.3,
        "gradient_clip_norm": 1.0, "amp": "bfloat16; GradScaler disabled on H100",
        "deterministic_algorithms": True, "save_period": 1,
        "selection_metric": SELECTION_METRIC, "selection_rule": "strict >; no tiebreak",
        "budget": "per invocation; discard incomplete epoch; final evaluation/export excluded",
    }
    contract = {
        "args": {key: value for key, value in vars(args).items()
                 if key not in {"out", "resume", "max_hours"}},
        "preprocessing": preprocessing, "training": training_spec,
        "manifest_sha256": {split: item["sha256"] for split, item in manifests.items()},
        "class_order": list(classes), "source_sha256": source["file_sha256"],
        "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
    }
    last_path = out / "last.pt"
    checkpoint = None
    previous_elapsed = 0.0
    experiment_id = str(uuid.uuid4())
    if args.resume:
        # Explicit opt-out of torch 2.6 weights-only default: TRUSTED LOCAL FILE ONLY.
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint["schema_version"] != SCHEMA_VERSION or checkpoint["resume_contract"] != contract:
            raise ValueError("Resume contract changed (args/data/classes/source/versions); use a new --out")
        experiment_id = checkpoint["experiment_id"]
        previous_elapsed = checkpoint["elapsed_seconds"]
        summary_path = out / "summary.json"
        if summary_path.exists():
            previous_summary = json.loads(summary_path.read_text())
            if previous_summary.get("experiment_id") == experiment_id:
                previous_elapsed = max(previous_elapsed, previous_summary.get("elapsed_seconds", 0.0))
        for split, manifest in manifests.items():
            existing = json.loads((out / f"{split}_manifest.json").read_text())
            if existing != manifest:
                raise ValueError(f"Persisted {split} manifest differs; dataset must stay immutable")
    else:
        for split, manifest in manifests.items():
            _write_json(out / f"{split}_manifest.json", manifest)

    model = build_model(args.variant, len(classes),
                        pretrained=not args.no_pretrained and not args.resume).cuda()
    # On resume initialization comes from the checkpoint, not another download.
    model.architecture_metadata["pretrained"] = not args.no_pretrained
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    epoch, patience_counter = 0, 0
    best = {"epoch": None, "metrics": None, "model": None}
    history = []
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["grad_scaler"])
        epoch, patience_counter = checkpoint["epoch"], checkpoint["patience_counter"]
        best, history = checkpoint["best"], checkpoint["history"]
    initial_epoch = epoch
    metadata = {
        "args": vars(args).copy(), "initial_args": checkpoint["metadata"]["initial_args"] if checkpoint else vars(args).copy(),
        "preprocessing": preprocessing, "training": training_spec, "source": source,
        "architecture": model.architecture_metadata, "class_order": list(classes),
        "class_to_idx": {name: i for i, name in enumerate(classes)},
        "dataset_root": str(data), "splits_accessed": ["train", "val"], "smoke": smoke,
        "manifests": {split: {key: value for key, value in item.items() if key != "entries"}
                      for split, item in manifests.items()},
        "environment": {"python": platform.python_version(),
                        "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                        "numpy": np.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0), "slurm_job_id": os.environ["SLURM_JOB_ID"],
                        "container": os.environ.get("APPTAINER_CONTAINER", os.environ.get("SINGULARITY_CONTAINER"))},
        "confidence": "uncalibrated softmax of inference logits (scaled cosine for ArcFace)",
        "qualitative_gradcam": "not implemented; no causal interpretation",
    }
    run = None
    wandb_id = checkpoint["wandb_run_id"] if checkpoint else None
    if args.wandb:
        import wandb

        # Public run IDs accept alphanumeric strings; avoid removed SDK util APIs.
        wandb_id = wandb_id or uuid.uuid4().hex[:8]
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                         name=args.wandb_run_name, id=wandb_id,
                         resume="must" if args.resume else "never", dir=str(out))
        if run is None or getattr(run, "disabled", False):
            raise RuntimeError("W&B tracking could not be initialized")
        run.config.update(metadata, allow_val_change=True)
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")

    def save_checkpoint():
        state = {
            "schema_version": SCHEMA_VERSION, "experiment_id": experiment_id,
            "resume_contract": contract, "metadata": metadata,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "grad_scaler": scaler.state_dict(),
            "epoch": epoch, "best": best, "patience_counter": patience_counter,
            "rng": _capture_rng(generators), "wandb_run_id": wandb_id,
            "elapsed_seconds": previous_elapsed + control.elapsed(), "history": history,
        }
        # best.model is embedded: last.pt is a SINGLE atomic, self-contained
        # transaction, even if a crash occurs between last.pt and best.pt exports.
        _atomic_write(last_path, lambda handle: torch.save(state, handle), binary=True)

    status, exit_code, error = "completed", 0, None
    exported_metrics, exported_epoch = None, None
    partial_epoch_discarded = False
    try:
        if checkpoint is not None:
            _restore_rng(checkpoint["rng"], generators)
            del checkpoint
        else:
            save_checkpoint()  # Safe epoch 0 fallback, including W&B ID and initial RNG.
        if run is not None:
            for split in ("train", "val"):
                run.save(str(out / f"{split}_manifest.json"), base_path=str(out), policy="now")
        while epoch < args.epochs:
            control.check()
            if patience_counter >= args.patience:
                status = "early_stop"
                break
            try:
                lr = optimizer.param_groups[0]["lr"]
                train_loss = _train_epoch(model, loaders["train"], optimizer, scaler, args, control)
                metrics, _ = _evaluate(model, loaders["val"], control, budget=True, classes=classes)
            except Exception:
                partial_epoch_discarded = True
                raise
            epoch += 1
            if is_improvement(metrics, best["metrics"]):
                best = {"epoch": epoch, "metrics": metrics,
                        "model": {name: value.detach().cpu().clone()
                                  for name, value in model.state_dict().items()}}
                patience_counter = 0
            else:
                patience_counter += 1
            scheduler.step()
            history.append({"epoch": epoch, "train_loss": train_loss, "lr": lr,
                            "val": metrics, "patience_counter": patience_counter})
            save_checkpoint()  # Only after a FULL train + val + scheduler step.
            print(json.dumps({"epoch": epoch, "val_species_macro_f1": metrics[SELECTION_METRIC],
                              "best_epoch": best["epoch"], "patience": patience_counter}), flush=True)
            _log_metrics(run, epoch, train_loss, metrics, lr)
            control.check(budget=False)
            if patience_counter >= args.patience:
                status = "early_stop"
                break
    except BudgetReached:
        status = "budget"
    except SignalInterrupted:
        status, exit_code = "interrupted", 75
    except Exception as exc:
        status, exit_code, error = "failed", 1, f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    # Read the authoritative full-epoch transaction, not possibly mutated live
    # optimizer/model/history objects from a discarded or failed epoch.
    try:
        if last_path.exists():
            completed = torch.load(last_path, map_location="cpu", weights_only=False)
            epoch, best, history = completed["epoch"], completed["best"], completed["history"]
            patience_counter = completed["patience_counter"]
        else:
            completed = None
        if exit_code == 0:
            control.check(budget=False)
            if completed is None:
                raise RuntimeError("No safe checkpoint exists for final evaluation")
            weights = best["model"] if best["model"] is not None else completed["model"]
            model.load_state_dict(weights, strict=True)
            exported_epoch = best["epoch"] if best["epoch"] is not None else 0
            exported_metrics, rows = _evaluate(model, loaders["val"], control, budget=False,
                                               collect_rows=True, classes=classes)
            control.check(budget=False)
            # If the budget ended before any complete epoch, export the INITIAL
            # checkpoint honestly: best_metrics remains null; exported_epoch=0.
            _atomic_write(out / "best.pt", lambda handle: torch.save({
                "model": weights, "epoch": exported_epoch, "metrics": exported_metrics,
                "metadata": metadata, "selection_metric": SELECTION_METRIC,
                "is_initial_fallback": best["model"] is None,
            }, handle), binary=True)
            _save_predictions(out / "val_predictions.csv", rows)
            _write_json(out / "confusion.json", {
                "split": "val", "epoch": exported_epoch, "class_order": list(classes),
                "axes": {"rows": "true", "columns": "predicted"},
                "matrix": exported_metrics["confusion"], "metrics": exported_metrics,
            })
            control.check(budget=False)
            if run is not None:
                for name in ("val_predictions.csv", "confusion.json"):
                    run.save(str(out / name), base_path=str(out), policy="now")
                run.summary["best/epoch"] = best["epoch"]
                run.summary["best/metrics"] = best["metrics"]
                run.summary["export/metrics"] = exported_metrics
    except SignalInterrupted:
        status, exit_code = "interrupted", 75
    except Exception as exc:
        status, exit_code, error = "failed", 1, f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    if control.signal_number is not None:
        status, exit_code = "interrupted", 75
    summary = {
        "schema_version": SCHEMA_VERSION, "experiment_id": experiment_id, "metadata": metadata,
        "status": status, "exit_code": exit_code, "split": "val", "error": error,
        "epochs_ran": epoch, "epochs_ran_this_invocation": epoch - initial_epoch,
        "elapsed_seconds": previous_elapsed + control.elapsed(),
        "invocation_elapsed_seconds": control.elapsed(), "max_hours_per_invocation": args.max_hours,
        "selection_metric": SELECTION_METRIC, "selection_rule": "strict >",
        "best_epoch": best["epoch"], "best_metrics": best["metrics"],
        "exported_epoch": exported_epoch, "exported_metrics": exported_metrics,
        "export_complete": exit_code == 0 and exported_metrics is not None,
        "initial_checkpoint_fallback": exported_epoch == 0,
        "patience_counter": patience_counter, "partial_epoch_discarded": partial_epoch_discarded,
        "last_completed_checkpoint": "last.pt", "save_period": 1,
        "signal": control.signal_number, "wandb_run_id": wandb_id,
        "history": history,
    }
    _write_json(out / "summary.json", summary)
    if run is not None:
        try:
            run.summary["status"] = status
            run.summary["selection_metric"] = SELECTION_METRIC
            run.summary["epochs_ran"] = epoch
            run.summary["elapsed_seconds"] = summary["elapsed_seconds"]
            run.save(str(out / "summary.json"), base_path=str(out), policy="now")
            if control.signal_number is not None:
                exit_code = 75
                run.summary["status"] = "interrupted"
            run.finish(exit_code=exit_code)
        except Exception as exc:
            traceback.print_exc()
            exit_code = 75 if control.signal_number is not None else 1
            summary.update(status="interrupted" if exit_code == 75 else "failed",
                           exit_code=exit_code, error=f"W&B finalization: {type(exc).__name__}: {exc}",
                           export_complete=False, signal=control.signal_number)
            _write_json(out / "summary.json", summary)
    # A signal arriving during uploads/finish still must NOT be reported as success.
    if control.signal_number is not None:
        summary.update(status="interrupted", exit_code=75, signal=control.signal_number, export_complete=False)
        _write_json(out / "summary.json", summary)
        exit_code = 75
    return exit_code


def main(argv=None):
    args = build_parser().parse_args(argv)
    control = Control(args.max_hours)
    previous_handlers = {}
    for number in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        previous_handlers[number] = signal.signal(number, control.handle_signal)
    try:
        data, out, smoke = _runtime_guard(args)
        return _run(args, data, out, smoke, control)
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())