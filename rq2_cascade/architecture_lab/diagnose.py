"""Validation-only Grad-CAM and classifier-only CUDA microbenchmark.

CLI: python -m architecture_lab.diagnose --run TRAIN_OUTPUT --samples 8 --batch 32
Requires an Apptainer Slurm GPU allocation and an existing ONLINE W&B run.
Only trusted local best.pt exports from architecture_lab.train are supported.
No training, test data, source-image writes, API E2E timing or causal claims.
Imports used by --help and the runtime guard are standard-library only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from .train import (
    EXPECTED_CLASSES, RAID, SCHEMA_VERSION, _check_split_tree, _digest,
    _manifest, _positive_int, _write_json, compute_metrics, validate_classes,
)


CAM_WARNING = "Grad-CAM is qualitative, not causal; not segmentation or localization ground truth."
PRIORITY_PAIRS = (("MD", "MV"), ("MC", "MV"), ("MF", "NOISE"))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Existing train.py output under /raid/user_marcospaulo")
    parser.add_argument("--samples", type=_positive_int, default=8,
                        help="Deterministic validation examples (default: 8; capped by available rows)")
    parser.add_argument("--batch", type=_positive_int, default=32,
                        help="Second synthetic benchmark batch, alongside batch 1 (default: 32)")
    return parser


def _runtime_guard(args):
    # Must execute before importing torch, torchvision, models, numpy or wandb.
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required; no login-node execution")
    if not any(os.environ.get(key) for key in (
        "APPTAINER_CONTAINER", "APPTAINER_NAME", "SINGULARITY_CONTAINER", "SINGULARITY_NAME"
    )):
        raise RuntimeError("Apptainer container required inside Slurm")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("SLURM_PROCID", "0")) != 0:
        raise RuntimeError("Single-process, single-GPU diagnostic required")
    if os.environ.get("WANDB_MODE", "online") != "online":
        raise RuntimeError("Online W&B is required for resume=must")
    run_dir = Path(args.run).resolve()
    if not run_dir.is_relative_to(RAID) or not run_dir.is_dir():
        raise ValueError("--run must be an existing RAID output directory")
    for name in ("best.pt", "summary.json", "val_manifest.json", "val_predictions.csv"):
        path = run_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing or symlinked training export")
    summary = json.loads((run_dir / "summary.json").read_text())
    if summary.get("schema_version") != SCHEMA_VERSION or not summary.get("export_complete"):
        raise ValueError("A completed train.py validation export is required")
    metadata = summary["metadata"]
    data = Path(metadata["dataset_root"]).resolve()
    if not data.is_relative_to(RAID) or not data.is_dir():
        raise ValueError("Dataset must be an existing RAID directory")
    if data == run_dir or run_dir.is_relative_to(data) or data.is_relative_to(run_dir):
        raise ValueError("Output and dataset trees must be separate")
    identity = {
        "id": summary.get("wandb_run_id"),
        "entity": metadata["args"].get("wandb_entity"),
        "project": metadata["args"].get("wandb_project"),
    }
    if not metadata["args"].get("wandb") or not all(
        isinstance(value, str) and value and all(c.isalnum() or c in "_-" for c in value)
        for value in identity.values()
    ):
        raise ValueError("Training export must identify an existing W&B run")
    if identity["entity"] != "pestline":
        raise ValueError("Expected W&B entity pestline")
    output = run_dir / "diagnostics"
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise FileExistsError("Refusing to overwrite an existing diagnostic; archive it first")
    cache = RAID / "cache" / "architecture_lab"
    for key, suffix in {
        "XDG_CACHE_HOME": "xdg", "TORCH_HOME": "torch", "HF_HOME": "huggingface",
        "HUGGINGFACE_HUB_CACHE": "huggingface/hub", "PIP_CACHE_DIR": "pip",
        "WANDB_CACHE_DIR": "wandb/cache", "WANDB_CONFIG_DIR": "wandb/config",
        "WANDB_DATA_DIR": "wandb/data", "TMPDIR": "tmp",
    }.items():
        target = (cache / suffix).resolve()
        if not target.is_relative_to(RAID):
            raise ValueError("Cache must remain on RAID")
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    os.environ["WANDB_DIR"] = str(run_dir)
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["WANDB_DISABLE_CODE"] = "true"
    os.environ["WANDB_DISABLE_GIT"] = "true"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # tempfile may have cached a directory before this invocation.
    import tempfile
    tempfile.tempdir = os.environ["TMPDIR"]
    return run_dir, data, output, summary, identity


def _same_metadata(checkpoint_metadata, json_metadata):
    """Compare JSON-safe metadata independent of tuple/list serialization."""
    return json.loads(json.dumps(checkpoint_metadata)) == json_metadata


def _validation_inputs(run_dir, data, checkpoint, summary):
    from torchvision import datasets, transforms
    from torchvision.transforms import InterpolationMode

    metadata = checkpoint["metadata"]
    # Torch preserves tuple strides, while JSON summaries turn tuples into lists.
    # Compare their JSON representations, not serialization-specific container types.
    if not _same_metadata(metadata, summary["metadata"]) or checkpoint["epoch"] != summary["exported_epoch"]:
        raise ValueError("Checkpoint and summary are from different exports")
    if checkpoint["metrics"] != summary["exported_metrics"]:
        raise ValueError("Checkpoint and summary metrics differ")
    validate_classes(metadata["class_order"])
    mapping = {name: i for i, name in enumerate(EXPECTED_CLASSES)}
    if metadata["class_to_idx"] != mapping:
        raise ValueError("Checkpoint class mapping differs")
    # Strides are not serialized in state_dict: refuse incompatible model code.
    expected_source = metadata["source"]["file_sha256"]["models.py"]
    if hashlib.sha256(Path(__file__).with_name("models.py").read_bytes()).hexdigest() != expected_source:
        raise ValueError("Model source differs from checkpoint provenance")
    prep = metadata["preprocessing"]
    size = metadata["args"]["img_size"]
    if (isinstance(size, bool) or not isinstance(size, int) or size < 1
            or prep["resize"] != [size, size] or prep["interpolation"] != "bilinear"
            or prep["antialias"] is not True
            or prep["val"] != "resize + tensor + ImageNet normalization; no augmentation"):
        raise ValueError("Unsupported checkpoint validation transformation")
    normalize = prep["normalization"]
    if (len(normalize["mean"]) != 3 or len(normalize["std"]) != 3
            or not all(math.isfinite(v) for v in normalize["mean"] + normalize["std"])
            or not all(v > 0 for v in normalize["std"])):
        raise ValueError("Invalid checkpoint normalization")
    transform = transforms.Compose([
        transforms.Resize(tuple(prep["resize"]), interpolation=InterpolationMode.BILINEAR,
                          antialias=prep["antialias"]),
        transforms.ToTensor(), transforms.Normalize(normalize["mean"], normalize["std"]),
    ])
    # Never call train._make_loaders: it opens train as well as val.
    _check_split_tree(data, "val")
    folder = datasets.ImageFolder(str(data / "val"))
    validate_classes(folder.classes)
    if folder.class_to_idx != mapping:
        raise ValueError("Validation class mapping differs")
    _, actual = _manifest(folder, data, "val", metadata["args"]["limit_val"], metadata["args"]["seed"])
    persisted = json.loads((run_dir / "val_manifest.json").read_text())
    if (actual != persisted or {k: v for k, v in actual.items() if k != "entries"}
            != metadata["manifests"]["val"]):
        raise ValueError("Validation data/manifest differs from the checkpoint")
    expected = {path: EXPECTED_CLASSES[target] for path, target, _size in actual["entries"]}
    with (run_dir / "val_predictions.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["path", "true", "pred", "confidence"]:
            raise ValueError("Unsupported validation CSV format")
        rows = list(reader)
    seen = set()
    for row in rows:
        path = row["path"]
        if path in seen or path not in expected or row["true"] != expected[path]:
            raise ValueError("Validation CSV does not match the selected manifest")
        if row["pred"] not in EXPECTED_CLASSES:
            raise ValueError("Invalid CSV prediction")
        row["confidence"] = float(row["confidence"])
        if not math.isfinite(row["confidence"]) or not 0 <= row["confidence"] <= 1:
            raise ValueError("Invalid CSV confidence")
        # Additional containment check even after ImageFolder's symlink rejection.
        if not (data / path).resolve().is_relative_to(data / "val"):
            raise ValueError("Only validation crop paths are permitted")
        seen.add(path)
    if seen != set(expected) or not rows:
        raise ValueError("Validation CSV is incomplete or empty")
    csv_metrics = compute_metrics([mapping[row["true"]] for row in rows],
                                  [mapping[row["pred"]] for row in rows])
    if any(checkpoint["metrics"].get(key) != value for key, value in csv_metrics.items()):
        raise ValueError("Validation CSV metrics differ from the best checkpoint export")
    return transform, rows, actual


def _select_samples(rows, count):
    """Round-robin hard pairs, reserve correct controls, then fill other errors.

    All ties use the relative path (no RNG, confidence ranking or label leakage
    into the forward pass). Pair membership is bidirectional.
    """
    rows = sorted(rows, key=lambda row: row["path"])
    count = min(count, len(rows))
    correct = [row for row in rows if row["true"] == row["pred"]]
    # Prefer class-diverse correct controls, then further correct examples.
    first, remaining, classes = [], [], set()
    for row in correct:
        (remaining if row["true"] in classes else first).append(row)
        classes.add(row["true"])
    correct = first + remaining
    reserve = min(len(correct), max(1, count // 4)) if count > 1 else 0
    groups = [[row for row in rows if row["true"] != row["pred"]
               and {row["true"], row["pred"]} == set(pair)] for pair in PRIORITY_PAIRS]
    priority = []
    for index in range(max(map(len, groups), default=0)):
        priority.extend(group[index] for group in groups if index < len(group))
    selected = priority[:count - reserve] + correct[:reserve]
    seen = {row["path"] for row in selected}
    fallback = priority + [row for row in rows if row["true"] != row["pred"]] + correct
    for row in fallback:
        if len(selected) >= count:
            break
        if row["path"] not in seen:
            selected.append(row)
            seen.add(row["path"])
    return selected


def _gradcam(model, image, transform):
    import torch

    model.zero_grad(set_to_none=True)
    with torch.enable_grad(), torch.autocast(device_type="cuda", enabled=False):
        inputs = transform(image).unsqueeze(0).to(device="cuda", dtype=torch.float32)
        outputs = model(inputs)  # Normal, label-independent inference, including ArcFace.
        logits = outputs["logits"]
        feature_map = outputs["feature_map"]
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError("Nonfinite diagnostic logits")
        feature_map.retain_grad()
        confidence, prediction = logits.detach().float().softmax(dim=1).max(dim=1)
        target = int(prediction.item())
        logits[0, target].backward()  # Always the PREDICTED logit, never ground truth.
        gradients = feature_map.grad
        if gradients is None or not torch.isfinite(gradients).all().item():
            raise FloatingPointError("Missing or nonfinite feature-map gradients")
        weights = gradients.float().mean(dim=(-2, -1), keepdim=True)
        cam = (weights * feature_map.detach().float()).sum(dim=1).relu()[0]
        if not torch.isfinite(cam).all().item():
            raise FloatingPointError("Nonfinite CAM")
        peak = float(cam.max().item())
        all_zero = peak == 0.0
        cam = torch.zeros_like(cam) if all_zero else cam / peak
        result = (cam.cpu().numpy(), target, float(confidence.item()), all_zero)
    model.zero_grad(set_to_none=True)
    return result


def _save_visualizations(image, cam, all_zero, output, stem):
    import numpy as np
    from PIL import Image

    # Only the CAM is interpolated to the ORIGINAL crop geometry for display.
    # This projection is qualitative; image is never resized or written back.
    heat = np.asarray(Image.fromarray(cam.astype(np.float32)).resize(
        image.size, resample=Image.Resampling.BILINEAR), dtype=np.float32).clip(0, 1)
    # Simple black -> red -> yellow palette, no plotting-library dependency.
    colors = np.stack((np.minimum(2 * heat, 1), np.maximum(2 * heat - 1, 0),
                       np.zeros_like(heat)), axis=-1)
    original = np.asarray(image, dtype=np.float32)
    alpha = (0.45 * heat)[..., None]
    overlay = original if all_zero else original * (1 - alpha) + 255 * colors * alpha
    heat_path, overlay_path = output / f"{stem}_heatmap.png", output / f"{stem}_overlay.png"
    Image.fromarray(np.rint(colors * 255).astype(np.uint8)).save(heat_path)
    Image.fromarray(np.rint(overlay).clip(0, 255).astype(np.uint8)).save(overlay_path)
    return heat_path, overlay_path


def _benchmark(model, size, batch):
    import numpy as np
    import torch

    model.zero_grad(set_to_none=True)
    model.eval()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    storage_bytes = sum(t.numel() * t.element_size() for t in
                        list(model.parameters()) + list(model.buffers()))
    model_allocated = torch.cuda.memory_allocated()
    results = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        for count in dict.fromkeys((1, batch)):
            inputs = torch.randn(count, 3, size, size, device="cuda", dtype=torch.float32)
            for _ in range(10):
                outputs = model(inputs)
                del outputs
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
            times = []
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(30):
                start.record()
                outputs = model(inputs)
                end.record()
                torch.cuda.synchronize()
                times.append(float(start.elapsed_time(end)))
                del outputs
            peak = torch.cuda.max_memory_allocated()
            p50, p95 = (float(v) for v in np.percentile(times, [50, 95]))
            results.append({
                "batch_size": count, "input_shape": list(inputs.shape),
                "batch_p50_ms": p50, "batch_p95_ms": p95,
                "item_p50_ms": p50 / count, "item_p95_ms": p95 / count,
                "item_latency_definition": "batch latency / batch size; amortized, not single-request latency",
                "samples_batch_ms": times, "peak_allocated_bytes": peak,
                "baseline_allocated_bytes": baseline,
                "incremental_peak_allocated_bytes": peak - baseline,
                "input_tensor_bytes": inputs.numel() * inputs.element_size(),
            })
            del inputs
            torch.cuda.empty_cache()
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "schema_version": 1, "scope": "classifier_only; NOT API E2E",
        "input": "synthetic resident CUDA tensors representing classifier crops, NOT whole images",
        "included": "normal model.forward including logits, embedding, feature_map and optional attention",
        "excluded": ["file I/O", "image decoding", "preprocessing", "host-to-device transfer",
                     "detector", "crop extraction", "Grad-CAM", "softmax/postprocessing", "API", "network"],
        "preprocessing_included": False, "warmup": 10, "repeats": 30,
        "timer": "CUDA events with device synchronization after each forward",
        "percentiles": "numpy.percentile, linear interpolation",
        "dtype": "float32", "autocast": False, "tf32": False,
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "gpu": properties.name,
        "gpu_index": device, "gpu_total_memory_bytes": properties.total_memory,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "deterministic_algorithms": True, "cudnn_benchmark": False,
        "model_parameter_and_buffer_bytes": storage_bytes,
        "model_resident_allocated_bytes": model_allocated,
        "memory_definition": "PyTorch allocated bytes; excludes allocator reserve, CUDA context and other processes",
        "results": results,
    }


def _run(args, context):
    import numpy as np
    import torch
    import torchvision
    import wandb
    from PIL import Image
    from .models import build_model

    run_dir, data, output, summary, identity = context
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    if wandb.run is not None:
        raise RuntimeError("Refusing to attach to an already-active SDK run")
    torch.cuda.set_device(0)
    seed = summary["metadata"]["args"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # best.pt contains only weights and basic metadata; never unpickle last.pt.
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    transform, rows, manifest = _validation_inputs(run_dir, data, checkpoint, summary)
    metadata = checkpoint["metadata"]
    variant = metadata["args"]["variant"]
    model = build_model(variant, len(EXPECTED_CLASSES), pretrained=False)
    model.architecture_metadata["pretrained"] = metadata["architecture"]["pretrained"]
    if _digest(model.architecture_metadata) != _digest(metadata["architecture"]):
        raise ValueError("Reconstructed architecture differs from checkpoint")
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.cuda().float().eval()
    del checkpoint
    selected = _select_samples(rows, args.samples)
    output.mkdir(parents=False, exist_ok=True)
    started = time.monotonic()
    report = {
        "schema_version": 1, "status": "running", "split": "val", "splits_accessed": ["val"],
        "checkpoint": "best.pt", "checkpoint_epoch": summary["exported_epoch"],
        "initial_checkpoint_fallback": summary["initial_checkpoint_fallback"],
        "experiment_id": summary["experiment_id"], "wandb": identity,
        "variant": variant, "preprocessing": metadata["preprocessing"],
        "val_manifest_sha256": manifest["sha256"], "manifest_hash_definition": manifest["hash_definition"],
        "csv_sha256": hashlib.sha256((run_dir / "val_predictions.csv").read_bytes()).hexdigest(),
        "requested_samples": args.samples, "selected_samples": len(selected),
        "selection": "path-sorted round-robin MD/MV, MC/MV, MF/NOISE errors; reserve up to samples//4 correct controls; fill remaining errors/correct",
        "warning": CAM_WARNING, "cam_target": "current predicted logit, never ground truth",
        "cam_method": "ReLU(sum_channels(mean_spatial(feature_map.grad) * feature_map)); divide by positive maximum",
        "cam_dtype": "float32", "confidence": "uncalibrated softmax probability of predicted class",
        "csv_comparison": "CSV uses training-export AMP bfloat16; current diagnostics use FP32, so predictions can differ",
        "visualization": "heatmap projected onto native crop dimensions; overlay only; dataset files unmodified",
        "dataset_files_modified": False, "benchmark_scope": "classifier only, NOT API E2E",
        "torchvision": str(torchvision.__version__),
        "checkpoint_versions": {key: metadata["environment"][key]
                    for key in ("torch", "torchvision")},
        "samples": [],
    }
    run = None
    exit_code = 1
    try:
        run = wandb.init(**identity, resume="must", mode="online", dir=str(run_dir),
                         settings=wandb.Settings(console="off", disable_git=True, save_code=False))
        if (run is None or getattr(run, "disabled", False) or run.id != identity["id"]
                or run.entity != identity["entity"] or run.project != identity["project"]
                or not run.resumed):
            raise RuntimeError("W&B did not resume the exact existing run")
        table = wandb.Table(columns=["path", "true", "pred", "confidence", "cam_target_class",
                                     "csv_pred", "csv_confidence", "prediction_changed", "all_zero",
                                     "warning", "heatmap", "native_crop_overlay"])
        for index, row in enumerate(selected):
            with Image.open(data / row["path"]) as source:
                image = source.convert("RGB")
            cam, predicted, confidence, all_zero = _gradcam(model, image, transform)
            name = EXPECTED_CLASSES[predicted]
            stem = f"{index + 1:02d}_{hashlib.sha256(row['path'].encode()).hexdigest()[:12]}"
            heat_path, overlay_path = _save_visualizations(image, cam, all_zero, output, stem)
            caption = (f"true={row['true']} | pred={name} | confidence({name})={confidence:.6f} | "
                       f"CAM target=PREDICTED {name} | all_zero={all_zero} | {CAM_WARNING}")
            item = {
                "path": row["path"], "true": row["true"], "pred": name, "confidence": confidence,
                "cam_target_class": name, "csv_pred": row["pred"], "csv_confidence": row["confidence"],
                "prediction_changed": name != row["pred"], "all_zero": all_zero,
                "native_crop_size_wh": list(image.size), "feature_map_size_hw": list(cam.shape),
                "heatmap": heat_path.relative_to(run_dir).as_posix(),
                "overlay": overlay_path.relative_to(run_dir).as_posix(), "caption": caption,
            }
            report["samples"].append(item)
            table.add_data(row["path"], row["true"], name, confidence, name, row["pred"],
                           row["confidence"], item["prediction_changed"], all_zero, CAM_WARNING,
                           wandb.Image(str(heat_path), caption=caption),
                           wandb.Image(str(overlay_path), caption=caption))
        latency = _benchmark(model, metadata["args"]["img_size"], args.batch)
        latency.update(variant=variant, checkpoint_epoch=summary["exported_epoch"],
                       preprocessing=metadata["preprocessing"])
        _write_json(output / "latency.json", latency)
        report.update(status="completed", latency_file="diagnostics/latency.json",
                      latency=latency, elapsed_seconds=time.monotonic() - started)
        _write_json(output / "diagnostics.json", report)
        values = {"diagnostics/gradcam": table}
        for result in latency["results"]:
            for key in ("batch_p50_ms", "batch_p95_ms", "item_p50_ms", "item_p95_ms", "peak_allocated_bytes"):
                values[f"diagnostics/classifier_only/batch{result['batch_size']}/{key}"] = result[key]
        run.log(values)  # Append to resumed history; never reset epoch/global step.
        run.summary["diagnostics"] = report
        for path in sorted(output.iterdir()):
            run.save(str(path), base_path=str(run_dir), policy="now")
        exit_code = 0
    except BaseException as exc:
        # Do not serialize exception messages/tracebacks: SDK errors can contain credentials.
        report.update(status="failed", error_type=type(exc).__name__,
                      elapsed_seconds=time.monotonic() - started)
        _write_json(output / "diagnostics.json", report)
        raise
    finally:
        if run is not None:
            try:
                run.finish(exit_code=exit_code)
            except BaseException as exc:
                report.update(status="failed", error_type=type(exc).__name__, stage="wandb_finish")
                _write_json(output / "diagnostics.json", report)
                raise
    print("Diagnostico concluido: PNGs, latency.json e diagnostics.json no diretorio diagnostics.")
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        context = _runtime_guard(args)
        return _run(args, context)
    except Exception as exc:
        print(f"Diagnostico interrompido ({type(exc).__name__}); verifique Slurm/Apptainer, "
              "export/manifesto, diretorio de saida e acesso ao W&B existente. "
              "Detalhes da excecao omitidos para proteger credenciais.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())