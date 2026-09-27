#!/usr/bin/env python3
"""Oraculo de identificacao: classificador de crops aplicado as caixas GT (sem detector).

Responde "quao bem o 2o estagio identifica a especie quando a localizacao e perfeita?".
Para cada classificador (best.pt do architecture_lab) e cada GT das 4 especies-alvo
(MD MV MC MF) do split, recorta o crop da caixa GT com o MESMO padding/pre-processamento
da cascata (eval_yolo_cascade.padded_box: pad 0.75 por lado, int + clipping; resize/normalize
do metadata do checkpoint via eval_cascade_matrix.load_classifier), classifica e escreve:

    <out>/<classifier>/per_instance.csv   image, gt_index, gt_class, gt_box, gt_size, pred_class,
                                          score, correct   (1 linha por GT de especie)
    <out>/<classifier>/metrics.json       species_accuracy, P/R/F1 por especie + macro,
                                          confusion (linhas = GT especie, colunas = classes do
                                          classificador), n_pred_outside_species, degenerate_crops
    <out>/summary.json                    metricas planas por classificador
    <out>/manifest.json                   checkpoints (sha256), split, imagens, versoes, args

Protocolo de especie identico ao de eval_cascade_matrix.py: so GT em SPECIES entram; o
classificador pode prever classes fora das 4 (INS/NOISE/MAR — ordem do checkpoint), o que
conta como erro e e reportado em n_pred_outside_species. Crop degenerado (caixa sem area)
-> pred_class vazio, score 0, correct False, contado em degenerate_crops.

Uso (GPU, dentro do Slurm/Apptainer; ver rq2_cascade/slurm/chain_D_cascade_cross.sh):
    python eval_oracle_crops.py --classifier C1/best.pt C2/best.pt \
        --data /raid/user_marcospaulo/datasets/DS-F2_v8.1.1 --split val --out $WORK/oracle_val

As funcoes puras (rows_from_probs, metrics_from_rows) rodam em CPU e sao cobertas por
tests/test_eval_oracle_crops.py com um classificador dummy.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import eval_cascade_matrix as M
import eval_yolo_cascade as EV

SPECIES = M.SPECIES
YOLO_CLASSES = M.YOLO_CLASSES
PER_INSTANCE_COLUMNS = ("image", "gt_index", "gt_class", "gt_box", "gt_size", "pred_class", "score", "correct")
load_classifier = M.load_classifier  # indirecao para monkeypatch nos testes CPU


# --------------------------------------------------------------------------- puras
def rows_from_probs(image: str, gt: list[dict], probs: list[list[float] | None], class_order) -> list[dict]:
    """1 linha por GT de especie; `probs[i]` e o softmax (ordem `class_order`) do crop do GT i ou None."""
    if len(probs) != len(gt):
        raise ValueError("probs e gt devem ter o mesmo tamanho")
    rows = []
    for g, (t, p) in enumerate(zip(gt, probs, strict=True)):
        gt_name = YOLO_CLASSES[t["class"]]
        if gt_name not in SPECIES:
            continue
        x1, y1, x2, y2 = t["box"]
        row = {"image": image, "gt_index": g, "gt_class": gt_name,
               "gt_box": ",".join(f"{v:.2f}" for v in t["box"]),
               "gt_size": round(math.sqrt(max(x2 - x1, 0.0) * max(y2 - y1, 0.0)), 2),
               "pred_class": "", "score": 0.0, "correct": False}
        if p is not None:
            if len(p) != len(class_order):
                raise ValueError("softmax e class_order com tamanhos diferentes")
            k = max(range(len(p)), key=p.__getitem__)
            row.update(pred_class=class_order[k], score=round(float(p[k]), 6))
            row["correct"] = class_order[k] == gt_name
        rows.append(row)
    return rows


def metrics_from_rows(rows: list[dict], class_order) -> dict:
    """Acuracia de especie, P/R/F1 por especie + macro e matriz de confusao GT(especie) x pred."""
    labels = list(class_order)
    confusion = {g: dict.fromkeys(labels + [""], 0) for g in SPECIES}
    for r in rows:
        confusion[r["gt_class"]][r["pred_class"]] += 1
    per_class = {}
    for name in SPECIES:
        support = sum(r["gt_class"] == name for r in rows)
        tp = sum(r["gt_class"] == name and r["correct"] for r in rows)
        predicted = sum(r["pred_class"] == name for r in rows)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"support": support, "tp": tp, "predicted": predicted,
                           "precision": precision, "recall": recall, "f1": f1}
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    degenerate = sum(r["pred_class"] == "" for r in rows)
    outside = sum(r["pred_class"] not in SPECIES and r["pred_class"] != "" for r in rows)
    return {
        "species_support": n, "species_correct": correct,
        "species_accuracy": correct / n if n else 0.0,
        "species_macro_recall": sum(per_class[c]["recall"] for c in SPECIES) / len(SPECIES),
        "species_macro_precision": sum(per_class[c]["precision"] for c in SPECIES) / len(SPECIES),
        "species_macro_f1": sum(per_class[c]["f1"] for c in SPECIES) / len(SPECIES),
        "per_class": per_class,
        "confusion": {"gt_rows": list(SPECIES), "pred_columns": labels + [""],
                      "counts": {g: [confusion[g][c] for c in labels + [""]] for g in SPECIES}},
        "n_pred_outside_species": outside, "degenerate_crops": degenerate,
        "classifier_class_order": labels,
    }


def write_per_instance(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PER_INSTANCE_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in PER_INSTANCE_COLUMNS})


def flat_metrics(metrics: dict) -> dict:
    flat = {k: metrics[k] for k in ("species_accuracy", "species_macro_recall", "species_macro_precision",
                                    "species_macro_f1", "n_pred_outside_species", "degenerate_crops")}
    for c in SPECIES:
        flat[f"{c}/recall"] = metrics["per_class"][c]["recall"]
        flat[f"{c}/precision"] = metrics["per_class"][c]["precision"]
        flat[f"{c}/f1"] = metrics["per_class"][c]["f1"]
    return flat


# --------------------------------------------------------------------------- GPU/CPU
def classify_gt_crops(image, gt: list[dict], model, transform, torch, device: str,
                      batch: int = 32) -> list[list[float] | None]:
    """Softmax por GT (None para crop degenerado); crops via EV.padded_box no tamanho real da imagem."""
    out: list[list[float] | None] = [None] * len(gt)
    valid = [(i, EV.padded_box(t["box"], image.width, image.height)) for i, t in enumerate(gt)]
    valid = [(i, b) for i, b in valid if b is not None]
    for start in range(0, len(valid), batch):
        chunk = valid[start:start + batch]
        tensor = torch.stack([transform(image.crop(b)) for _, b in chunk]).to(device, dtype=torch.float32)
        probs = model(tensor)["logits"].softmax(dim=1).cpu().tolist()
        for (i, _), p in zip(chunk, probs, strict=True):
            out[i] = p
    return out


def run(args) -> int:
    smoke_cpu = args.device == "cpu"
    if smoke_cpu:
        EV.require(args.limit_images is not None, "--device cpu e smoke-only: exige --limit-images")
    else:
        EV.allocation_guard()
    import torch

    if not smoke_cpu:
        EV.require(torch.cuda.is_available(), "CUDA obrigatoria; --device cpu so para smoke")
    from PIL import Image
    from torchvision import transforms

    data_root = Path(args.data).resolve()
    pairs = M.list_split(data_root, args.split)[: args.limit_images]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    classifiers = [Path(p).resolve() for p in args.classifier]
    clf_names = args.classifier_names or [M.derive_name(p) for p in classifiers]
    EV.require(len(clf_names) == len(classifiers), "--classifier-names deve casar com os checkpoints")
    manifest = {"split": args.split, "data": str(data_root), "images": len(pairs),
                "started_utc": dt.datetime.now(dt.UTC).isoformat(), "classifiers": {},
                "crop": "eval_yolo_cascade.padded_box (pad 0.75/lado, int, clipping) sobre a caixa GT",
                "species": list(SPECIES), "device": args.device,
                "versions": {"torch": torch.__version__,
                             "gpu": torch.cuda.get_device_name(0) if not smoke_cpu else None},
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "args": vars(args) | {"out": str(out)}}

    heads = {}
    for name, path in zip(clf_names, classifiers, strict=True):
        model, transform, order, info = load_classifier(path, torch, transforms)
        heads[name] = (model.to(args.device), transform, order)
        manifest["classifiers"][name] = info
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    wandb_run = None
    if args.wandb:
        import wandb

        run_name = args.wandb_run_name or os.environ.get("WANDB_RUN_NAME") or f"oracle_crops_{args.split}"
        wandb_run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=run_name,
                               job_type="eval", dir=str(out), config=manifest)
        print(f"[wandb] run={wandb_run.name} url={wandb_run.url}", flush=True)

    # Decodifica cada imagem uma vez; todos os classificadores veem os mesmos crops.
    images = []
    for img_path, lbl_path in pairs:
        with Image.open(img_path) as raw:
            image = raw.convert("RGB")
        images.append((img_path.name, image, M.load_gt(lbl_path, image.width, image.height)))

    summary = {}
    with torch.inference_mode():
        for clf_name, (model, transform, order) in heads.items():
            rows = []
            for name, image, gt in images:
                probs = classify_gt_crops(image, gt, model, transform, torch, args.device, args.batch)
                rows.extend(rows_from_probs(name, gt, probs, order))
            metrics = metrics_from_rows(rows, order)
            metrics.update(classifier_name=clf_name, classifier=manifest["classifiers"][clf_name],
                           split=args.split, images=len(images))
            folder = out / clf_name
            write_per_instance(rows, folder / "per_instance.csv")
            (folder / "metrics.json").write_text(json.dumps(metrics, indent=2))
            summary[clf_name] = flat_metrics(metrics)
            if wandb_run is not None:
                wandb_run.summary.update({f"{clf_name}/{k}": v for k, v in summary[clf_name].items()})
            print(f"[eval_oracle_crops] {clf_name}: species_acc={metrics['species_accuracy']:.4f} "
                  f"macro_f1={metrics['species_macro_f1']:.4f} n={metrics['species_support']}", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if wandb_run is not None:
        wandb_run.save(str(out / "summary.json"), base_path=str(out), policy="now")
        wandb_run.save(str(out / "manifest.json"), base_path=str(out), policy="now")
        wandb_run.finish()
    print(f"[eval_oracle_crops] OK -> {out}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--classifier", nargs="+", required=True, help="best.pt do architecture_lab (1+)")
    p.add_argument("--classifier-names", nargs="*", default=None)
    p.add_argument("--data", required=True, help="raiz do dataset com images/<split> e labels/<split>")
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--limit-images", type=int, default=None)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="cpu = smoke/testes apenas")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-entity", default="pestline")
    p.add_argument("--wandb-project", default=M.DEFAULT_WANDB_PROJECT)
    p.add_argument("--wandb-run-name", default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.perf_counter()
    code = run(args)
    print(f"[eval_oracle_crops] {time.perf_counter() - t0:.1f}s", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
