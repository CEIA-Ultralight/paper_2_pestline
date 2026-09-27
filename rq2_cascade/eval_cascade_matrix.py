#!/usr/bin/env python3
"""Harness da cascata (detector -> classificador de crops) com predicao POR INSTANCIA.

Para cada tripla (detector_ckpt, classifier_ckpt, policy) escreve em
    <out>/<detector>/<classifier>/<policy>/
        per_instance.csv        1 linha por GT de especie: image, gt_index, gt_class, gt_box,
                                gt_size, matched, iou, pred_class_single, pred_class_cascade,
                                score_single, score_cascade, correct_single, correct_cascade
        per_image_preds.jsonl   1 linha por imagem: GT + TODAS as propostas (conf >= .001) com
                                classe/score YOLO e classe/score/probabilidades da cascata
        metrics.json            P/R/F1 por classe (single e cascade), acuracia de especie,
                                latencia por estagio (detector, classificador) e contrato
    <out>/manifest.json         checkpoints (sha256), split, imagens, versoes, args

Matching (McNemar-ready): greedy classe-agnostica IoU >= --iou-match sobre as propostas com
yolo_conf >= --conf-match. O MESMO par (GT, proposta) alimenta pred_class_single (classe YOLO)
e pred_class_cascade (classe do classificador) -> correct_single/correct_cascade sao pareados
por instancia. Sem classificador (--classifier ausente) grava so o ramo "single".

Uso (GPU, dentro do Slurm/Apptainer; ver rq2_cascade/slurm/chain_D_cascade_cross.sh):
    python eval_cascade_matrix.py --detector D1/best.pt D2/best.pt \
        --classifier C1/best.pt C2/best.pt --policies label_only product \
        --data /raid/user_marcospaulo/datasets/DS-F2_v8.1.1 --split test --out $WORK/matrix

As funcoes puras (rows_for_image, prf_from_rows, derive_name, list_split, ...) rodam em CPU e
sao cobertas por tests/test_eval_cascade_matrix.py com deteccoes sinteticas.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import functools
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import eval_yolo_cascade as EV

YOLO_CLASSES = EV.YOLO_CLASSES  # ('MD','MV','MC','MF','INS','NOISE','MAR')
SPECIES = ("MD", "MV", "MC", "MF")
POLICIES = ("label_only", "product")
PREDICT = dict(EV.PREDICT)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PER_INSTANCE_COLUMNS = (
    "image", "gt_index", "gt_class", "gt_box", "gt_size", "matched", "iou",
    "pred_class_single", "pred_class_cascade", "score_single", "score_cascade",
    "correct_single", "correct_cascade",
)
DEFAULT_WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "paper2-rq2-cascade")


# --------------------------------------------------------------------------- puras
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_name(path) -> str:
    """Nome curto de um checkpoint: primeiro ancestral que nao e weights/runs/*.pt/best|last.

    .../flydet_runs/E0_s0/runs/yolo26m.pt/weights/best.pt -> 'E0_s0'
    .../flydet_runs/clf_convnext_t_ce_s42/best.pt          -> 'clf_convnext_t_ce_s42'
    """
    p = Path(path).resolve()
    for parent in p.parents:
        name = parent.name
        if name in ("weights", "runs", "train", "") or name.endswith(".pt"):
            continue
        return name
    return p.stem


def list_split(data_root: Path, split: str) -> list[tuple[Path, Path]]:
    """Pares (imagem, label) de images/<split> e labels/<split>; label ausente = sem GT."""
    img_dir, lbl_dir = data_root / "images" / split, data_root / "labels" / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"split ausente: {img_dir}")
    pairs = []
    for img in sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        pairs.append((img, lbl_dir / img.relative_to(img_dir).with_suffix(".txt")))
    if not pairs:
        raise FileNotFoundError(f"nenhuma imagem em {img_dir}")
    return pairs


def load_gt(label_path: Path, width: int, height: int) -> list[dict]:
    """GT YOLO normalizado -> [{'class': int, 'box': [x1,y1,x2,y2]}] em pixels."""
    if not label_path.is_file():
        return []
    gt = []
    for line in label_path.read_text().splitlines():
        f = line.split()
        if len(f) != 5:
            continue
        c, cx, cy, w, h = int(f[0]), *map(float, f[1:])
        gt.append({"class": c, "box": [(cx - w / 2) * width, (cy - h / 2) * height,
                                       (cx + w / 2) * width, (cy + h / 2) * height]})
    return gt


def relabel_named(prediction: dict, probabilities, class_order) -> dict:
    """Aplica o softmax do classificador (ordem `class_order`) a uma proposta YOLO.

    Retorna copia com class (indice YOLO), probabilities (ordem YOLO, None onde a classe
    nao existe no classificador) e fallback ('degenerate_crop' | 'background' | None).
    Argmax em 'BG' mantem a classe YOLO e marca fallback='background'.
    """
    out = dict(prediction, probabilities=None, fallback=None)
    if probabilities is None:
        out["fallback"] = "degenerate_crop"
        return out
    if len(probabilities) != len(class_order):
        raise ValueError("softmax e class_order com tamanhos diferentes")
    by_name = dict(zip(class_order, probabilities, strict=True))
    out["probabilities"] = [by_name.get(n) for n in YOLO_CLASSES]
    best = max(class_order, key=by_name.__getitem__)
    if best in YOLO_CLASSES:
        out["class"] = YOLO_CLASSES.index(best)
    else:
        out["fallback"] = "background"
    return out


def apply_policy(prediction: dict, policy: str) -> float:
    """Score da cascata: label_only = conf YOLO; product = conf YOLO * max softmax."""
    if policy not in POLICIES:
        raise ValueError(f"policy desconhecida: {policy}")
    conf = prediction["yolo_confidence"]
    probs = [p for p in (prediction.get("probabilities") or []) if p is not None]
    if policy == "product" and probs:
        return conf * max(probs)
    return conf


def rows_for_image(image: str, gt: list[dict], single: list[dict], cascade: list[dict] | None,
                   policy: str = "label_only", conf_match: float = 0.25,
                   iou_match: float = 0.5) -> list[dict]:
    """Linhas de per_instance.csv para uma imagem (1 por GT).

    `single` e `cascade` sao listas paralelas (mesma geometria, mesma ordem); `cascade`
    pode ser None (so detector). Matching classe-agnostico greedy sobre propostas com
    yolo_confidence >= conf_match; o mesmo par alimenta os dois ramos.
    """
    if cascade is not None and len(cascade) != len(single):
        raise ValueError("single e cascade devem ter o mesmo numero de propostas")
    keep = [i for i, p in enumerate(single) if p["yolo_confidence"] >= conf_match]
    selected = [single[i] for i in keep]
    matches = {g: (keep[p], iou) for g, p, iou in EV.greedy(EV.candidates(gt, selected), iou_match)}
    rows = []
    for g, t in enumerate(gt):
        gt_name = YOLO_CLASSES[t["class"]]
        x1, y1, x2, y2 = t["box"]
        row = {
            "image": image, "gt_index": g, "gt_class": gt_name,
            "gt_box": ",".join(f"{v:.2f}" for v in t["box"]),
            "gt_size": round(math.sqrt(max(x2 - x1, 0.0) * max(y2 - y1, 0.0)), 2),
            "matched": False, "iou": 0.0,
            "pred_class_single": "", "pred_class_cascade": "",
            "score_single": 0.0, "score_cascade": 0.0,
            "correct_single": False, "correct_cascade": False,
        }
        if g in matches:
            idx, iou = matches[g]
            s = single[idx]
            row.update(matched=True, iou=round(iou, 4),
                       pred_class_single=YOLO_CLASSES[s["class"]],
                       score_single=round(s["yolo_confidence"], 6))
            row["correct_single"] = row["pred_class_single"] == gt_name
            if cascade is not None:
                c = cascade[idx]
                row.update(pred_class_cascade=YOLO_CLASSES[c["class"]],
                           score_cascade=round(apply_policy(c, policy), 6))
                row["correct_cascade"] = row["pred_class_cascade"] == gt_name
        rows.append(row)
    return rows


def false_positives(gt: list[dict], preds: list[dict], conf_match: float = 0.25,
                    iou_match: float = 0.5) -> dict[str, int]:
    """Propostas (conf >= conf_match) sem GT correspondente, contadas por classe predita."""
    selected = [p for p in preds if p["yolo_confidence"] >= conf_match]
    used = {p for _, p, _ in EV.greedy(EV.candidates(gt, selected), iou_match)}
    fp = dict.fromkeys(YOLO_CLASSES, 0)
    for i, p in enumerate(selected):
        if i not in used:
            fp[YOLO_CLASSES[p["class"]]] += 1
    return fp


def prf_from_rows(rows: list[dict], fp: dict[str, int], branch: str) -> dict:
    """Precisao/recall/F1 por classe + acuracia de especie a partir das linhas per-instance.

    branch: 'single' | 'cascade'. recall = TP / n_GT; precision = TP / (TP + FP_da_classe +
    GT_de_outra_classe_previsto_como_ela). Especie: MD/MV/MC/MF.
    """
    pred_col, ok_col = f"pred_class_{branch}", f"correct_{branch}"
    per_class = {}
    for name in YOLO_CLASSES:
        support = sum(r["gt_class"] == name for r in rows)
        tp = sum(r["gt_class"] == name and r[ok_col] for r in rows)
        predicted = sum(r["matched"] and r[pred_col] == name for r in rows) + fp.get(name, 0)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"support": support, "tp": tp, "predicted": predicted,
                           "fp_unmatched": fp.get(name, 0), "precision": precision,
                           "recall": recall, "f1": f1}
    sp = [r for r in rows if r["gt_class"] in SPECIES]
    matched = [r for r in sp if r["matched"]]
    correct = sum(r[ok_col] for r in sp)
    return {
        "per_class": per_class,
        "species_support": len(sp), "species_matched": len(matched), "species_correct": correct,
        "species_detection_recall": len(matched) / len(sp) if sp else 0.0,
        "species_accuracy_all_gt": correct / len(sp) if sp else 0.0,
        "species_accuracy_conditional_matched": correct / len(matched) if matched else 0.0,
        "species_macro_recall": sum(per_class[n]["recall"] for n in SPECIES) / len(SPECIES),
        "species_macro_precision": sum(per_class[n]["precision"] for n in SPECIES) / len(SPECIES),
        "species_macro_f1": sum(per_class[n]["f1"] for n in SPECIES) / len(SPECIES),
    }


def latency_summary(seconds: list[float]) -> dict:
    if not seconds:
        return {"n": 0}
    xs = sorted(seconds)
    mean = sum(xs) / len(xs)
    std = math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))

    def pct(q):
        return xs[min(len(xs) - 1, round(q * (len(xs) - 1)))]

    return {"n": len(xs), "mean_s": mean, "std_s": std, "p50_s": pct(0.5), "p95_s": pct(0.95),
            "min_s": xs[0], "max_s": xs[-1]}


def write_per_instance(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PER_INSTANCE_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in PER_INSTANCE_COLUMNS})


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")


def assemble_outputs(image_records: list[dict], policy: str, with_cascade: bool,
                     conf_match: float, iou_match: float) -> tuple[list[dict], list[dict], dict]:
    """De registros por imagem (gt, single, cascade, seconds) -> (rows, jsonl_records, metrics).

    Funcao pura usada pelo caminho GPU e pelos testes sinteticos.
    """
    rows, jsonl, fp_single, fp_cascade = [], [], dict.fromkeys(YOLO_CLASSES, 0), dict.fromkeys(YOLO_CLASSES, 0)
    det_s, clf_s = [], []
    for rec in image_records:
        gt, single = rec["gt"], rec["single"]
        cascade = rec.get("cascade") if with_cascade else None
        rows.extend(rows_for_image(rec["image"], gt, single, cascade, policy, conf_match, iou_match))
        for name, count in false_positives(gt, single, conf_match, iou_match).items():
            fp_single[name] += count
        if cascade is not None:
            for name, count in false_positives(gt, cascade, conf_match, iou_match).items():
                fp_cascade[name] += count
        det_s.append(rec["seconds"]["detector"])
        if cascade is not None:
            clf_s.append(rec["seconds"].get("classifier", 0.0))
        proposals = []
        for i, s in enumerate(single):
            item = {"box": [round(v, 2) for v in s["box"]], "yolo_class": YOLO_CLASSES[s["class"]],
                    "yolo_conf": round(s["yolo_confidence"], 6)}
            if cascade is not None:
                c = cascade[i]
                item.update(cascade_class=YOLO_CLASSES[c["class"]],
                            cascade_score=round(apply_policy(c, policy), 6),
                            probabilities=None if c["probabilities"] is None
                            else [None if p is None else round(p, 6) for p in c["probabilities"]],
                            fallback=c.get("fallback"))
            proposals.append(item)
        jsonl.append({"image": rec["image"],
                      "gt": [{"class": YOLO_CLASSES[t["class"]], "box": [round(v, 2) for v in t["box"]]}
                             for t in gt],
                      "proposals": proposals, "seconds": rec["seconds"]})
    metrics = {"policy": policy, "conf_match": conf_match, "iou_match": iou_match,
               "images": len(image_records), "gt_objects": len(rows),
               "single": prf_from_rows(rows, fp_single, "single"),
               "latency": {"detector": latency_summary(det_s)},
               "probability_class_order": list(YOLO_CLASSES)}
    if with_cascade:
        metrics["cascade"] = prf_from_rows(rows, fp_cascade, "cascade")
        metrics["latency"]["classifier_extra"] = latency_summary(clf_s)
        metrics["latency"]["end_to_end"] = latency_summary([a + b for a, b in zip(det_s, clf_s, strict=True)])
        metrics["paired_discordant"] = {
            "single_right_cascade_wrong": sum(r["correct_single"] and not r["correct_cascade"] for r in rows),
            "single_wrong_cascade_right": sum(r["correct_cascade"] and not r["correct_single"] for r in rows),
        }
    return rows, jsonl, metrics


# --------------------------------------------------------------------------- GPU
def load_classifier(path: Path, torch, transforms):
    """Carrega best.pt do architecture_lab (metadata + state_dict) e reconstroi o modelo."""
    from architecture_lab.models import build_model

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["metadata"]
    class_order = tuple(meta["class_order"])
    variant = meta["architecture"]["variant"]
    model = build_model(variant, len(class_order), pretrained=False).float().eval()
    model.load_state_dict(ckpt["model"], strict=True)
    pre = meta["preprocessing"]
    size = tuple(pre.get("resize", [384, 384]))
    norm = pre["normalization"]
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(), transforms.Normalize(norm["mean"], norm["std"])])
    info = {"path": str(path), "sha256": sha256(path), "variant": variant, "class_order": list(class_order),
            "epoch": ckpt.get("epoch"), "input": pre.get("input"), "resize": list(size)}
    return model.cuda(), transform, class_order, info


def classify_proposals(image, predictions, model, transform, class_order, torch, batch=32):
    out = [relabel_named(p, None, class_order) for p in predictions]
    valid = [(i, EV.padded_box(p["box"])) for i, p in enumerate(predictions)]
    valid = [(i, b) for i, b in valid if b is not None]
    for start in range(0, len(valid), batch):
        chunk = valid[start:start + batch]
        tensor = torch.stack([transform(image.crop(b)) for _, b in chunk]).to("cuda", dtype=torch.float32)
        probs = model(tensor)["logits"].softmax(dim=1).cpu().tolist()
        for (i, _), p in zip(chunk, probs, strict=True):
            out[i] = relabel_named(predictions[i], p, class_order)
    return out


def run(args) -> int:
    EV.allocation_guard()
    import torch

    EV.require(torch.cuda.is_available(), "CUDA obrigatoria; sem fallback CPU")
    import numpy as np
    import ultralytics
    from PIL import Image
    from torchvision import transforms
    from ultralytics import YOLO

    torch.backends.cudnn.benchmark = False
    data_root = Path(args.data).resolve()
    pairs = list_split(data_root, args.split)[: args.limit_images]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    detectors = [Path(p).resolve() for p in args.detector]
    classifiers = [Path(p).resolve() for p in args.classifier]
    det_names = args.detector_names or [derive_name(p) for p in detectors]
    clf_names = args.classifier_names or [derive_name(p) for p in classifiers]
    EV.require(len(det_names) == len(detectors) and len(clf_names) == len(classifiers),
               "--detector-names/--classifier-names devem casar com os checkpoints")
    started = dt.datetime.now(dt.UTC).isoformat()
    manifest = {"split": args.split, "data": str(data_root), "images": len(pairs), "started_utc": started,
                "detectors": {n: {"path": str(p), "sha256": sha256(p)} for n, p in zip(det_names, detectors, strict=True)},
                "classifiers": {}, "policies": list(args.policies), "predict": PREDICT,
                "conf_match": args.conf_match, "iou_match": args.iou_match,
                "versions": {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                             "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "args": vars(args) | {"out": str(out)}}

    heads = {}
    for name, path in zip(clf_names, classifiers, strict=True):
        model, transform, order, info = load_classifier(path, torch, transforms)
        heads[name] = (model, transform, order)
        manifest["classifiers"][name] = info
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    wandb_run = None
    if args.wandb:
        import wandb

        run_name = args.wandb_run_name or os.environ.get("WANDB_RUN_NAME") or \
            f"EVAL_matrix_{args.split}_{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}"
        wandb_run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=run_name,
                               job_type="eval", dir=str(out), config=manifest)
        print(f"[wandb] run={wandb_run.name} url={wandb_run.url}", flush=True)

    summary = {}
    with torch.inference_mode():
        for det_name, det_path in zip(det_names, detectors, strict=True):
            detector = YOLO(str(det_path))
            detector.model.float().eval()
            EV.class_names(detector.names)
            for _ in range(2):  # warmup
                detector.predict(source=np.zeros((1080, 1920, 3), dtype=np.uint8), **PREDICT)
            # Estagio 1: detector em todas as imagens (uma vez por detector).
            base = []
            for img_path, lbl_path in pairs:
                with Image.open(img_path) as raw:
                    image = raw.convert("RGB")
                gt = load_gt(lbl_path, image.width, image.height)
                bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
                outputs, det_s = EV.timed(torch, functools.partial(detector.predict, source=bgr, **PREDICT))
                boxes = outputs[0].boxes
                single = [{"box": b, "class": int(c), "confidence": s, "old_class": int(c),
                           "yolo_confidence": s, "probabilities": None, "fallback": None}
                          for b, c, s in zip(boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(),
                                             boxes.conf.cpu().tolist(), strict=True)]
                base.append({"image": img_path.name, "pil": image, "gt": gt, "single": single,
                             "seconds": {"detector": det_s}})
            # Sem classificador: ramo single.
            rows, jsonl, metrics = assemble_outputs(
                [{k: v for k, v in r.items() if k != "pil"} for r in base], "label_only", False,
                args.conf_match, args.iou_match)
            _emit(out / det_name / "single" / "label_only", rows, jsonl, metrics, summary,
                  f"{det_name}/single/label_only", wandb_run)
            # Estagio 2: cada classificador sobre as MESMAS propostas.
            for clf_name, (model, transform, order) in heads.items():
                for _ in range(2):
                    model(torch.zeros((1, 3, 384, 384), device="cuda"))["logits"]
                recs = []
                for r in base:
                    cascade, clf_s = EV.timed(
                        torch, functools.partial(classify_proposals, r["pil"], r["single"], model, transform,
                                                 order, torch, args.batch))
                    recs.append({"image": r["image"], "gt": r["gt"], "single": r["single"], "cascade": cascade,
                                 "seconds": {"detector": r["seconds"]["detector"], "classifier": clf_s}})
                for policy in args.policies:
                    rows, jsonl, metrics = assemble_outputs(recs, policy, True, args.conf_match, args.iou_match)
                    metrics["classifier"] = manifest["classifiers"][clf_name]
                    _emit(out / det_name / clf_name / policy, rows, jsonl, metrics, summary,
                          f"{det_name}/{clf_name}/{policy}", wandb_run)
            detector = None
            torch.cuda.empty_cache()
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if wandb_run is not None:
        wandb_run.save(str(out / "summary.json"), base_path=str(out), policy="now")
        wandb_run.save(str(out / "manifest.json"), base_path=str(out), policy="now")
        wandb_run.finish()
    print(f"[eval_cascade_matrix] OK -> {out}", flush=True)
    return 0


def _emit(folder: Path, rows, jsonl, metrics, summary, key, wandb_run):
    folder.mkdir(parents=True, exist_ok=True)
    write_per_instance(rows, folder / "per_instance.csv")
    write_jsonl(jsonl, folder / "per_image_preds.jsonl")
    metrics["detector_name"], metrics["classifier_name"], metrics["policy_name"] = key.split("/")
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2))
    flat = {}
    for branch in ("single", "cascade"):
        if branch in metrics:
            m = metrics[branch]
            flat.update({f"{branch}/species_accuracy_all_gt": m["species_accuracy_all_gt"],
                         f"{branch}/species_accuracy_conditional": m["species_accuracy_conditional_matched"],
                         f"{branch}/species_macro_recall": m["species_macro_recall"],
                         f"{branch}/species_macro_precision": m["species_macro_precision"],
                         f"{branch}/species_macro_f1": m["species_macro_f1"]})
            for cls in SPECIES:
                flat[f"{branch}/{cls}/recall"] = m["per_class"][cls]["recall"]
                flat[f"{branch}/{cls}/precision"] = m["per_class"][cls]["precision"]
    flat["latency/detector_mean_s"] = metrics["latency"]["detector"].get("mean_s")
    if "classifier_extra" in metrics["latency"]:
        flat["latency/classifier_extra_mean_s"] = metrics["latency"]["classifier_extra"].get("mean_s")
    summary[key] = flat
    if wandb_run is not None:
        wandb_run.summary.update({f"{key}/{k}": v for k, v in flat.items()})
    acc = flat.get("cascade/species_accuracy_all_gt", flat.get("single/species_accuracy_all_gt"))
    print(f"[eval_cascade_matrix] {key}: species_acc_all_gt={acc:.4f} n_gt={metrics['gt_objects']}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detector", nargs="+", required=True, help="best.pt YOLO (1+)")
    p.add_argument("--detector-names", nargs="*", default=None)
    p.add_argument("--classifier", nargs="*", default=[], help="best.pt do architecture_lab (0+)")
    p.add_argument("--classifier-names", nargs="*", default=None)
    p.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES)
    p.add_argument("--data", required=True, help="raiz do dataset com images/<split> e labels/<split>")
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--out", required=True)
    p.add_argument("--conf-match", type=float, default=0.25)
    p.add_argument("--iou-match", type=float, default=0.5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--limit-images", type=int, default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-entity", default="pestline")
    p.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    p.add_argument("--wandb-run-name", default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.perf_counter()
    code = run(args)
    print(f"[eval_cascade_matrix] {time.perf_counter() - t0:.1f}s", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
