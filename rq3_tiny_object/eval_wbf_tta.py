#!/usr/bin/env python3
"""Avaliacao de detectores no VAL: vanilla, TTA e ensemble WBF (experimento item 6).

Uso (dentro do container):
    python eval_wbf_tta.py --data /path/to/data.yaml --detectors tag=ckpt.pt [tag=ckpt.pt ...]

Produz em --out:
    val_vanilla_<tag>/   — metrics do Ultralytics por detector
    val_tta_<tag>/       — idem com augment=True (TTA)
    val_wbf/             — summary.json com mAP50/mAP50:95 do ensemble WBF

WBF: funde as predicoes dos detectores por imagem (Weighted Boxes Fusion,
Solovyev et al. 2021, arXiv:1910.13302) e avalia contra o GT do data.yaml.
Implementacao propria, sem dependencia externa.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO


def wbf_fuse(boxes_list, scores_list, labels_list, iou_thr=0.55, skip_box_thr=0.0):
    """Weighted Boxes Fusion para UMA imagem.

    boxes_list: lista de np.ndarray (Ni, 4) xyxy (pixels) por modelo
    scores_list: lista de np.ndarray (Ni,) por modelo
    labels_list: lista de np.ndarray (Ni,) por modelo
    Retorna (boxes, scores, labels) fundidos.
    """
    n_models = len(boxes_list)
    all_boxes, all_scores, all_labels = [], [], []
    for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
        keep = scores > skip_box_thr
        all_boxes.append(boxes[keep])
        all_scores.append(scores[keep])
        all_labels.append(labels[keep])
    boxes = np.concatenate(all_boxes, 0)
    scores = np.concatenate(all_scores, 0)
    labels = np.concatenate(all_labels, 0)
    if len(boxes) == 0:
        return boxes.reshape(0, 4), scores, labels

    # ordena por confiança desc
    order = scores.argsort()[::-1]
    boxes, scores, labels = boxes[order], scores[order], labels[order]

    fused_boxes, fused_scores, fused_labels = [], [], []
    used = np.zeros(len(boxes), bool)
    for i in range(len(boxes)):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if used[j] or labels[j] != labels[i]:
                continue
            iou = _iou(boxes[i], boxes[j])
            if iou > iou_thr:
                cluster.append(j)
                used[j] = True
        cb = boxes[cluster]
        cs = scores[cluster]
        w = cs / cs.sum()
        fb = (cb * w[:, None]).sum(0)
        # score: media ponderada * fator de concordancia entre modelos
        fs = min(1.0, len(cluster) / n_models) * (cs.sum() / len(cluster))
        fused_boxes.append(fb)
        fused_scores.append(fs)
        fused_labels.append(labels[i])
    return np.array(fused_boxes), np.array(fused_scores), np.array(fused_labels)


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


def evaluate(pred_per_img, gt_per_img, n_classes, iou_thr=0.5):
    """AP por classe (VOC-style, ponto único iou_thr) e mAP."""
    aps = []
    for c in range(n_classes):
        preds = []  # (score, img_id, box)
        n_gt = 0
        for img_id in gt_per_img:
            gt_c = [b for b, l in gt_per_img[img_id] if l == c]
            n_gt += len(gt_c)
            for s, b, l in pred_per_img.get(img_id, []):
                if l == c:
                    preds.append((s, img_id, b))
        if n_gt == 0:
            continue
        preds.sort(key=lambda x: -x[0])
        tp, fp = np.zeros(len(preds)), np.zeros(len(preds))
        matched = {img_id: np.zeros(len([b for b, l in gt_per_img[img_id] if l == c])) for img_id in gt_per_img}
        for k, (s, img_id, b) in enumerate(preds):
            gt_c = [(i, bb) for i, (bb, l) in enumerate(gt_per_img.get(img_id, [])) if l == c]
            best_iou, best_i = 0.0, -1
            for i, bb in gt_c:
                v = _iou(b, bb)
                if v > best_iou:
                    best_iou, best_i = v, i
            if best_iou >= iou_thr and best_i >= 0 and matched[img_id][best_i] == 0:
                tp[k] = 1
                matched[img_id][best_i] = 1
            else:
                fp[k] = 1
        tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
        rec = tp_c / n_gt
        prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
        aps.append(_voc_ap(rec, prec))
    return float(np.mean(aps)) if aps else 0.0


def load_gt(data_yaml_path):
    """Le o split val do data.yaml e retorna {img_stem: [(box_xyxy_px, cls), ...]}."""
    with open(data_yaml_path) as fh:
        data = yaml.safe_load(fh)
    root = Path(data["path"])
    val = data["val"]
    val_dir = root / val if not str(val).startswith("/") else Path(val)
    imgs = sorted(p for p in val_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    gt = {}
    for img in imgs:
        lbl = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
        from PIL import Image
        with Image.open(img) as im:
            W, H = im.size
        boxes = []
        if lbl.is_file():
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                c, x, y, w, h = int(p[0]), float(p[1]) * W, float(p[2]) * H, float(p[3]) * W, float(p[4]) * H
                boxes.append(([x - w / 2, y - h / 2, x + w / 2, y + h / 2], c))
        gt[str(img)] = boxes
    return gt, data["names"], imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--detectors", nargs="+", required=True, help="tag=caminho.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.001)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gt, names, imgs = load_gt(args.data)
    n_classes = len(names)
    print(f"VAL: {len(imgs)} imagens, {sum(len(v) for v in gt.values())} GT, {n_classes} classes")

    results = {}
    preds_per_detector = {}

    for spec in args.detectors:
        tag, ckpt = spec.split("=", 1)
        if not Path(ckpt).is_file():
            print(f"SKIP {tag}: sem checkpoint {ckpt}")
            continue
        model = YOLO(ckpt)

        # vanilla
        m = model.val(data=args.data, split="val", imgsz=args.imgsz, project=str(out), name=f"val_vanilla_{tag}", verbose=False)
        results[f"vanilla_{tag}"] = {"map50": float(m.box.map50), "map5095": float(m.box.map)}
        print(f"vanilla {tag}: mAP50={m.box.map50:.4f} mAP50:95={m.box.map:.4f}")

        # TTA
        m = model.val(data=args.data, split="val", imgsz=args.imgsz, augment=True, project=str(out), name=f"val_tta_{tag}", verbose=False)
        results[f"tta_{tag}"] = {"map50": float(m.box.map50), "map5095": float(m.box.map)}
        print(f"tta     {tag}: mAP50={m.box.map50:.4f} mAP50:95={m.box.map:.4f}")

        # predicoes para WBF (conf baixo, fundimos depois)
        dets = {}
        for img in imgs:
            r = model.predict(str(img), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            labels = r.boxes.cls.cpu().numpy().astype(int)
            dets[str(img)] = (boxes, scores, labels)
        preds_per_detector[tag] = dets

    # WBF entre todos os detectores disponíveis
    if len(preds_per_detector) >= 2:
        fused = {}
        for img in imgs:
            key = str(img)
            bl, sl, ll = zip(*(preds_per_detector[t][key] for t in preds_per_detector))
            fb, fs, fl = wbf_fuse([np.asarray(b) for b in bl], [np.asarray(s) for s in sl], [np.asarray(l) for l in ll])
            fused[key] = list(zip(fs.tolist(), fb.tolist(), fl.tolist()))
        map50 = evaluate(fused, gt, n_classes, iou_thr=0.5)
        results["wbf_ensemble"] = {"map50": map50, "detectors": list(preds_per_detector)}
        print(f"WBF ensemble ({'+'.join(preds_per_detector)}): mAP50={map50:.4f}")

    (out / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"SUMMARY {out / 'summary.json'}")


if __name__ == "__main__":
    main()
