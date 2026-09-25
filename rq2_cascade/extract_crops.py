#!/usr/bin/env python3
"""Extrai crops por objeto de um dataset YOLO para treinar um classificador de espécies (E3).

Para cada label (cls cx cy w h, normalizado), recorta a bbox da imagem em
RESOLUÇÃO NATIVA (sem downscale), com padding de contexto, e salva em
<dst>/<split>/<CLASSE>/<img>_<i>.jpg — formato ImageFolder p/ treino de classificação.

Uso:
  python extract_crops.py --src /raid/.../DS-F2_v8.1.1 --dst /raid/.../DS-F2_crops \
      --pad 0.15 --min-size 8
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise SystemExit("opencv (cv2) necessário") from e

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_names(src: Path):
    y = src / "data.yaml"
    if y.is_file():
        import yaml
        return yaml.safe_load(y.read_text()).get("names", [])
    return []


def extract_split(src: Path, dst: Path, split: str, names, pad: float, min_size: int):
    img_dir = src / "images" / split
    lbl_dir = src / "labels" / split
    counts = Counter()
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        stem = img_path.stem
        lbl = lbl_dir / f"{stem}.txt"
        if not lbl.is_file():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        for i, line in enumerate(lbl.read_text().splitlines()):
            p = line.split()
            if len(p) < 5:
                continue
            c = int(p[0])
            cx, cy, bw, bh = map(float, p[1:5])
            # YOLO normalizado -> px
            x0 = (cx - bw / 2) * w
            y0 = (cy - bh / 2) * h
            x1 = (cx + bw / 2) * w
            y1 = (cy + bh / 2) * h
            # padding de contexto
            pw, ph = (x1 - x0) * pad, (y1 - y0) * pad
            xa, ya = max(0, int(x0 - pw)), max(0, int(y0 - ph))
            xb, yb = min(w, int(x1 + pw)), min(h, int(y1 + ph))
            if (xb - xa) < min_size or (yb - ya) < min_size:
                continue
            crop = img[ya:yb, xa:xb]
            cname = names[c] if c < len(names) else f"cls{c}"
            out = dst / split / cname
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / f"{stem}_{i:03d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            counts[cname] += 1
    print(f"[{split}] crops por classe: {dict(sorted(counts.items()))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--min-size", type=int, default=8)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    names = read_names(src)
    for split in ["train", "val", "test"]:
        if (src / "images" / split).is_dir():
            extract_split(src, dst, split, names, args.pad, args.min_size)
    print(f"OK -> {dst}")


if __name__ == "__main__":
    main()
