#!/usr/bin/env python3
"""Gera um dataset YOLO fatiado (tiles) para treino de tiny objects (E2 — SAHI fine-tune).

Para cada imagem de um split (train/val/test), fatia em tiles com overlap e
reescreve as anotações YOLO (cls cx cy w h, normalizadas) no sistema de
coordenadas de cada tile, descartando caixas cuja interseção com o tile é muito
pequena. A RESOLUÇÃO NÃO é reduzida: cada tile é um recorte em resolução nativa.

Uso:
  python slice_dataset.py \
      --src /raid/user_marcospaulo/datasets/DS-F2_v8.1.1 \
      --dst /raid/user_marcospaulo/datasets/DS-F2_v8.1.1_sliced512 \
      --tile 512 --overlap 0.25 --min-visibility 0.3

Saída: <dst>/{images,labels}/{train,val,test} + data.yaml pronto p/ Ultralytics.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise SystemExit("opencv (cv2) é necessário: pip install opencv-python-headless") from e

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def tile_windows(w: int, h: int, tile: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    """Janelas (x0, y0, x1, y1) cobrindo a imagem WxH com tiles e overlap."""
    stride = max(1, int(tile * (1.0 - overlap)))

    def starts(length: int) -> List[int]:
        xs = list(range(0, max(1, length - tile + 1), stride))
        if not xs or xs[-1] + tile < length:
            xs.append(max(0, length - tile))
        return sorted(set(xs))

    out = []
    for y0 in starts(h):
        for x0 in starts(w):
            out.append((x0, y0, min(x0 + tile, w), min(y0 + tile, h)))
    return out


def clip_box_to_tile(box: Tuple[float, float, float, float], tile_box, img_w, img_h, min_vis):
    """box em coords normalizadas YOLO; tile_box em px. Retorna box YOLO no tile ou None."""
    cx, cy, bw, bh = box
    # YOLO normalizado -> px absoluto
    ax0 = (cx - bw / 2) * img_w
    ay0 = (cy - bh / 2) * img_h
    ax1 = (cx + bw / 2) * img_w
    ay1 = (cy + bh / 2) * img_h
    tx0, ty0, tx1, ty1 = tile_box
    # interseção
    ix0, iy0 = max(ax0, tx0), max(ay0, ty0)
    ix1, iy1 = min(ax1, tx1), min(ay1, ty1)
    iw, ih = ix1 - ix0, iy1 - iy0
    if iw <= 0 or ih <= 0:
        return None
    # visibilidade = fração da área original que cai no tile
    orig_area = max(1e-6, (ax1 - ax0) * (ay1 - ay0))
    if (iw * ih) / orig_area < min_vis:
        return None
    tw, th = tx1 - tx0, ty1 - ty0
    ncx = ((ix0 + ix1) / 2 - tx0) / tw
    ncy = ((iy0 + iy1) / 2 - ty0) / th
    nw, nh = iw / tw, ih / th
    return (ncx, ncy, nw, nh)


def process_split(src: Path, dst: Path, split: str, tile: int, overlap: float, min_vis: float, keep_empty: bool):
    img_dir = src / "images" / split
    lbl_dir = src / "labels" / split
    out_img = dst / "images" / split
    out_lbl = dst / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    n_img = n_tile = n_box = 0
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"
        anns = []
        if lbl_path.is_file():
            for line in lbl_path.read_text().splitlines():
                p = line.split()
                if len(p) >= 5:
                    anns.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        n_img += 1
        for (tx0, ty0, tx1, ty1) in tile_windows(w, h, tile, overlap):
            crop = img[ty0:ty1, tx0:tx1]
            tile_lines = []
            for (c, cx, cy, bw, bh) in anns:
                r = clip_box_to_tile((cx, cy, bw, bh), (tx0, ty0, tx1, ty1), w, h, min_vis)
                if r is not None:
                    tile_lines.append(f"{c} {r[0]:.6f} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f}")
            if not tile_lines and not keep_empty:
                continue
            tname = f"{stem}__{tx0}_{ty0}"
            cv2.imwrite(str(out_img / f"{tname}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out_lbl / f"{tname}.txt").write_text("\n".join(tile_lines) + ("\n" if tile_lines else ""))
            n_tile += 1
            n_box += len(tile_lines)
    print(f"[{split}] {n_img} imgs -> {n_tile} tiles, {n_box} caixas")
    return n_tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--min-visibility", type=float, default=0.3)
    ap.add_argument("--keep-empty", action="store_true", help="mantém tiles sem anotação")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    names = None
    src_yaml = src / "data.yaml"
    if src_yaml.is_file():
        import yaml
        names = yaml.safe_load(src_yaml.read_text()).get("names")

    total = 0
    for split in ["train", "val", "test"]:
        if (src / "images" / split).is_dir():
            total += process_split(src, dst, split, args.tile, args.overlap, args.min_visibility, args.keep_empty)

    # data.yaml do dataset fatiado
    lines = [f"path: {dst}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    for n in (names or []):
        lines.append(f"- {n}")
    (dst / "data.yaml").write_text("\n".join(lines) + "\n")
    print(f"OK: {total} tiles -> {dst}")


if __name__ == "__main__":
    main()
