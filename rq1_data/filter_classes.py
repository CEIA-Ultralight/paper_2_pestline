#!/usr/bin/env python3
"""Cria uma CÓPIA do dataset YOLO removendo classes específicas (ex.: NOISE, MAR),
mantendo as demais (inclui INS). NUNCA modifica o original — escreve em <dst>.

Remapeia os ids de classe para ficarem contíguos (0..K-1) e reescreve data.yaml.
Imagens sem nenhum objeto restante viram "background" (label vazio, mantido).

Uso:
  python filter_classes.py --src /raid/.../DS-F2_v8.1.1 \
      --dst /raid/.../DS-F2_v8.1.1_no-noise-mar \
      --drop NOISE MAR
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--drop", nargs="+", required=True, help="nomes de classes a remover")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    drop = set(args.drop)

    names = yaml.safe_load((src / "data.yaml").read_text())["names"]
    keep = [n for n in names if n not in drop]
    # mapa: id antigo -> id novo (ou None p/ remover)
    id_map = {i: (keep.index(n) if n in keep else None) for i, n in enumerate(names)}
    print("classes mantidas:", keep)
    print("removendo:", sorted(drop))

    for split in ["train", "val", "test"]:
        si, li = src / "images" / split, src / "labels" / split
        if not si.is_dir():
            continue
        oi, ol = dst / "images" / split, dst / "labels" / split
        oi.mkdir(parents=True, exist_ok=True)
        ol.mkdir(parents=True, exist_ok=True)
        n_img = n_kept = n_drop = 0
        for img in sorted(si.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            stem = img.stem
            shutil.copy2(img, oi / img.name)
            n_img += 1
            lines = []
            lf = li / f"{stem}.txt"
            if lf.is_file():
                for line in lf.read_text().splitlines():
                    p = line.split()
                    if len(p) < 5:
                        continue
                    nid = id_map.get(int(p[0]))
                    if nid is None:
                        n_drop += 1
                        continue
                    lines.append(" ".join([str(nid)] + p[1:5]))
                    n_kept += 1
            (ol / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"[{split}] {n_img} imgs | caixas mantidas={n_kept} removidas={n_drop}")

    (dst / "data.yaml").write_text(
        f"path: {dst}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + "".join(f"- {n}\n" for n in keep)
    )
    print(f"OK -> {dst} (original em {src} intacto)")


if __name__ == "__main__":
    main()
