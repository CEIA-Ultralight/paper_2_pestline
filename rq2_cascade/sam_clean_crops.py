#!/usr/bin/env python3
"""E4 — Limpa o fundo dos crops usando SAM guiado pela bbox (prompt de caixa).

Para cada crop (ImageFolder), roda SAM com a bbox central como prompt e aplica
a máscara: pixels fora da máscara viram branco (remove textura da cola/placa).
Gera um novo dataset "limpo" para treinar o classificador.

Requer: pip install segment-anything + pesos sam_vit_b_01ec64.pth.

Uso:
  python sam_clean_crops.py --src /raid/.../DS-F2_crops_pad75 \
      --dst /raid/.../DS-F2_crops_sam --ckpt /raid/.../models/sam_vit_b_01ec64.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as e:
    raise SystemExit("opencv necessário") from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-type", default="vit_b")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from segment_anything import sam_model_registry, SamPredictor

    src, dst = Path(args.src), Path(args.dst)
    sam = sam_model_registry[args.model_type](checkpoint=args.ckpt).to(args.device)
    predictor = SamPredictor(sam)

    n_ok = n_fail = 0
    for split in ["train", "val", "test"]:
        sdir = src / split
        if not sdir.is_dir():
            continue
        for cls_dir in sorted(sdir.iterdir()):
            if not cls_dir.is_dir():
                continue
            out_dir = dst / split / cls_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for img_path in sorted(cls_dir.glob("*.jpg")):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                # prompt = bbox central (o crop já é centrado na mosca com pad)
                box = np.array([w * 0.15, h * 0.15, w * 0.85, h * 0.85])
                try:
                    predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
                    best = masks[int(np.argmax(scores))]
                    # aplica máscara: fundo -> branco
                    clean = img.copy()
                    clean[~best] = 255
                    cv2.imwrite(str(out_dir / img_path.name), clean, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    n_ok += 1
                except Exception:
                    # fallback: copia o original se SAM falhar
                    cv2.imwrite(str(out_dir / img_path.name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    n_fail += 1
        print(f"[{split}] ok={n_ok} fallback={n_fail}", flush=True)
    print(f"OK -> {dst} | total ok={n_ok} fallback={n_fail}")


if __name__ == "__main__":
    main()
