"""Verificação de vazamento treino -> VAL/TEST no DS-F2 (CPU).

Para cada instância de VAL/TEST, acha o crop de treino mais parecido (correlação
normalizada em 32x32 cinza, com e sem espelhamento) e para cada imagem de VAL/TEST
a miniatura de treino mais parecida. Separa imagens de copy-paste (nome com
``-instance-``) das originais. Alta similaridade = possível vazamento (suspeito, não prova).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CROP = 32
THUMB = 48


def _norm(v: np.ndarray) -> np.ndarray:
    v = v.reshape(len(v), -1).astype(np.float32)
    v -= v.mean(1, keepdims=True)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-6)


def read_labels(path: Path) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 5))
    return np.loadtxt(path, ndmin=2)[:, :5]


def split_feats(root: Path, split: str):
    crops, meta, thumbs, tmeta = [], [], [], []
    for img_path in sorted((root / "images" / split).iterdir()):
        im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        h, w = im.shape
        paste = "-instance-" in img_path.name
        thumbs.append(cv2.resize(im, (THUMB, THUMB), interpolation=cv2.INTER_AREA))
        tmeta.append((img_path.name, paste))
        for k, (c, cx, cy, bw, bh) in enumerate(read_labels(root / "labels" / split / (img_path.stem + ".txt"))):
            x0, x1 = int(round((cx - bw / 2) * w)), int(round((cx + bw / 2) * w))
            y0, y1 = int(round((cy - bh / 2) * h)), int(round((cy + bh / 2) * h))
            x0, y0 = max(x0, 0), max(y0, 0)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            crops.append(cv2.resize(im[y0:y1, x0:x1], (CROP, CROP), interpolation=cv2.INTER_AREA))
            meta.append((img_path.name, k, int(c), paste))
    return (np.array(crops), pd.DataFrame(meta, columns=["image", "k", "cls", "paste"]),
            np.array(thumbs), pd.DataFrame(tmeta, columns=["image", "paste"]))


def nearest(q: np.ndarray, ref: np.ndarray, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    q, ref = _norm(q), _norm(ref)
    best = np.full(len(q), -1.0, dtype=np.float32)
    arg = np.zeros(len(q), dtype=int)
    for i in range(0, len(ref), 8192):
        s = q @ ref[i:i + 8192].T
        j = s.argmax(1)
        m = s[np.arange(len(q)), j]
        upd = m > best
        best[upd], arg[upd] = m[upd], j[upd] + i
    return best, arg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/raid/user_marcospaulo/datasets/DS-F2_v9.0.5")
    ap.add_argument("--out", default="rq2_cascade/results/v9_leakage_check.csv")
    a = ap.parse_args(argv)
    root = Path(a.root)
    tr_c, tr_m, tr_t, tr_tm = split_feats(root, "train")
    tr_c_all = np.concatenate([tr_c, tr_c[:, :, ::-1]])
    tr_m_all = pd.concat([tr_m, tr_m], ignore_index=True)
    tr_t_all = np.concatenate([tr_t, tr_t[:, :, ::-1]])
    tr_tm_all = pd.concat([tr_tm, tr_tm], ignore_index=True)
    rows = []
    for split in ("val", "test"):
        c, m, t, tm = split_feats(root, split)
        for level, q, ref, qm, rm in (("crop", c, tr_c_all, m, tr_m_all), ("image", t, tr_t_all, tm, tr_tm_all)):
            best, arg = nearest(q, ref)
            hit_paste = rm["paste"].to_numpy()[arg]
            for thr in (0.90, 0.95, 0.98):
                sel = best >= thr
                rows.append(dict(split=split, level=level, thr=thr, n=len(best), n_ge=int(sel.sum()),
                                 frac_ge=round(float(sel.mean()), 4),
                                 n_ge_nearest_is_paste=int((sel & hit_paste).sum()),
                                 median_sim=round(float(np.median(best)), 4),
                                 max_sim=round(float(best.max()), 4)))
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
