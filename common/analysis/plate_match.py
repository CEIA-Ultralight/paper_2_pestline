"""Checagem de vazamento por placa: casa cada imagem original de VAL/TEST com todas as originais de TRAIN
(e TEST com VAL) por SIFT restrito as caixas de insetos anotadas (classes 0-5; MAR, logo e grade ficam de fora)
+ homografia RANSAC. Fotos da mesma placa compartilham a disposicao dos insetos -> muitos inliers.
Saida: um CSV por par avaliado com inliers > 0 e um resumo do melhor par de cada imagem de avaliacao.
Uso: python -I common/analysis/plate_match.py <V9_ROOT> <OUT_PAIRS_CSV> <OUT_BEST_CSV> [workers]"""
import csv, os, re, sys
import multiprocessing as mp
import cv2
import numpy as np
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()

ROOT, OUT_PAIRS, OUT_BEST = sys.argv[1:4]
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
AUG = re.compile(r'-[0-9a-f]{8}-[0-9a-f]{4}-')


def originals(split):
    d = os.path.join(ROOT, 'images', split)
    return [(split, f) for f in sorted(os.listdir(d)) if '-instance' not in f and not AUG.search(f)]


def features(item):
    split, f = item
    im = np.array(Image.open(os.path.join(ROOT, 'images', split, f)).convert('L'))
    h, w = im.shape
    mask = np.zeros_like(im)
    for line in open(os.path.join(ROOT, 'labels', split, os.path.splitext(f)[0] + '.txt')):
        p = line.split()
        if not p or int(p[0]) > 5:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        x0, y0 = int((cx - bw / 2) * w) - 8, int((cy - bh / 2) * h) - 8
        x1, y1 = int((cx + bw / 2) * w) + 8, int((cy + bh / 2) * h) + 8
        mask[max(0, y0):y1, max(0, x0):x1] = 255
    kp, des = cv2.SIFT_create(nfeatures=3000).detectAndCompute(im, mask)
    pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
    return item, pts, des


def inliers(a, b):
    (_, pa, da), (_, pb, db) = a, b
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return 0
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(good) < 8:
        return 0
    src = pa[[g.queryIdx for g in good]].reshape(-1, 1, 2)
    dst = pb[[g.trainIdx for g in good]].reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(inl.sum()) if inl is not None else 0


FEATS = {}


def row(key):
    ev = FEATS[key]
    out = []
    for other_key, other in FEATS.items():
        # TEST contra TRAIN e VAL; VAL contra TRAIN
        if other_key[0] == 'train' or (key[0] == 'test' and other_key[0] == 'val'):
            out.append((key, other_key, inliers(ev, other)))
    return out


if __name__ == '__main__':
    items = originals('train') + originals('val') + originals('test')
    with mp.get_context("fork").Pool(WORKERS) as pool:
        for item, pts, des in pool.map(features, items, chunksize=4):
            FEATS[item] = (item, pts, des)
    evals = [k for k in FEATS if k[0] in ('val', 'test')]
    with mp.get_context("fork").Pool(WORKERS) as pool:
        results = pool.map(row, evals, chunksize=2)
    with open(OUT_PAIRS, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['eval_split', 'eval_file', 'ref_split', 'ref_file', 'inliers'])
        for rs in results:
            for (es, ef), (rs_, rf), n in rs:
                if n > 0:
                    w.writerow([es, ef, rs_, rf, n])
    with open(OUT_BEST, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['eval_split', 'eval_file', 'best_ref_split', 'best_ref_file', 'best_inliers', 'second_inliers'])
        for rs in results:
            rs = sorted(rs, key=lambda r: -r[2])
            (es, ef), (bs, bf), n = rs[0]
            w.writerow([es, ef, bs, bf, n, rs[1][2] if len(rs) > 1 else 0])
