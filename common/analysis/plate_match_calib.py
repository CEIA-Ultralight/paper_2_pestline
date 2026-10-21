"""Calibracao do plate_match.py.
(1) Positivos sinteticos: cada original de TEST contra uma "refoto" dela mesma (perspectiva, escala, giro de 180 graus
    em metade dos casos, brilho/contraste), com as caixas transformadas pela mesma homografia.
(2) Negativos/positivos reais: TRAIN contra TRAIN (pares distintos), melhor par de cada imagem.
Uso: python -I common/analysis/plate_match_calib.py <V9_ROOT> <OUT_CSV> [workers]"""
import csv, os, re, sys, random
import multiprocessing as mp
import cv2
import numpy as np
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()

ROOT, OUT = sys.argv[1:3]
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
AUG = re.compile(r'-[0-9a-f]{8}-[0-9a-f]{4}-')
SIFT_N = 3000


def originals(split):
    d = os.path.join(ROOT, 'images', split)
    return [(split, f) for f in sorted(os.listdir(d)) if '-instance' not in f and not AUG.search(f)]


def load(item):
    split, f = item
    im = np.array(Image.open(os.path.join(ROOT, 'images', split, f)).convert('L'))
    boxes = []
    for line in open(os.path.join(ROOT, 'labels', split, os.path.splitext(f)[0] + '.txt')):
        p = line.split()
        if p and int(p[0]) <= 5:
            boxes.append([float(v) for v in p[1:5]])
    return im, boxes


def feats(im, corners):
    mask = np.zeros_like(im)
    for c in corners:
        x0, y0 = np.floor(c.min(0)).astype(int) - 8
        x1, y1 = np.ceil(c.max(0)).astype(int) + 8
        mask[max(0, y0):y1, max(0, x0):x1] = 255
    kp, des = cv2.SIFT_create(nfeatures=SIFT_N).detectAndCompute(im, mask)
    return (np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)), des


def box_corners(boxes, w, h):
    out = []
    for cx, cy, bw, bh in boxes:
        x0, y0, x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
        out.append(np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
    return out


def inliers(a, b):
    (pa, da), (pb, db) = a, b
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return 0
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(good) < 8:
        return 0
    H, inl = cv2.findHomography(pa[[g.queryIdx for g in good]].reshape(-1, 1, 2),
                                pb[[g.trainIdx for g in good]].reshape(-1, 1, 2), cv2.RANSAC, 5.0)
    return int(inl.sum()) if inl is not None else 0


def synthetic(item):
    rnd = random.Random(hash(item[1]) & 0xffff)
    im, boxes = load(item)
    h, w = im.shape
    cs = box_corners(boxes, w, h)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    j = 0.06
    dst = np.float32([[x + rnd.uniform(-j, j) * w, y + rnd.uniform(-j, j) * h] for x, y in src])
    dst = (dst - [w / 2, h / 2]) * rnd.uniform(0.85, 1.0) + [w / 2, h / 2]
    if rnd.random() < 0.5:
        dst = np.float32([w, h]) - dst
    H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    warped = cv2.warpPerspective(im, H, (w, h), borderValue=200)
    warped = np.clip(warped.astype(np.float32) * rnd.uniform(0.8, 1.2) + rnd.uniform(-20, 20), 0, 255).astype(np.uint8)
    wcs = [cv2.perspectiveTransform(c.reshape(-1, 1, 2), H).reshape(-1, 2) for c in cs]
    return ('synthetic', item[1], 'self', inliers(feats(im, cs), feats(warped, wcs)))


TRAIN = {}


def train_feat(item):
    im, boxes = load(item)
    h, w = im.shape
    return item, feats(im, box_corners(boxes, w, h))


def train_row(item):
    best = max(((inliers(TRAIN[item], TRAIN[o]), o) for o in TRAIN if o != item), key=lambda t: t[0])
    return ('train_vs_train', item[1], best[1][1], best[0])


if __name__ == '__main__':
    ctx = mp.get_context('fork')
    with ctx.Pool(WORKERS) as pool:
        rows = pool.map(synthetic, originals('test'))
        for item, fd in pool.map(train_feat, originals('train'), chunksize=4):
            TRAIN[item] = fd
    with ctx.Pool(WORKERS) as pool:
        rows += pool.map(train_row, list(TRAIN), chunksize=2)
    with open(OUT, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['kind', 'file', 'best_ref', 'inliers']); w.writerows(rows)
