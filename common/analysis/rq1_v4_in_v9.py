"""Verifica se imagens do v4 (paper antigo, 1280x720) reaparecem (redimensionadas/reexportadas) no v9.0.5.
Correlacao normalizada de miniaturas cinza 96x54. Uso: python -I rq1_v4_in_v9.py <V4_ROOT> <V9_ROOT> <OUT_CSV>"""
import sys, pathlib, re, csv
import numpy as np
from PIL import Image
v4, v9, out = map(pathlib.Path, sys.argv[1:4])
AUG = re.compile(r"-instance|-[0-9a-f]{8}-[0-9a-f]{4}-")
def thumb(p):
    a = np.asarray(Image.open(p).convert('L').resize((96, 54), Image.BILINEAR), dtype=np.float32).ravel()
    a -= a.mean(); n = np.linalg.norm(a); return a / (n if n else 1)
def load(root, skip_aug):
    L = []
    for s in ('train', 'val', 'test'):
        for p in sorted((root/'images'/s).iterdir()):
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png') and not (skip_aug and AUG.search(p.stem)):
                L.append((s, p))
    return L
A = load(v4, False); B = load(v9, True)
TA = np.stack([thumb(p) for _, p in A]); TB = np.stack([thumb(p) for _, p in B])
S = TA @ TB.T
rows = []
for i, (s, p) in enumerate(A):
    j = int(S[i].argmax()); rows.append((s, p.name, B[j][0], B[j][1].name, round(float(S[i, j]), 4)))
with open(out, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['v4_split', 'v4_img', 'v9_split', 'v9_best_match', 'corr']); w.writerows(rows)
c = np.array([r[4] for r in rows])
print('v4 imgs', len(rows), 'v9 originais', len(B))
for t in (0.99, 0.95, 0.9, 0.8): print(f'corr>={t}:', int((c >= t).sum()))
print('mediana', float(np.median(c)), 'max', float(c.max()))
