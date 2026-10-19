"""Composicao do treino do DS-F2_v9.0.5 por lote e tipo (original / aug de imagem / copy-paste).
Uso: python3 -I common/analysis/ladder_composition.py <V9_ROOT> rq1_data/results/ladder_composition.csv"""
import csv, os, re, sys, collections
root, out = sys.argv[1], sys.argv[2]
C = collections.Counter()
for f in os.listdir(os.path.join(root, 'images', 'train')):
    s = os.path.splitext(f)[0]
    m = re.match(r'DS-F2_(v[\d.]+?)_\d', s)
    lot = m.group(1) if m else 'other'
    k = 'copypaste' if '-instance' in s else ('img_aug' if re.search(r'-[0-9a-f]{8}-[0-9a-f]{4}-', s) else 'orig')
    C[(lot, k)] += 1
lots = sorted({l for l, _ in C}, key=lambda x: (x == 'other', x))
with open(out, 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['lot', 'orig', 'img_aug', 'real_total', 'copypaste'])
    for l in lots:
        o, a, c = C[(l, 'orig')], C[(l, 'img_aug')], C[(l, 'copypaste')]
        w.writerow([l, o, a, o + a, c])
