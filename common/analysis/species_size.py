"""Tamanho (sqrt(w*h) em px, 1920x1080) das caixas por grupo de classe no DS-F2_v9.0.5, imagens originais (sem aug/copy-paste).
Uso: python3 -I common/analysis/species_size.py <V9_ROOT> rq1_data/results/species_size.csv"""
import csv, math, os, re, statistics as st, sys
root, out = sys.argv[1:3]
AUG = re.compile(r'-[0-9a-f]{8}-[0-9a-f]{4}-')
rows = []
for split in ('train', 'val', 'test'):
    for f in os.listdir(os.path.join(root, 'labels', split)):
        if '-instance' in f or AUG.search(f):
            continue
        for line in open(os.path.join(root, 'labels', split, f)):
            p = line.split()
            if p:
                c, w, h = int(p[0]), float(p[3]) * 1920, float(p[4]) * 1080
                rows.append((split, 'species' if c <= 3 else 'all_nonspecies', math.sqrt(w * h)))
with open(out, 'w', newline='') as fh:
    wr = csv.writer(fh); wr.writerow(['split', 'group', 'n', 'mean_px', 'median_px', 'frac_below_32'])
    for split in ('train', 'val', 'test'):
        for g in ('species', 'all'):
            v = [s for sp, gg, s in rows if sp == split and (g == 'all' or gg == g)]
            wr.writerow([split, g, len(v), round(st.mean(v), 1), round(st.median(v), 1), round(sum(x < 32 for x in v) / len(v), 3)])
