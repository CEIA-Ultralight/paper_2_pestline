"""Perfil das imagens originais (sem aug) do DS-F2_v9.0.5 por lote e split: n de imagens,
imagens so-INS (nenhuma caixa MD-MF, ao menos uma INS) e maximo de instancias de especie por imagem.
Uso: python3 -I common/analysis/lot_profile.py <V9_ROOT> rq1_data/results/lot_profile.csv"""
import csv, os, re, sys, collections
root, out = sys.argv[1], sys.argv[2]
rows = collections.defaultdict(lambda: [0, 0, 0])  # n, ins_only, max_species
for split in ('train', 'val', 'test'):
    for f in os.listdir(os.path.join(root, 'images', split)):
        s = os.path.splitext(f)[0]
        if '-instance' in s or re.search(r'-[0-9a-f]{8}-[0-9a-f]{4}-', s):
            continue
        m = re.match(r'DS-F2_(v[\d.]+?)_\d', s)
        lot = m.group(1) if m else 'other'
        cls = collections.Counter()
        lp = os.path.join(root, 'labels', split, s + '.txt')
        if os.path.exists(lp):
            for line in open(lp):
                if line.strip():
                    cls[int(line.split()[0])] += 1
        sp = sum(cls[c] for c in range(4))
        r = rows[(lot, split)]
        r[0] += 1; r[1] += int(sp == 0 and cls[4] > 0); r[2] = max(r[2], sp)
with open(out, 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['lot', 'split', 'orig_images', 'ins_only', 'max_species_per_image'])
    for (lot, split), r in sorted(rows.items()):
        w.writerow([lot, split, *r])
