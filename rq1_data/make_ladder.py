"""Monta os subconjuntos da escada RQ1 por hardlink (o treino proibe symlinks) a partir do DS-F2_v9.0.5.
VAL e TEST sao os do v9.0.5 em todos os bracos; so o TRAIN muda.
  L0     = lote v5.1                      (originais + aug de imagem; sem '-instance')
  L1     = L0 + v6.0 + v7
  L2     = L1 + v8 + v9                    (sem copy-paste)
  L2ORG  = L2 + DS-ORG (train do v8.1.1, mesma ontologia)
  (L3 = v9.0.5 completo = L2 + '-instance'; ja treinado na RQ2 como E0)
Uso: python make_ladder.py <V9_ROOT> <V8_ROOT> <OUT_ROOT>"""
import sys, re, os, pathlib, csv, collections
v9, v8, out = map(pathlib.Path, sys.argv[1:4])
LOTS = {'L0': ['v5.1'], 'L1': ['v5.1', 'v6.0', 'v7'], 'L2': ['v5.1', 'v6.0', 'v7', 'v8', 'v9']}
NAMES = ['MD', 'MV', 'MC', 'MF', 'INS', 'NOISE', 'MAR']
def lot_of(stem):
    m = re.match(r'DS-F2_(v[\d.]+?)_\d', stem); return m.group(1) if m else None
def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): dst.unlink()
    os.link(src, dst)
def put(root, split, imgdir, labdir, p, rows, arm):
    lb = labdir/(p.stem + '.txt')
    assert lb.exists(), lb
    link(p, root/'images'/split/p.name); link(lb, root/'labels'/split/lb.name)
    rows.append((arm, split, p.name, 'inst' if '-instance' in p.stem else ('img' if re.search(r'-[0-9a-f]{8}-[0-9a-f]{4}-', p.stem) else 'orig')))
manifest = []
for arm in ('L0', 'L1', 'L2', 'L2ORG'):
    root = out/arm; lots = LOTS['L2' if arm == 'L2ORG' else arm]
    for split in ('train', 'val', 'test'):
        for p in sorted((v9/'images'/split).iterdir()):
            if p.suffix.lower() not in ('.jpg', '.jpeg', '.png'): continue
            if split == 'train' and ('-instance' in p.stem or lot_of(p.stem) not in lots): continue
            put(root, split, None, v9/'labels'/split, p, manifest, arm)
    if arm == 'L2ORG':
        for p in sorted((v8/'images'/'train').iterdir()):
            if p.name.startswith('DS-ORG') and p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                put(root, 'train', None, v8/'labels'/'train', p, manifest, arm)
    (root/'data.yaml').write_text(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" + ''.join(f"- {n}\n" for n in NAMES))
with open(out/'manifest.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['arm', 'split', 'file', 'aug']); w.writerows(manifest)
C = collections.Counter((a, s) for a, s, _, _ in manifest)
for k, v in sorted(C.items()): print(k, v)
