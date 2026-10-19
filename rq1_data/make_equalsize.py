"""Controle de mesmo tamanho da escada RQ1: subamostra aleatoria do TRAIN do DS-F2_v9.0.5 completo (L3) com N imagens.
VAL e TEST sao os do v9.0.5. Hardlinks (o treino proibe symlinks). Uma raiz por (N, seed): <OUT>/EQ<N>_s<seed>.
Uso: python make_equalsize.py <V9_ROOT> <OUT_ROOT> <N> [<N> ...]   (seeds 0,1,2)"""
import sys, os, random, pathlib
v9, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); sizes = [int(x) for x in sys.argv[3:]]
NAMES = ['MD', 'MV', 'MC', 'MF', 'INS', 'NOISE', 'MAR']
EXT = ('.jpg', '.jpeg', '.png')
def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): dst.unlink()
    os.link(src, dst)
train = sorted(p for p in (v9/'images'/'train').iterdir() if p.suffix.lower() in EXT)
for n in sizes:
    for s in (0, 1, 2):
        root = out/f'EQ{n}_s{s}'
        chosen = random.Random(1000 * n + s).sample(train, n)
        for split, items in (('train', chosen), ('val', sorted((v9/'images'/'val').iterdir())), ('test', sorted((v9/'images'/'test').iterdir()))):
            for p in items:
                if p.suffix.lower() not in EXT: continue
                lb = v9/'labels'/split/(p.stem + '.txt'); assert lb.exists(), lb
                link(p, root/'images'/split/p.name); link(lb, root/'labels'/split/lb.name)
        (root/'data.yaml').write_text(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" + ''.join(f"- {x}\n" for x in NAMES))
        print(root, n)
