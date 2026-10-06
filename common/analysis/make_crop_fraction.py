"""Cria subconjuntos estratificados do treino de crops (ImageFolder) via symlinks, para a curva de aprendizado.

val/test sao linkados inteiros. Seleção determinística por classe (seed fixa). Nunca copia nem altera os crops.
Uso: make_crop_fraction.py --src <crops> --dst <saida> --frac 0.25 [--seed 0]
"""
import argparse, os, random
from pathlib import Path


def build(src: Path, dst: Path, frac: float, seed: int = 0) -> dict:
    counts = {}
    dst.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        if (src / split).is_dir() and not (dst / split).exists():
            (dst / split).symlink_to((src / split).resolve())
    for cdir in sorted(p for p in (src / "train").iterdir() if p.is_dir()):
        files = sorted(f.name for f in cdir.iterdir())
        rng = random.Random(f"{seed}-{cdir.name}")
        keep = sorted(rng.sample(files, max(1, round(len(files) * frac))))
        out = dst / "train" / cdir.name
        out.mkdir(parents=True, exist_ok=True)
        for n in keep:
            link = out / n
            if not link.is_symlink():
                link.symlink_to(os.path.realpath(cdir / n))
        counts[cdir.name] = len(keep)
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--frac", type=float, required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(build(Path(a.src), Path(a.dst), a.frac, a.seed))
