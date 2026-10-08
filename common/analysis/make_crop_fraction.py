"""Cria subconjuntos estratificados do treino de crops (ImageFolder) via hardlinks (o treino proibe symlinks), para a curva de aprendizado.

val/test sao linkados inteiros. Seleção determinística por classe (seed fixa). Hardlinks no mesmo filesystem: nao copia nem altera os crops.
Uso: make_crop_fraction.py --src <crops> --dst <saida> --frac 0.25 [--seed 0]
"""
import argparse, os, random
from pathlib import Path


def build(src: Path, dst: Path, frac: float, seed: int = 0) -> dict:
    counts = {}
    dst.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        if (src / split).is_dir():
            for cdir in sorted(p for p in (src / split).iterdir() if p.is_dir()):
                out = dst / split / cdir.name
                out.mkdir(parents=True, exist_ok=True)
                for f in cdir.iterdir():
                    if not (out / f.name).exists():
                        os.link(f, out / f.name)
    for cdir in sorted(p for p in (src / "train").iterdir() if p.is_dir()):
        files = sorted(f.name for f in cdir.iterdir())
        rng = random.Random(f"{seed}-{cdir.name}")
        keep = sorted(rng.sample(files, max(1, round(len(files) * frac))))
        out = dst / "train" / cdir.name
        out.mkdir(parents=True, exist_ok=True)
        for n in keep:
            if not (out / n).exists():
                os.link(cdir / n, out / n)
        counts[cdir.name] = len(keep)
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--frac", type=float, required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(build(Path(a.src), Path(a.dst), a.frac, a.seed))
