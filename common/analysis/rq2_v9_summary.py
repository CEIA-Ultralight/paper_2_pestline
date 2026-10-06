"""Tabelas-base da RQ2 v9: metricas por combinacao (detector x classificador) e oraculo, lidas de metrics.json.

Saidas em rq2_cascade/results/: v9_cascade_summary.csv, v9_cascade_aggregate.csv, v9_oracle_summary.csv.
Politica label_only (identica a product no harness). F1 macro das 4 especies. VAL escolheu best.pt (nao e held-out).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[2] / "rq2_cascade" / "results"
KEYS = ("species_macro_f1", "species_macro_recall", "species_macro_precision", "species_accuracy_all_gt",
        "species_accuracy_conditional_matched", "species_detection_recall")


def cascade_rows(root: Path, split: str) -> list[dict]:
    rows = []
    for top in sorted(root.glob(f"cascade_*_{split}")):
        for det in sorted(p for p in top.iterdir() if p.is_dir()):
            for clf in sorted(p for p in det.iterdir() if p.is_dir() and p.name.startswith("clf_")):
                f = clf / "label_only" / "metrics.json"
                if not f.exists():
                    continue
                m = json.loads(f.read_text())
                r = {"split": split, "suite": top.name, "detector": det.name, "clf": clf.name,
                     "clf_family": clf.name.removeprefix("clf_").rsplit("_s", 1)[0]}
                for k in KEYS:
                    r[f"single_{k}"], r[f"cascade_{k}"] = m["single"][k], m["cascade"][k]
                r["delta_f1"] = r["cascade_species_macro_f1"] - r["single_species_macro_f1"]
                r["images"], r["gt_objects"] = m["images"], m["gt_objects"]
                rows.append(r)
    return rows


def oracle_rows(root: Path, split: str) -> list[dict]:
    rows = []
    for top in sorted(root.glob(f"oracle_*_{split}")):
        for d in sorted(p for p in top.iterdir() if p.is_dir() and p.name.startswith("clf_")):
            f = d / "metrics.json"
            if not f.exists():
                continue
            m = json.loads(f.read_text())
            flat = {k: v for k, v in m.items() if isinstance(v, (int, float, str))}
            rows.append({"split": split, "suite": top.name, "clf": d.name, **flat})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/raid/user_marcospaulo/flydet_runs_v9")
    ap.add_argument("--out", default=str(RES))
    a = ap.parse_args(argv)
    root, out = Path(a.runs_root), Path(a.out)
    c = pd.DataFrame([r for s in ("val", "test") for r in cascade_rows(root, s)])
    c.to_csv(out / "v9_cascade_summary.csv", index=False, float_format="%.5f")
    det = c.assign(det_group=c["detector"].where(~c["detector"].str.startswith("E0"), "E0")).groupby(
        ["split", "det_group", "clf_family"], as_index=False).agg(
        n=("delta_f1", "size"), single_f1=("single_species_macro_f1", "mean"), cascade_f1=("cascade_species_macro_f1", "mean"),
        delta_f1_mean=("delta_f1", "mean"), delta_f1_std=("delta_f1", "std"),
        single_recall=("single_species_macro_recall", "mean"), cascade_recall=("cascade_species_macro_recall", "mean"),
        single_prec=("single_species_macro_precision", "mean"), cascade_prec=("cascade_species_macro_precision", "mean"))
    det.to_csv(out / "v9_cascade_aggregate.csv", index=False, float_format="%.5f")
    o = pd.DataFrame([r for s in ("val", "test") for r in oracle_rows(root, s)])
    o.to_csv(out / "v9_oracle_summary.csv", index=False, float_format="%.5f")
    print(c.shape, det.shape, o.shape)


if __name__ == "__main__":
    main()
