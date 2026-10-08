"""Curva de aprendizado (25/50/100% dos crops, ConvNeXt-T seed 42) e oraculo BG (3 seeds).

Le metrics.json de oracle_lc*/oracle_bg*/oracle_crops* em RUNS_ROOT e escreve
rq2_cascade/results/v9_learning_curve.csv e v9_oracle_bg.csv. Sem GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RUNS = Path("/raid/user_marcospaulo/flydet_runs_v9")
RES = Path(__file__).resolve().parents[2] / "rq2_cascade" / "results"


def _m(d: Path) -> dict:
    return json.loads(d.read_text())


def main() -> None:
    lc, bg = [], []
    for split in ("val", "test"):
        for pct, d in ((25, f"oracle_lc25_{split}/clf_convnext_t_lc25_s42"),
                       (50, f"oracle_lc50_{split}/clf_convnext_t_lc50_s42"),
                       (100, f"oracle_crops_{split}/clf_convnext_t_ce_s42")):
            m = _m(RUNS / d / "metrics.json")
            pc = m["per_class"]
            row = dict(split=split, pct_crops=pct, n=m["species_support"], accuracy=m["species_accuracy"],
                       macro_f1=m["species_macro_f1"], macro_recall=m["species_macro_recall"], seed=42)
            lc.append(row)
        for s in (42, 84, 126):
            m = _m(RUNS / f"oracle_bg_{split}/clf_convnext_t_bg_s{s}/metrics.json")
            bg.append(dict(split=split, seed=s, accuracy=m["species_accuracy"], macro_f1=m["species_macro_f1"]))
    pd.DataFrame(lc).to_csv(RES / "v9_learning_curve.csv", index=False)
    b = pd.DataFrame(bg)
    b.to_csv(RES / "v9_oracle_bg.csv", index=False)
    print(pd.DataFrame(lc).to_string()); print(b.groupby("split")[["accuracy", "macro_f1"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
