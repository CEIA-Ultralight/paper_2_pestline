"""Analises CPU complementares da RQ2 v9: confusao single x cascata, latencia, fusao tardia (aproximada), localizacao.

Fusao tardia: o harness guarda so classe e confianca top-1 do detector, entao a distribuicao do detector e
APROXIMADA (conf na classe predita, (1-conf)/6 nas demais). p_fus ~ p_clf * p_det**w. Nao e a distribuicao real.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq2_v9_analysis as A  # noqa: E402

YC, SP, RES = A.YC, A.SP, A.RES
NONSP = ("INS", "NOISE", "MAR")
ROOT = Path("/raid/user_marcospaulo/flydet_runs_v9")


def _pairs(split: str, clf_prefix: str):
    for f in sorted(ROOT.glob(f"cascade_cross_{split}/E0_s*/{clf_prefix}_s*/label_only")):
        yield f.parents[1].name, f.parent.name, f


def confusion(split: str = "test", clf_prefix: str = "clf_convnext_t_ce") -> pd.DataFrame:
    """Taxa (%) de instancias GT de especie atribuidas a cada destino; single, cascata e delta por par."""
    cols = list(SP) + ["non-species", "missed"]
    per_pair = []
    for det, clf, f in _pairs(split, clf_prefix):
        df = pd.read_csv(f / "per_instance.csv")
        df = df[df.gt_class.isin(SP)]
        row = {}
        for br in ("single", "cascade"):
            dest = np.where(~df.matched, "missed", df[f"pred_class_{br}"])
            dest = np.where(np.isin(dest, NONSP), "non-species", dest)
            for g in SP:
                sub = dest[(df.gt_class == g).to_numpy()]
                for d in cols:
                    row[(br, g, d)] = 100.0 * float(np.mean(sub == d)) if len(sub) else np.nan
        per_pair.append(row)
    out = []
    for g in SP:
        for d in cols:
            s = np.array([r[("single", g, d)] for r in per_pair]); c = np.array([r[("cascade", g, d)] for r in per_pair])
            out.append(dict(split=split, clf=clf_prefix, annotated=g, predicted=d, n_pairs=len(per_pair),
                            single=s.mean(), cascade=c.mean(), delta_mean=(c - s).mean(), delta_std=(c - s).std(ddof=1)))
    return pd.DataFrame(out)


def latency(split: str = "test") -> pd.DataFrame:
    rows = []
    for f in sorted(ROOT.glob(f"cascade_*_{split}/*/clf_*/label_only")):
        if not (f / "metrics.json").exists():
            continue
        m = json.loads((f / "metrics.json").read_text())["latency"]
        rows.append(dict(split=split, detector=f.parents[1].name, clf=f.parent.name.rsplit("_s", 1)[0],
                         det_ms=1e3 * m["detector"]["mean_s"], clf_ms=1e3 * m["classifier_extra"]["mean_s"],
                         e2e_ms=1e3 * m["end_to_end"]["mean_s"], e2e_p95_ms=1e3 * m["end_to_end"]["p95_s"]))
    df = pd.DataFrame(rows)
    df["detector"] = df.detector.str.replace(r"_s\d+$", "", regex=True)
    return df.groupby(["split", "detector", "clf"], as_index=False).agg(
        n_runs=("e2e_ms", "size"), det_ms=("det_ms", "mean"), clf_ms=("clf_ms", "mean"),
        e2e_ms=("e2e_ms", "mean"), e2e_ms_std=("e2e_ms", "std"), e2e_p95_ms=("e2e_p95_ms", "mean"))


def localization(split: str = "test") -> pd.DataFrame:
    rows = []
    for f in sorted(ROOT.glob(f"cascade_*_{split}/*/clf_convnext_t_ce_s42/label_only")):
        m = json.loads((f / "metrics.json").read_text())["single"]
        rows.append(dict(split=split, detector=f.parents[1].name, species_detection_recall=100 * m["species_detection_recall"],
                         single_species_macro_f1=100 * m["species_macro_f1"]))
    return pd.DataFrame(rows)


def fused_counts(rec: dict, w: float) -> np.ndarray:
    rec = json.loads(json.dumps(rec))
    for p in rec["proposals"]:
        probs = p.get("probabilities")
        if not probs or any(q is None for q in probs):
            continue
        conf, y = float(p["yolo_conf"]), YC.index(p["yolo_class"])
        det = np.full(len(YC), (1 - conf) / (len(YC) - 1)); det[y] = conf
        f = np.asarray(probs) * np.power(np.maximum(det, 1e-6), w)
        p["cascade_class"] = YC[int(f.argmax())]
    return A.image_counts(rec)


def late_fusion(split: str = "test", ws=(0.0, 0.25, 0.5, 1.0, 2.0), clf_prefix: str = "clf_convnext_t_ce") -> pd.DataFrame:
    rows = []
    for det, clf, f in _pairs(split, clf_prefix):
        recs = [json.loads(l) for l in (f / "per_image_preds.jsonl").read_text().splitlines() if l.strip()]
        base = A.f1_macro(np.stack([A.image_counts(r) for r in recs]).sum(0), 3, 4)  # cascata sem fusao
        single = A.f1_macro(np.stack([A.image_counts(r) for r in recs]).sum(0), 1, 2)
        for w in ws:
            tot = np.stack([fused_counts(r, w) for r in recs]).sum(0)
            rows.append(dict(split=split, detector=det, clf=clf, w=w, f1_single=single, f1_cascade=base,
                             f1_fused=A.f1_macro(tot, 3, 4)))
    df = pd.DataFrame(rows)
    return df.groupby(["split", "w"], as_index=False).agg(n_pairs=("clf", "size"), f1_single=("f1_single", "mean"),
                                                         f1_cascade=("f1_cascade", "mean"), f1_fused=("f1_fused", "mean"))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(RES)); a = ap.parse_args(argv)
    out = Path(a.out)
    for name, dfs in {
        "v9_confusion_single_vs_cascade": [confusion(s) for s in ("test", "val")],
        "v9_latency": [latency("test")],
        "v9_localization": [localization("test"), localization("val")],
        "v9_late_fusion_approx": [late_fusion(s) for s in ("test", "val")],
    }.items():
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(out / f"{name}.csv", index=False, float_format="%.4f"); print(name, df.shape)


if __name__ == "__main__":
    main()
