"""Helpers da campanha B200 (analyst). CPU only. Fonte: /raid/user_marcospaulo/flydet_runs/."""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/raid/user_marcospaulo/paper_2_pestline")
R = Path("/raid/user_marcospaulo/flydet_runs")
sys.path.insert(0, str(REPO / "common"))
import stats  # noqa: E402

SP = ["MD", "MV", "MC", "MF"]
NONSP = ["INS", "NOISE", "MAR"]
BOOT_N = 10000
BOOT_SEED = 20260930


def rel(p) -> str:
    """Caminho como aparece no cluster (rastreabilidade)."""
    return str(p)


def pct(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(100.0 * float(x), 4)


def load_json(p):
    return json.loads(Path(p).read_text())


def species_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.gt_class.isin(SP)].reset_index(drop=True)


def is_true(s):
    return s.astype(str).str.lower().isin(("true", "1"))


def harness_metrics(b: dict) -> dict:
    """Metricas de especie (MD/MV/MC/MF) de um bloco single/cascade do metrics.json do harness."""
    pc = b["per_class"]
    tp = sum(pc[s]["tp"] for s in SP)
    pred = sum(pc[s]["predicted"] for s in SP)
    sup = b["species_support"]
    p, r = tp / pred, tp / sup
    d = {
        "recall_micro": r, "precision_micro": p, "f1_micro": 2 * p * r / (p + r),
        "recall_macro": b["species_macro_recall"], "precision_macro": b["species_macro_precision"],
        "f1_macro": b["species_macro_f1"],
        "acc_all": b["species_accuracy_all_gt"], "acc_loc": b["species_accuracy_conditional_matched"],
        "det_recall": b["species_detection_recall"],
    }
    for s in SP:
        d[f"{s}_recall"], d[f"{s}_precision"], d[f"{s}_f1"] = pc[s]["recall"], pc[s]["precision"], pc[s]["f1"]
    d["n_support"], d["n_matched"], d["n_correct"] = sup, b["species_matched"], b["species_correct"]
    d["tp_micro"], d["predicted_micro"] = tp, pred
    return d


METRIC_KEYS = (["recall_micro", "precision_micro", "f1_micro", "recall_macro", "precision_macro", "f1_macro",
                "acc_all", "acc_loc", "det_recall"]
               + [f"{s}_{m}" for s in SP for m in ("recall", "precision", "f1")])
COUNT_KEYS = ["n_support", "n_matched", "n_correct", "tp_micro", "predicted_micro"]


def check_pi_vs_metrics(pi: pd.DataFrame, b: dict, branch: str, tag: str):
    """Confere que per_instance reproduz suporte/localizados/corretos/tp do metrics.json."""
    sp = species_rows(pi)
    matched = is_true(sp.matched)
    ok = matched & (sp[f"pred_class_{branch}"] == sp.gt_class)
    assert len(sp) == b["species_support"], (tag, len(sp), b["species_support"])
    assert int(matched.sum()) == b["species_matched"], tag
    assert int(ok.sum()) == b["species_correct"], tag
    assert int(ok.sum()) == sum(b["per_class"][s]["tp"] for s in SP), tag
    col = is_true(sp[f"correct_{branch}"])
    assert (col == ok).all(), tag
    return ok.to_numpy()


def mean_std(vals):
    return stats.mean_std(vals)


def agg_dicts(rows: list[dict], keys) -> tuple[dict, dict]:
    m, s = {}, {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        m[k], s[k] = (mean_std(vals) if vals else (None, None))
    return m, s


# ------------------------------------------------------------------ pareados
def mcnemar_exact(ok_a, ok_b) -> dict:
    """b = A certo e B errado; c = A errado e B certo. p exato binomial bicaudal SEMPRE (inclui b+c>=25)."""
    r = stats.mcnemar_paired(list(ok_a), list(ok_b))
    b, c = r["b"], r["c"]
    return {"b": b, "c": c, "n_discordant": b + c, "p_exact": stats._binom_two_sided_p(b, b + c),
            "p_chi2_cc": r["p_value"] if not r["exact"] else None}


def boot_delta_acc(images, ok_a, ok_b, n=BOOT_N, seed=BOOT_SEED):
    """Delta = acc(B) - acc(A) sobre GT de especie; reamostragem de IMAGENS (n=10000, seed fixa)."""
    images = np.asarray(images)
    uniq, inv = np.unique(images, return_inverse=True)
    k = len(uniq)
    n_i = np.bincount(inv, minlength=k).astype(float)
    a_i = np.bincount(inv, weights=np.asarray(ok_a, float), minlength=k)
    b_i = np.bincount(inv, weights=np.asarray(ok_b, float), minlength=k)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n, k))
    d = (b_i[idx].sum(1) - a_i[idx].sum(1)) / n_i[idx].sum(1)
    point = (b_i.sum() - a_i.sum()) / n_i.sum()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(point), float(lo), float(hi), int(k)


def holm(pvals: list[float]) -> list[float]:
    """p ajustados de Holm-Bonferroni (step-down, monotonicos)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj


def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"escrito {path.relative_to(REPO)}  {df.shape}")
