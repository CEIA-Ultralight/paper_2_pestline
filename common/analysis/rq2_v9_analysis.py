"""Analises CPU da RQ2 (campanha v9) a partir das saidas do harness (per_instance.csv, per_image_preds.jsonl).

Gera CSVs em rq2_cascade/results/ (prefixo v9_). Nada de GPU. Politica analisada: label_only
(identica a product no harness atual). Metrica: F1 macro das 4 especies (MD MV MC MF).

  bootstrap  Delta F1 (cascata - detector) por bootstrap pareado de IMAGENS (IC percentil, nao BCa)
             + teste de permutacao pareado (troca aleatoria dos ramos por imagem, equivale ao sign-flip).
             Uma reamostragem compartilhada entre todas as combinacoes de uma familia (detector x seed clf).
  mcnemar    McNemar exato single vs cascata nas instancias de especie (por combinacao).
  oracle     acuracia do classificador em crops GT por faixa de tamanho; confianca dos erros;
             concordancia entre classificadores; ensemble por voto; acuracia de familia MD/MV/MC;
             calibracao (ECE) e risco x cobertura.
  hybrid     cascata com abstencao: usa o classificador so se max-softmax >= tau, senao mantem o rotulo do detector.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "rq2_cascade"))
sys.path.insert(0, str(REPO / "common"))
import eval_cascade_matrix as M  # noqa: E402
from stats import mcnemar_paired  # noqa: E402

YC = list(M.YOLO_CLASSES)
SP = list(M.SPECIES)
SP_IDX = [YC.index(c) for c in SP]
FAMILY = ("MD", "MV", "MC")
RES = REPO / "rq2_cascade" / "results"


# ----------------------------------------------------------------------------- contagens por imagem
def _preds(rec: dict, branch: str, tau: float | None = None) -> list[dict]:
    out = []
    for p in rec["proposals"]:
        if branch == "single":
            name = p["yolo_class"]
        else:
            name = p["cascade_class"]
            if tau is not None:  # hibrido: classificador so se confiante
                probs = [q for q in (p.get("probabilities") or []) if q is not None]
                if not probs or max(probs) < tau:
                    name = p["yolo_class"]
        out.append({"class": YC.index(name), "yolo_confidence": p["yolo_conf"], "box": p["box"]})
    return out


def image_counts(rec: dict, tau: float | None = None) -> np.ndarray:
    """Vetor por imagem: [support, tp_s, pred_s, tp_c, pred_c] x 4 especies -> shape (5,4)."""
    gt = [{"class": YC.index(g["class"]), "box": g["box"]} for g in rec["gt"]]
    single, casc = _preds(rec, "single"), _preds(rec, "cascade", tau)
    rows = M.rows_for_image(rec["image"], gt, single, casc, "label_only")
    fps, fpc = M.false_positives(gt, single), M.false_positives(gt, casc)
    a = np.zeros((5, 4))
    for j, c in enumerate(SP):
        a[0, j] = sum(r["gt_class"] == c for r in rows)
        a[1, j] = sum(r["gt_class"] == c and r["correct_single"] for r in rows)
        a[2, j] = sum(r["matched"] and r["pred_class_single"] == c for r in rows) + fps[c]
        a[3, j] = sum(r["gt_class"] == c and r["correct_cascade"] for r in rows)
        a[4, j] = sum(r["matched"] and r["pred_class_cascade"] == c for r in rows) + fpc[c]
    return a


def f1_macro(total: np.ndarray, tp_row: int, pred_row: int) -> np.ndarray:
    """total: (..., 5, 4). Retorna F1 macro (...)."""
    sup, tp, pr = total[..., 0, :], total[..., tp_row, :], total[..., pred_row, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(pr > 0, tp / pr, 0.0)
        r = np.where(sup > 0, tp / sup, 0.0)
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return f.mean(-1)


def delta_f1(total: np.ndarray) -> np.ndarray:
    return f1_macro(total, 3, 4) - f1_macro(total, 1, 2)


def load_counts(folder: Path, tau: float | None = None) -> np.ndarray:
    recs = [json.loads(line) for line in (folder / "per_image_preds.jsonl").read_text().splitlines() if line.strip()]
    recs.sort(key=lambda r: r["image"])
    return np.stack([image_counts(r, tau) for r in recs])  # (n_img,5,4)


def paired_bootstrap(counts: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    """counts: (n_comb, n_img, 5, 4). Media das combinacoes; reamostra imagens (indices compartilhados)."""
    rng = np.random.default_rng(seed)
    n_comb, n_img = counts.shape[:2]
    obs_each = delta_f1(counts.sum(1))
    obs = obs_each.mean()
    boots = np.empty(n_boot)
    perms = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_img, n_img)
        boots[b] = delta_f1(counts[:, idx].sum(1)).mean()
        swap = rng.random(n_img) < 0.5  # troca ramos single<->cascata por imagem
        c = counts.copy()
        c[:, swap, 1:3], c[:, swap, 3:5] = counts[:, swap, 3:5], counts[:, swap, 1:3]
        perms[b] = delta_f1(c.sum(1)).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = (1 + np.sum(np.abs(perms) >= abs(obs))) / (n_boot + 1)
    return {"delta_f1": obs, "ci_low": lo, "ci_high": hi, "p_perm": p,
            "delta_min": obs_each.min(), "delta_max": obs_each.max(), "n_comb": n_comb, "n_img": n_img}


# ----------------------------------------------------------------------------- descoberta de pastas
def cascade_dirs(root: Path, split: str) -> list[tuple[str, str, str, Path]]:
    """(familia_clf, detector, clf, pasta com per_image_preds.jsonl) em todas as cascade_*_{split}."""
    out = []
    for top in sorted(root.glob(f"cascade_*_{split}")):
        for det in sorted(p for p in top.iterdir() if p.is_dir()):
            for clf in sorted(p for p in det.iterdir() if p.is_dir() and p.name.startswith("clf_")):
                f = clf / "label_only"
                if (f / "per_image_preds.jsonl").exists():
                    fam = clf.name.removeprefix("clf_").rsplit("_s", 1)[0]
                    out.append((fam, det.name, clf.name, f))
    return out


def run_bootstrap(root: Path, split: str, n_boot: int) -> pd.DataFrame:
    groups: dict[tuple[str, str], list[Path]] = {}
    for fam, det, _clf, f in cascade_dirs(root, split):
        groups.setdefault((fam, det.rsplit("_s", 1)[0] if det.startswith("E0") else det), []).append(f)
    rows = []
    for (fam, det), folders in sorted(groups.items()):
        cnt = np.stack([load_counts(f) for f in folders])
        r = paired_bootstrap(cnt, n_boot)
        rows.append({"split": split, "clf": fam, "detector": det, "policy": "label_only", **r})
    return pd.DataFrame(rows)


def run_mcnemar(root: Path, split: str) -> pd.DataFrame:
    rows = []
    for fam, det, clf, f in cascade_dirs(root, split):
        df = pd.read_csv(f / "per_instance.csv")
        df = df[df["gt_class"].isin(SP)]
        m = mcnemar_paired(df["correct_single"].astype(bool).tolist(), df["correct_cascade"].astype(bool).tolist())
        rows.append({"split": split, "clf": fam, "detector": det, "run": clf, "n": len(df),
                     **{k: m[k] for k in ("b", "c", "p_value", "exact")}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- oraculo
def oracle_frames(root: Path, split: str) -> dict[str, pd.DataFrame]:
    out = {}
    for top in sorted(root.glob(f"oracle_*_{split}")):
        if top.name.startswith("oracle_bg") or top.name.startswith("oracle_lc"):
            continue
        for d in sorted(top.iterdir()):
            f = d / "per_instance.csv"
            if f.exists() and d.name.startswith("clf_"):
                out[d.name] = pd.read_csv(f).sort_values(["image", "gt_index"]).reset_index(drop=True)
    return out


def ece(conf: np.ndarray, ok: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(tot)


def run_oracle(root: Path, split: str) -> dict[str, pd.DataFrame]:
    fr = oracle_frames(root, split)
    if not fr:
        return {}
    names = sorted(fr)
    base = fr[names[0]]
    for n in names:
        assert (fr[n][["image", "gt_index"]].values == base[["image", "gt_index"]].values).all(), n
    size = base["gt_size"].to_numpy()
    gt = base["gt_class"].to_numpy()
    edges = [0, 24, 32, 48, 64, math.inf]
    labels = ["<24", "24-32", "32-48", "48-64", ">=64"]
    ok = np.stack([fr[n]["correct"].astype(bool).to_numpy() for n in names])  # (K,N)
    conf = np.stack([fr[n]["score"].to_numpy() for n in names])
    pred = np.stack([fr[n]["pred_class"].to_numpy() for n in names])
    res = {}
    # acuracia por tamanho
    rows = []
    for lab, lo, hi in zip(labels, edges[:-1], edges[1:]):
        m = (size >= lo) & (size < hi)
        per = ok[:, m].mean(1) if m.any() else np.array([np.nan])
        rows.append({"split": split, "size_px": lab, "n": int(m.sum()), "acc_mean": per.mean(), "acc_min": per.min(), "acc_max": per.max(),
                     **{f"acc_{c}": float(ok[:, m & (gt == c)].mean()) if (m & (gt == c)).any() else np.nan for c in SP}})
    res["oracle_by_size"] = pd.DataFrame(rows)
    # confianca dos erros / calibracao
    allok, allc = ok.ravel(), conf.ravel()
    rows = [{"split": split, "n_pred": allok.size, "acc": allok.mean(), "conf_mean_correct": allc[allok].mean(),
             "conf_mean_wrong": allc[~allok].mean(), "wrong_conf_ge_0.8": float((allc[~allok] >= 0.8).mean()),
             "wrong_conf_lt_0.5": float((allc[~allok] < 0.5).mean()), "ece": ece(allc, allok.astype(float))}]
    res["oracle_confidence"] = pd.DataFrame(rows)
    # concordancia
    ncorrect = ok.sum(0)
    K = len(names)
    cnt = Counter(ncorrect.tolist())
    mode = np.array([Counter(pred[:, i]).most_common(1)[0][0] for i in range(pred.shape[1])])
    vote_acc = float((mode == gt).mean())
    pair = [(pred[i] == pred[j]).mean() for i, j in itertools.combinations(range(K), 2)]
    res["oracle_agreement"] = pd.DataFrame([{
        "split": split, "n_clf": K, "n_inst": len(gt), "all_correct": cnt.get(K, 0) / len(gt), "all_wrong": cnt.get(0, 0) / len(gt),
        "mixed": 1 - (cnt.get(K, 0) + cnt.get(0, 0)) / len(gt), "vote_acc": vote_acc,
        "mean_single_acc": float(ok.mean()), "mean_pairwise_agreement": float(np.mean(pair)),
        "all_wrong_gt_" + "_".join(SP): " ".join(f"{c}:{int(((ncorrect == 0) & (gt == c)).sum())}/{int((gt == c).sum())}" for c in SP)}])
    # acuracia de familia MD/MV/MC (colapsa as tres em um rotulo)
    fam = np.isin(gt, FAMILY)
    fam_pred_ok = np.isin(pred[:, fam], FAMILY)
    res["oracle_family"] = pd.DataFrame([{
        "split": split, "n_family_gt": int(fam.sum()), "acc_species_in_family": float(ok[:, fam].mean()),
        "acc_family_level": float(fam_pred_ok.mean()),
        "share_errors_within_family": float((~ok[:, fam] & fam_pred_ok).sum() / max((~ok[:, fam]).sum(), 1))}])
    # risco x cobertura (media dos classificadores)
    rows = []
    flat_c, flat_ok = allc, allok
    order = np.argsort(-flat_c)
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
        k = int(round(cov * len(order)))
        sel = order[:k]
        rows.append({"split": split, "coverage": cov, "accuracy": float(flat_ok[sel].mean()), "conf_threshold": float(flat_c[sel].min())})
    res["oracle_risk_coverage"] = pd.DataFrame(rows)
    return res


# ----------------------------------------------------------------------------- hibrido (abstencao)
def run_hybrid(root: Path, split: str, taus=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01)) -> pd.DataFrame:
    rows = []
    for fam, det, clf, f in cascade_dirs(root, split):
        if det != "E0_s0":
            continue
        for tau in taus:
            c = load_counts(f, tau).sum(0)
            rows.append({"split": split, "clf": fam, "run": clf, "tau": tau,
                         "f1_single": float(f1_macro(c, 1, 2)), "f1_hybrid": float(f1_macro(c, 3, 4))})
    df = pd.DataFrame(rows)
    return df.groupby(["split", "clf", "tau"], as_index=False)[["f1_single", "f1_hybrid"]].mean() if len(df) else df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/raid/user_marcospaulo/flydet_runs_v9")
    ap.add_argument("--out", default=str(RES))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--splits", nargs="+", default=["test", "val"])
    a = ap.parse_args(argv)
    root, out = Path(a.runs_root), Path(a.out)
    parts: dict[str, list[pd.DataFrame]] = {}
    for sp in a.splits:
        parts.setdefault("v9_bootstrap_delta_f1", []).append(run_bootstrap(root, sp, a.n_boot))
        parts.setdefault("v9_mcnemar", []).append(run_mcnemar(root, sp))
        parts.setdefault("v9_hybrid_abstention", []).append(run_hybrid(root, sp))
        for k, v in run_oracle(root, sp).items():
            parts.setdefault("v9_" + k, []).append(v)
    for name, dfs in parts.items():
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(out / f"{name}.csv", index=False, float_format="%.4f")
        print(name, df.shape)


if __name__ == "__main__":
    main()
