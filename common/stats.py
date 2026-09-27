"""Estatistica descritiva e testes pareados para o paper 2 (CPU, sem scipy).

Funcoes puras sobre listas/arrays/DataFrames; usadas pelo `analyst` para transformar
`per_instance.csv` (harness rq2_cascade/eval_cascade_matrix.py) em numeros do paper.

- mean_std(values)                      -> (media, desvio amostral ddof=1)
- mcnemar_paired(correct_a, correct_b)  -> b, c, statistic, p_value, exact (binomial se b+c<25)
- bootstrap_ci(values, ...)             -> (media, low, high) percentil, seed fixa
- confusion_from_per_instance(csv, ...) -> DataFrame GT x pred (+ coluna BG p/ nao detectados)
- recall_by_size_bins(df, bins=...)     -> recall por faixa de sqrt(w*h) do GT
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

SPECIES = ("MD", "MV", "MC", "MF", "INS", "NOISE", "MAR")
BG = "BG"  # GT sem deteccao correspondente (matched == False)


# --------------------------------------------------------------------------- descritiva
def mean_std(values: Iterable[float]) -> tuple[float, float]:
    """Media e desvio-padrao amostral (ddof=1). n=1 -> std=0.0; vazio -> ValueError."""
    xs = [float(v) for v in values]
    if not xs:
        raise ValueError("mean_std: sequencia vazia")
    n = len(xs)
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var)


# --------------------------------------------------------------------------- McNemar
def _binom_two_sided_p(k: int, n: int) -> float:
    """p bicaudal exato de Binomial(n, 0.5): 2 * P(X <= min(k, n-k)), truncado em 1."""
    if n == 0:
        return 1.0
    kk = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(kk + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _chi2_1df_sf(x: float) -> float:
    """Sobrevivencia da chi-quadrado com 1 g.l.: P(X > x) = erfc(sqrt(x/2))."""
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def mcnemar_paired(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict:
    """Teste de McNemar para dois classificadores avaliados nas MESMAS instancias.

    b = A certo & B errado; c = A errado & B certo.
    - b + c < 25: binomial exato bicaudal (exact=True), statistic = min(b, c).
    - caso contrario: chi2 com correcao de continuidade de Edwards, (|b-c|-1)^2/(b+c), 1 g.l.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("mcnemar_paired: vetores de tamanhos diferentes")
    b = c = 0
    for ca, cb in zip(correct_a, correct_b, strict=True):
        ca, cb = bool(ca), bool(cb)
        if ca and not cb:
            b += 1
        elif cb and not ca:
            c += 1
    n = b + c
    if n < 25:
        return {"b": b, "c": c, "n_discordant": n, "statistic": float(min(b, c)),
                "p_value": _binom_two_sided_p(b, n), "exact": True}
    stat = (abs(b - c) - 1) ** 2 / n
    return {"b": b, "c": c, "n_discordant": n, "statistic": float(stat),
            "p_value": _chi2_1df_sf(stat), "exact": False}


# --------------------------------------------------------------------------- bootstrap
def bootstrap_ci(values: Sequence[float], n: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """IC percentil (1-alpha) da media por bootstrap; retorna (media, low, high)."""
    xs = [float(v) for v in values]
    if not xs:
        raise ValueError("bootstrap_ci: sequencia vazia")
    m = len(xs)
    mean = sum(xs) / m
    if m == 1:
        return mean, mean, mean
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=m)) / m for _ in range(n))
    lo_idx = math.floor((alpha / 2) * (n - 1))
    hi_idx = math.ceil((1 - alpha / 2) * (n - 1))
    return mean, means[lo_idx], means[hi_idx]


# --------------------------------------------------------------------------- per_instance
def _load_per_instance(source) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source
    return pd.read_csv(Path(source))


def confusion_from_per_instance(csv_path, classes: Sequence[str] = SPECIES,
                                pred_col: str = "pred_class_cascade") -> pd.DataFrame:
    """Matriz de confusao GT (linhas) x predicao (colunas) a partir de per_instance.csv.

    GT nao detectado (matched False / pred vazio) conta na coluna BG. Predicoes fora de
    `classes` viram BG tambem (fail-soft, mas avisado via coluna extra `_other`, se houver).
    """
    df = _load_per_instance(csv_path)
    cols = list(classes) + [BG]
    mat = pd.DataFrame(0, index=list(classes), columns=cols, dtype=int)
    matched = df["matched"].astype(str).str.lower().isin(("true", "1"))
    for gt, ok, pred in zip(df["gt_class"], matched, df[pred_col], strict=True):
        if gt not in mat.index:
            continue
        p = pred if (ok and isinstance(pred, str) and pred in classes) else BG
        mat.loc[gt, p] += 1
    return mat


def _gt_size(box: str) -> float:
    x1, y1, x2, y2 = (float(v) for v in str(box).split(","))
    return math.sqrt(max(x2 - x1, 0.0) * max(y2 - y1, 0.0))


def recall_by_size_bins(per_instance_df, bins: Sequence[float] = (0, 32, 48, math.inf),
                        pred_col: str | None = None) -> pd.DataFrame:
    """Recall por faixa de tamanho do GT (sqrt(w*h) em px; gt_box = 'x1,y1,x2,y2').

    Sem `pred_col`: recall de deteccao (matched). Com `pred_col`: recall de especie
    (matched e pred == gt_class). Colunas: bin, n_gt, n_hit, recall.
    """
    df = _load_per_instance(per_instance_df)
    edges = list(bins)
    sizes = df["gt_box"].map(_gt_size)
    matched = df["matched"].astype(str).str.lower().isin(("true", "1"))
    hit = matched if pred_col is None else matched & (df[pred_col] == df["gt_class"])
    rows = []
    for lo, hi in itertools.pairwise(edges):
        sel = (sizes >= lo) & (sizes < hi)
        n = int(sel.sum())
        h = int(hit[sel].sum())
        label = f"[{lo:g},{hi:g})" if math.isfinite(hi) else f"[{lo:g},inf)"
        rows.append({"bin": label, "n_gt": n, "n_hit": h,
                     "recall": (h / n) if n else float("nan")})
    return pd.DataFrame(rows)
