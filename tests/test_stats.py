"""Testes CPU de common/stats.py (sem scipy; valores de referencia calculados a mao)."""

from __future__ import annotations

import math

import pandas as pd
import pytest
import stats  # common/ esta no sys.path via tests/conftest.py


def test_mean_std_basic():
    m, s = stats.mean_std([2, 4, 4, 4, 5, 5, 7, 9])
    assert m == pytest.approx(5.0)
    assert s == pytest.approx(2.138089935)  # ddof=1 -> sqrt(32/7)


def test_mean_std_single_and_empty():
    assert stats.mean_std([3.0]) == (3.0, 0.0)
    with pytest.raises(ValueError):
        stats.mean_std([])


def test_mcnemar_exact_small_discordant():
    # b=3 (A certo, B errado), c=1: n=4 <25 -> binomial exato
    a = [1, 1, 1, 0, 1, 1, 0, 0]
    b = [0, 0, 0, 1, 1, 1, 0, 0]
    r = stats.mcnemar_paired(a, b)
    assert (r["b"], r["c"]) == (3, 1)
    assert r["exact"] is True
    # 2 * P(X<=1 | n=4, p=.5) = 2 * (1+4)/16 = 0.625
    assert r["p_value"] == pytest.approx(0.625)


def test_mcnemar_chi2_large_discordant():
    # b=30, c=10 -> n=40 >=25 -> chi2 com correcao: (|20|-1)^2/40 = 9.025
    a = [1] * 30 + [0] * 10 + [1] * 5
    b = [0] * 30 + [1] * 10 + [1] * 5
    r = stats.mcnemar_paired(a, b)
    assert r["exact"] is False
    assert r["statistic"] == pytest.approx(9.025)
    assert r["p_value"] == pytest.approx(math.erfc(math.sqrt(9.025 / 2)))
    assert 0.002 < r["p_value"] < 0.003  # ~0.00266


def test_mcnemar_no_discordance():
    r = stats.mcnemar_paired([1, 0, 1], [1, 0, 1])
    assert r["n_discordant"] == 0 and r["p_value"] == 1.0


def test_mcnemar_length_mismatch():
    with pytest.raises(ValueError):
        stats.mcnemar_paired([1, 0], [1])


def test_bootstrap_ci_deterministic_and_contains_mean():
    xs = [0.80, 0.82, 0.79, 0.85, 0.81]
    r1 = stats.bootstrap_ci(xs, n=2000, seed=0)
    r2 = stats.bootstrap_ci(xs, n=2000, seed=0)
    assert r1 == r2
    mean, lo, hi = r1
    assert lo <= mean <= hi
    assert min(xs) <= lo and hi <= max(xs)
    assert stats.bootstrap_ci([0.5]) == (0.5, 0.5, 0.5)


def _per_instance_df():
    return pd.DataFrame(
        {
            "image": ["a", "a", "b", "b", "c"],
            "gt_class": ["MD", "MV", "MD", "MC", "MAR"],
            "gt_box": ["0,0,20,20", "0,0,40,40", "0,0,50,50", "0,0,30,30", "0,0,100,100"],
            "matched": [True, True, False, True, True],
            "pred_class_single": ["MD", "MD", "", "MC", "MAR"],
            "pred_class_cascade": ["MD", "MV", "", "MF", "MAR"],
        }
    )


def test_confusion_from_per_instance(tmp_path):
    df = _per_instance_df()
    p = tmp_path / "per_instance.csv"
    df.to_csv(p, index=False)
    cm = stats.confusion_from_per_instance(p)
    assert list(cm.index) == list(stats.SPECIES)
    assert list(cm.columns) == list(stats.SPECIES) + ["BG"]
    assert cm.loc["MD", "MD"] == 1
    assert cm.loc["MD", "BG"] == 1  # nao detectado
    assert cm.loc["MV", "MV"] == 1
    assert cm.loc["MC", "MF"] == 1
    assert cm.loc["MAR", "MAR"] == 1
    assert int(cm.values.sum()) == 5
    cm_single = stats.confusion_from_per_instance(df, pred_col="pred_class_single")
    assert cm_single.loc["MV", "MD"] == 1


def test_recall_by_size_bins():
    df = _per_instance_df()  # tamanhos: 20, 40, 50, 30, 100
    out = stats.recall_by_size_bins(df)  # bins (0,32,48,inf)
    assert list(out["bin"]) == ["[0,32)", "[32,48)", "[48,inf)"]
    assert list(out["n_gt"]) == [2, 1, 2]
    assert list(out["n_hit"]) == [2, 1, 1]  # 50 px nao detectado
    assert out["recall"].tolist() == pytest.approx([1.0, 1.0, 0.5])
    # recall de especie da cascata: MC (30 px) previsto MF -> erro na primeira faixa
    sp = stats.recall_by_size_bins(df, pred_col="pred_class_cascade")
    assert sp["recall"].tolist() == pytest.approx([0.5, 1.0, 0.5])
