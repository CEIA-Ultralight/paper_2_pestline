"""Cadeia C (job 33363): ``evaluate`` indexava ``matched`` com o indice do GT na lista completa da imagem.

``matched[img]`` tem so os GT da classe c; com GT de varias classes por imagem o indice
estourava (IndexError) ou marcava o GT errado (TP/FP trocados, sem erro).
"""

from __future__ import annotations

import numpy as np
import pytest
from eval_wbf_tta import evaluate, wbf_fuse


def _box(x: float) -> list[float]:
    return [x, 0.0, x + 10.0, 10.0]


def test_evaluate_gt_de_varias_classes_nao_estoura():
    # classe 1 so tem o GT de indice global 2; matched[img] da classe 1 tem tamanho 1
    gt = {"img": [(_box(0), 0), (_box(100), 0), (_box(200), 1)]}
    pred = {"img": [(0.9, _box(200), 1), (0.8, _box(0), 0), (0.7, _box(100), 0)]}
    assert evaluate(pred, gt, n_classes=2) == pytest.approx(1.0)


def test_evaluate_ensemble_wbf_de_dois_modelos():
    gt = {"img": [(_box(0), 0), (_box(100), 0), (_box(200), 1), (_box(300), 1), (_box(400), 2)]}
    m1 = (np.array([_box(200), _box(300), _box(400), _box(0)], float),
          np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 1, 2, 0]))
    m2 = (np.array([_box(201), _box(301), _box(401), _box(100)], float),
          np.array([0.85, 0.75, 0.65, 0.55]), np.array([1, 1, 2, 0]))
    fb, fs, fl = wbf_fuse([m1[0], m2[0]], [m1[1], m2[1]], [m1[2], m2[2]])
    fused = {"img": list(zip(fs.tolist(), fb.tolist(), fl.tolist(), strict=True))}
    # todas as classes recuperam seus GT: AP = 1 (classe 0 tem 2 GT, ambos detectados)
    assert evaluate(fused, gt, n_classes=3) == pytest.approx(1.0)


def test_evaluate_duplicata_conta_fp_com_gt_de_varias_classes():
    gt = {"img": [(_box(0), 0), (_box(100), 1), (_box(200), 1)]}
    pred = {"img": [(0.95, _box(0), 0), (0.9, _box(100), 1), (0.8, _box(100), 1), (0.7, _box(200), 1)]}
    # classe 1: TP, FP (duplicata), TP -> AP = 0.5*1 + 0.5*(2/3)
    assert evaluate(pred, gt, n_classes=2) == pytest.approx((1.0 + (0.5 + 0.5 * 2 / 3)) / 2)
