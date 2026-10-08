import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common" / "analysis"))
import rq2_v9_analysis as A  # noqa: E402


def _rec(single_cls, casc_cls, gt_cls, probs=None):
    box = [10, 10, 50, 50]
    return {"image": "a.jpg", "gt": [{"class": gt_cls, "box": box}],
            "proposals": [{"box": box, "yolo_class": single_cls, "yolo_conf": 0.9, "cascade_class": casc_cls,
                           "cascade_score": 0.9, "probabilities": probs or [0.25, 0.25, 0.25, 0.25, 0, 0, 0], "fallback": None}]}


def test_counts_and_hybrid():
    rec = _rec("MD", "MV", "MV", [0.1, 0.4, 0.2, 0.3, 0, 0, 0])
    c = A.image_counts(rec)
    assert c[0, 1] == 1 and c[1, 1] == 0 and c[3, 1] == 1  # cascata acerta MV, detector erra
    h = A.image_counts(rec, tau=0.9)  # classificador pouco confiante -> mantem detector
    assert h[3, 1] == 0 and h[3, 0] == 0


def test_f1_and_bootstrap_sign():
    good = np.zeros((20, 5, 4)); good[:, 0, :] = 1; good[:, 1:5, :] = 1  # sem diferenca
    r = A.paired_bootstrap(good[None], n_boot=200)
    assert abs(r["delta_f1"]) < 1e-9 and r["p_perm"] > 0.5
    better = good.copy(); better[:, 1, :] = 0  # detector perde tp
    r = A.paired_bootstrap(better[None], n_boot=200)
    assert r["delta_f1"] > 0 and r["ci_low"] > 0 and r["p_perm"] < 0.05
    assert r["ci_low_bca"] <= r["delta_f1"] <= r["ci_high_bca"]


def test_ece_perfect():
    conf = np.array([1.0, 1.0, 0.5, 0.5]); ok = np.array([1, 1, 1, 0.0])
    assert A.ece(conf, ok) == 0.0


def test_late_fusion_w0_equals_classifier():
    import rq2_v9_extra as E
    rec = _rec("MD", "MV", "MV", [0.1, 0.5, 0.2, 0.2, 0, 0, 0])
    rec["proposals"][0]["probabilities"] = [0.1, 0.5, 0.2, 0.2, 0.0, 0.0, 0.0]
    a = E.fused_counts(rec, 0.0)
    assert a[3, 1] == 1  # w=0: so o classificador (MV, correto)
    rec["proposals"][0]["yolo_conf"] = 0.999
    b = E.fused_counts(rec, 50.0)
    assert b[3, 1] == 0  # w grande: segue o detector (MD, errado)

