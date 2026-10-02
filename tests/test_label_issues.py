import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rq2_cascade"))
from label_issues import confident_joint, find_label_issues  # noqa: E402


def _data():
    rng = np.random.default_rng(0)
    n, c = 300, 3
    labels = np.repeat(np.arange(c), n // c)
    probs = np.full((n, c), 0.025) + rng.uniform(0, 0.004, size=(n, c))
    probs[np.arange(n), labels] = 0.95
    probs /= probs.sum(1, keepdims=True)
    flip = [0, 150]  # rótulos dados errados: o modelo está confiante na outra classe
    wrong = labels.copy()
    wrong[0], wrong[150] = 1, 2
    return wrong, probs, flip


def test_flags_injected_errors():
    wrong, probs, flip = _data()
    r = find_label_issues(wrong, probs)
    assert r["issue"][flip].all()
    assert r["issue"].sum() <= 10  # poucos falsos suspeitos
    assert r["score"][flip].min() > np.median(r["score"])


def test_joint_counts_off_diagonal():
    wrong, probs, _ = _data()
    j = confident_joint(wrong, probs)
    assert j.shape == (3, 3) and j[1, 0] >= 1 and j[2, 1] >= 1
    assert np.trace(j) > 0.9 * j.sum()


def test_shape_validation():
    import pytest
    with pytest.raises(ValueError):
        find_label_issues(np.zeros(4, dtype=int), np.zeros((5, 3)))
