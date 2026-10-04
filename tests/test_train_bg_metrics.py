import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rq2_cascade"))
from architecture_lab.train import BG_CLASSES, EXPECTED_CLASSES, compute_metrics  # noqa: E402


def test_compute_metrics_accepts_bg_classes():
    m = compute_metrics([0, 1, 4, 7], [0, 1, 5, 7], BG_CLASSES)
    assert m["class_order"] == list(BG_CLASSES) and m["num_samples"] == 4
    assert m["per_class"]["BG"]["tp"] == 1


def test_compute_metrics_7_classes_still_valid_and_strict():
    assert compute_metrics([0, 1], [0, 1], EXPECTED_CLASSES)["overall_accuracy"] == 1.0
    with pytest.raises(ValueError):
        compute_metrics([0], [0], tuple(reversed(EXPECTED_CLASSES)))
