import numpy as np
import pytest
from sklearn.metrics import f1_score

from sentiment.metrics import (
    bootstrap_ci,
    classification_metrics,
    expected_calibration_error,
    paired_bootstrap,
)


def test_metrics_match_sklearn_and_hand_values():
    # Reproduces the original model's reported TweetEval test numbers from its confusion matrix
    cm = np.array([[3063, 781, 128], [1262, 3787, 888], [68, 508, 1799]])
    y, p = [], []
    for i in range(3):
        for j in range(3):
            y += [i] * cm[i, j]
            p += [j] * cm[i, j]
    m = classification_metrics(y, p)
    assert m["n"] == 12284
    assert round(m["accuracy"], 4) == 0.7041
    assert round(m["macro_f1"], 4) == 0.7044
    assert round(m["macro_recall"], 4) == 0.7222
    assert m["confusion_matrix"] == cm.tolist()
    assert m["macro_f1"] == pytest.approx(f1_score(y, p, average="macro"))
    assert m["per_class"]["neutral"]["support"] == 5937


def test_missing_class_is_scored_zero_not_dropped():
    m = classification_metrics([0, 0, 1], [0, 0, 1])
    assert m["per_class"]["positive"]["f1"] == 0.0
    assert m["macro_f1"] == pytest.approx(2 / 3)


def test_input_validation():
    with pytest.raises(ValueError):
        classification_metrics([0, 1], [0])
    with pytest.raises(ValueError):
        classification_metrics([], [])


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 500)
    p = np.where(rng.random(500) < 0.7, y, rng.integers(0, 3, 500))
    point, lo, hi = bootstrap_ci(y, p, "accuracy", n_boot=300)
    assert lo <= point <= hi and hi - lo < 0.15


def test_paired_bootstrap_detects_improvement_and_null():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 1000)
    a = np.where(rng.random(1000) < 0.6, y, (y + 1) % 3)
    b = np.where(rng.random(1000) < 0.8, y, (y + 1) % 3)
    res = paired_bootstrap(y, a, b, n_boot=300)
    assert res["diff"] > 0 and res["ci95"][0] > 0 and res["p_not_better"] < 0.01
    same = paired_bootstrap(y, a, a, n_boot=50)
    assert same["diff"] == 0 and same["ci95"] == [0.0, 0.0]


def test_ece():
    y = np.array([0, 1, 2, 0])
    perfect = np.eye(3)[y]
    assert expected_calibration_error(perfect, y) == pytest.approx(0.0)
    overconf = np.tile([0.98, 0.01, 0.01], (4, 1))  # always says class 0 with 98%
    assert expected_calibration_error(overconf, y) == pytest.approx(0.98 - 0.5)
