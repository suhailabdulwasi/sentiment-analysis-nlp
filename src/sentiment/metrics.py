"""Evaluation metrics with uncertainty estimates.

Note: the *official* TweetEval sentiment metric is macro-averaged **recall**
(Barbieri et al., 2020). This project selects models by macro-F1 (as requested)
but always reports macro-recall too so numbers are comparable with the literature.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)

from sentiment.labels import LABELS

_IDS = list(range(len(LABELS)))


def _check(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    if yt.size == 0:
        raise ValueError("empty inputs")
    return yt, yp


def classification_metrics(y_true, y_pred) -> dict:
    """Accuracy, macro-F1, macro-recall, per-class P/R/F1 and confusion matrix."""
    yt, yp = _check(y_true, y_pred)
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=_IDS, zero_division=0)
    return {
        "n": int(yt.size),
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, labels=_IDS, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(yt, yp, labels=_IDS, average="macro", zero_division=0)),
        "per_class": {
            LABELS[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i in _IDS
        },
        "confusion_matrix": confusion_matrix(yt, yp, labels=_IDS).tolist(),
    }


def bootstrap_ci(
    y_true, y_pred, metric: str = "macro_f1", n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap CI over examples. Returns (point, low, high)."""
    yt, yp = _check(y_true, y_pred)
    fn = _METRIC_FNS[metric]
    rng = np.random.default_rng(seed)
    n = yt.size
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = fn(yt[idx], yp[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(fn(yt, yp)), float(lo), float(hi)


def paired_bootstrap(
    y_true, pred_a, pred_b, metric: str = "macro_f1", n_boot: int = 2000, seed: int = 0
) -> dict:
    """Paired bootstrap for metric(B) - metric(A).

    Returns the observed difference, its 95% CI and a one-sided p-value
    estimate of P(diff <= 0) — i.e. evidence that B is *not* better than A.
    """
    yt, pa = _check(y_true, pred_a)
    _, pb = _check(y_true, pred_b)
    fn = _METRIC_FNS[metric]
    rng = np.random.default_rng(seed)
    n = yt.size
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = fn(yt[idx], pb[idx]) - fn(yt[idx], pa[idx])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "diff": float(fn(yt, pb) - fn(yt, pa)),
        "ci95": [float(lo), float(hi)],
        "p_not_better": float((diffs <= 0).mean()),
    }


def expected_calibration_error(probs, y_true, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins."""
    probs = np.asarray(probs, dtype=float)
    yt = np.asarray(y_true)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == yt).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


_METRIC_FNS = {
    "accuracy": lambda t, p: accuracy_score(t, p),
    "macro_f1": lambda t, p: f1_score(t, p, labels=_IDS, average="macro", zero_division=0),
    "macro_recall": lambda t, p: recall_score(t, p, labels=_IDS, average="macro", zero_division=0),
}
