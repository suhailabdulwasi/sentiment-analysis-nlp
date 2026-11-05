"""One-time held-out evaluation of the selected model / ensemble.

Run only after model selection is frozen. Writes a lock file recording which
members were evaluated; re-running with a *different* selection is refused
unless ``force=True`` so that test-set results cannot be iterated on silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sentiment.inference import SentimentClassifier
from sentiment.metrics import bootstrap_ci, classification_metrics, expected_calibration_error


def predict_probs(model_dirs: list[str], texts: list[str], batch_size: int = 128) -> np.ndarray:
    """Average softmax probabilities of several checkpoints."""
    import gc

    import torch

    all_probs = []
    for d in model_dirs:
        clf = SentimentClassifier(d)
        preds = clf.predict(texts, batch_size=batch_size)
        from sentiment.labels import LABELS

        all_probs.append(
            np.array([[p.probabilities.get(k, np.nan) for k in LABELS] for p in preds])
        )
        del clf
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.mean(all_probs, axis=0)


def fit_temperature(val_probs: np.ndarray, y_val: np.ndarray) -> float:
    """Single-parameter temperature scaling fitted on validation log-probs."""
    logp = np.log(np.clip(val_probs, 1e-12, 1))
    best_t, best_nll = 1.0, np.inf
    for t in np.linspace(0.25, 5.0, 476):
        z = logp / t
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        nll = -lp[np.arange(len(y_val)), y_val].mean()
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return best_t


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(probs, 1e-12, 1)) / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def summarize(name: str, y: np.ndarray, probs: np.ndarray, n_boot: int = 2000) -> dict:
    pred = probs.argmax(1)
    m = classification_metrics(y, pred)
    for metric in ("accuracy", "macro_f1", "macro_recall"):
        _, lo, hi = bootstrap_ci(y, pred, metric=metric, n_boot=n_boot)
        m[f"{metric}_ci95"] = [lo, hi]
    m["ece"] = expected_calibration_error(probs, y)
    m["dataset"] = name
    return m


def check_lock(lock_path: Path, members: list[str], force: bool) -> None:
    if lock_path.exists():
        prev = json.loads(lock_path.read_text())
        if sorted(prev["members"]) != sorted(members) and not force:
            raise RuntimeError(
                f"Test set already evaluated for {prev['members']}. Evaluating a different "
                "selection would turn the test set into a tuning set. Pass force=True only "
                "if you document why."
            )


def save_predictions(path: Path, df: pd.DataFrame, probs: np.ndarray) -> None:
    from sentiment.labels import LABELS

    out = df[["text", "label"]].copy()
    out["pred"] = probs.argmax(1)
    for i, k in enumerate(LABELS):
        out[f"p_{k}"] = probs[:, i]
    out["confidence"] = probs.max(1)
    out.to_csv(path, index=False)
