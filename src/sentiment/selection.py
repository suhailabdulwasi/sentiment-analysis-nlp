"""Compare experiments and choose the final model using validation data only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sentiment.metrics import classification_metrics, paired_bootstrap


def load_results(run_root: str | Path) -> list[dict]:
    out = []
    for p in sorted(Path(run_root).glob("*/result.json")):
        r = json.loads(p.read_text())
        r["_dir"] = str(p.parent)
        out.append(r)
    return out


def comparison_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        c, v = r["config"], r["validation"]
        rows.append(
            {
                "experiment": r["name"],
                "stage": c.get("stage", ""),
                "model": c["model_name"],
                "lr": c["learning_rate"],
                "batch": c["per_device_train_batch_size"] * c.get("gradient_accumulation_steps", 1),
                "max_epochs": c["num_train_epochs"],
                "epochs_run": r.get("epochs_completed"),
                "seed": c["seed"],
                "class_weighting": c.get("class_weighting", False),
                "val_accuracy": round(v["accuracy"], 4),
                "val_macro_f1": round(v["macro_f1"], 4),
                "val_macro_recall": round(v["macro_recall"], 4),
                "val_ece": round(v.get("ece", float("nan")), 4),
                "train_min": round(r["train_seconds"] / 60, 1),
                "gpu": r.get("environment", {}).get("gpu", "cpu"),
                "hypothesis": c.get("hypothesis", ""),
            }
        )
    return pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)


def to_markdown(df: pd.DataFrame) -> str:
    """Minimal GitHub-markdown table (avoids the optional ``tabulate`` dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def seed_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std of validation metrics across seeds for identical configs."""
    keys = ["model", "lr", "batch", "class_weighting"]
    g = table.groupby(keys).agg(
        runs=("seed", "nunique"),
        val_macro_f1_mean=("val_macro_f1", "mean"),
        val_macro_f1_std=("val_macro_f1", "std"),
        val_accuracy_mean=("val_accuracy", "mean"),
    )
    return g.reset_index().sort_values("val_macro_f1_mean", ascending=False)


def ensemble_candidates(
    results: list[dict], y_val: np.ndarray, max_members: int = 5, one_per_model: bool = True
) -> list[dict]:
    """Average probabilities of the top-k runs (k = 1..max_members) by val macro-F1.

    Only runs with a saved checkpoint are eligible. With ``one_per_model`` the
    members come from distinct architectures, which tends to give more diverse
    errors than several seeds of one model.
    """
    ranked = sorted(results, key=lambda r: r["validation"]["macro_f1"], reverse=True)
    ranked = [r for r in ranked if (Path(r["_dir"]) / "best").exists()]
    if one_per_model:
        seen, uniq = set(), []
        for r in ranked:
            if r["config"]["model_name"] not in seen:
                seen.add(r["config"]["model_name"])
                uniq.append(r)
        ranked = uniq
    out = []
    for k in range(1, min(max_members, len(ranked)) + 1):
        members = ranked[:k]
        probs = np.mean([np.load(Path(m["_dir"]) / "val_probs.npy") for m in members], axis=0)
        m = classification_metrics(y_val, probs.argmax(1))
        out.append(
            {
                "k": k,
                "members": [x["name"] for x in members],
                "member_dirs": [x["_dir"] for x in members],
                "val_macro_f1": m["macro_f1"],
                "val_accuracy": m["accuracy"],
                "val_probs": probs,
            }
        )
    return out


def choose_final(
    candidates: list[dict], y_val: np.ndarray, min_gain: float = 0.005
) -> dict:
    """Prefer the simplest candidate unless a bigger ensemble is clearly better.

    A larger ensemble is accepted only if its validation macro-F1 gain over the
    current choice is >= ``min_gain`` *and* the paired-bootstrap CI excludes 0.
    This guards against selecting noise on a 2,000-example validation set.
    """
    if not candidates:
        raise ValueError("no candidates")
    best = candidates[0]
    for c in candidates[1:]:
        test = paired_bootstrap(y_val, best["val_probs"].argmax(1), c["val_probs"].argmax(1))
        if test["diff"] >= min_gain and test["ci95"][0] > 0:
            best = {**c, "vs_previous": test}
    return best
