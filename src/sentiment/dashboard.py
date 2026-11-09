"""Read experiment reports for the app's model-comparison tab (no ML imports)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

LOG_COLUMNS = ["experiment", "stage", "model", "lr", "batch", "seed", "class_weighting",
               "val_accuracy", "val_macro_f1", "val_macro_recall", "train_min"]


def load_experiment_log(report_dir: str | Path) -> pd.DataFrame | None:
    """Experiment table (validation metrics only), best first; None if absent."""
    p = Path(report_dir) / "experiment_log.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    keep = [c for c in LOG_COLUMNS if c in df.columns]
    return df[keep].sort_values("val_macro_f1", ascending=False).reset_index(drop=True)


def load_json(report_dir: str | Path, name: str) -> dict | None:
    p = Path(report_dir) / name
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def held_out_table(final: dict) -> pd.DataFrame:
    """One row per held-out dataset: selected vs original model, with CIs and paired test."""
    names = {"tweeteval_test": "TweetEval test", "gop_full_dedup": "GOP election (all)",
             "gop_conf1": "GOP election (high-agreement)"}
    rows = []
    for key, label in names.items():
        r = final.get(key)
        if not r:
            continue
        row = {
            "dataset": label,
            "n": r["n"],
            "accuracy": r["accuracy"],
            "accuracy_ci95": _fmt_ci(r.get("accuracy_ci95")),
            "macro_f1": r["macro_f1"],
            "macro_f1_ci95": _fmt_ci(r.get("macro_f1_ci95")),
        }
        base = r.get("baseline")
        if base:
            paired = r.get("paired_vs_baseline_macro_f1", {})
            row.update({
                "original_accuracy": base["accuracy"],
                "original_macro_f1": base["macro_f1"],
                "delta_macro_f1": paired.get("diff"),
                "delta_ci95": _fmt_ci(paired.get("ci95")),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt_ci(ci) -> str:
    return "" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
