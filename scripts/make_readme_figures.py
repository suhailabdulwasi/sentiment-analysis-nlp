"""Figures for the README from verified numbers (data audit + recorded confusion matrices)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sentiment.error_analysis import plot_confusion  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
SERIES = {"train": "#2a78d6", "validation": "#eb6834", "test": "#1baf7a"}


def label_shift() -> None:
    audit = json.loads((ROOT / "reports" / "data_audit.json").read_text())["tweeteval"]["label_distribution"]
    classes = ["negative", "neutral", "positive"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    w = 0.26
    for i, (split, color) in enumerate(SERIES.items()):
        vals = [audit[split][c]["pct"] for c in classes]
        xs = [j + (i - 1) * (w + 0.02) for j in range(3)]
        ax.bar(xs, vals, width=w, color=color, label=split.capitalize(), zorder=2)
        for x, v in zip(xs, vals, strict=True):
            ax.text(x, v + 0.8, f"{v:.0f}%", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(3), [c.capitalize() for c in classes], color=INK)
    ax.set_ylabel("Share of split (%)", color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.set_ylim(0, 55)
    ax.legend(frameon=False, ncols=3, loc="upper left", fontsize=8)
    ax.set_title("TweetEval label distribution: test is twice as negative as train/validation",
                 fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "tweeteval_label_shift.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def confusions() -> None:
    original = [[3063, 781, 128], [1262, 3787, 888], [68, 508, 1799]]  # notebook cell 20
    plot_confusion(original, "Original DistilRoBERTa - TweetEval test (row-normalised)",
                   FIG / "confusion_original_tweeteval_test.png")
    base = json.loads((ROOT / "reports" / "baseline_results.json").read_text())
    plot_confusion(base["test"]["confusion_matrix"], "TF-IDF baseline B1 - TweetEval test",
                   FIG / "confusion_baseline_tweeteval_test.png")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    label_shift()
    confusions()
    print(sorted(p.name for p in FIG.iterdir()))
