"""Error analysis utilities: heuristic slice tags, confusion pairs and figures.

The slice tags are *heuristics* (?:keyword/regex based) meant to surface where
errors concentrate; they are not ground-truth annotations of sarcasm etc.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from sentiment.labels import LABELS

NEGATION = re.compile(
    r"\b(?:not|no|never|nothing|nobody|neither|nor|cannot|can't|won't|don't|doesn't|didn't|"
    r"isn't|wasn't|aren't|weren't|haven't|hasn't|couldn't|shouldn't|wouldn't|ain't)\b",
    re.IGNORECASE,
)
SARCASM = re.compile(
    r"#sarcasm|#irony|\byeah right\b|\bsure+\b.*\b(?:great|smart|genius)|/s\b|🙄|😏|\bthanks a lot\b|"
    r"\bjust what (?:i|we) needed\b|\boh great\b",
    re.IGNORECASE,
)
POLITICAL = re.compile(
    r"\b(?:trump|clinton|hillary|obama|biden|harris|sanders|bernie|cruz|rubio|bush|gop|democrat\w*|"
    r"republican\w*|election|vote\w*|voting|ballot|campaign|senate|senator|congress|president\w*|"
    r"debate|putin|isis|brexit|liberal\w*|conservative\w*|policy|government)\b",
    re.IGNORECASE,
)
NEWS = re.compile(
    r"\b(?:breaking|report(?:s|ed)?|announce[sd]?|according to|says|said|via|live|update[sd]?|"
    r"officials?|police|killed|dead|attack|fire|crash|survey|poll)\b|\bhttp\b",
    re.IGNORECASE,
)
POS_WORDS = re.compile(
    r"\b(?:love|great|good|best|happy|amazing|awesome|excited|win|wonderful|beautiful|thank\w*|"
    r"enjoy\w*|glad|perfect|nice|fun)\b",
    re.IGNORECASE,
)
NEG_WORDS = re.compile(
    r"\b(?:hate|bad|worst|sad|angry|terrible|awful|kill\w*|dead|death|fail\w*|wrong|sick|"
    r"stupid|disappoint\w*|attack\w*|war|crisis|lose|lost|fear|horrible)\b",
    re.IGNORECASE,
)


def tag_slices(texts: pd.Series) -> pd.DataFrame:
    t = texts.fillna("").astype(str)
    n_words = t.str.split().str.len()
    return pd.DataFrame(
        {
            "negation": t.str.contains(NEGATION),
            "sarcasm_marker": t.str.contains(SARCASM),
            "political": t.str.contains(POLITICAL),
            "news_like": t.str.contains(NEWS),
            "mixed_lexicon": t.str.contains(POS_WORDS) & t.str.contains(NEG_WORDS),
            "neg_words_only": t.str.contains(NEG_WORDS) & ~t.str.contains(POS_WORDS),
            "question": t.str.contains(r"\?"),
            "short_<=8_words": n_words <= 8,
            "hashtag_heavy_>=3": t.str.count(r"#\w+") >= 3,
        },
        index=texts.index,
    )


def slice_report(df: pd.DataFrame) -> pd.DataFrame:
    """Error rate and macro-F1 per heuristic slice. ``df`` needs text/label/pred."""
    from sentiment.metrics import classification_metrics

    tags = tag_slices(df["text"])
    overall_err = float((df["label"] != df["pred"]).mean())
    rows = [{"slice": "ALL", "n": len(df), "share": 1.0, "error_rate": overall_err,
             "macro_f1": classification_metrics(df["label"], df["pred"])["macro_f1"]}]
    for col in tags:
        sub = df[tags[col]]
        if len(sub) < 20:
            continue
        rows.append(
            {
                "slice": col,
                "n": len(sub),
                "share": len(sub) / len(df),
                "error_rate": float((sub["label"] != sub["pred"]).mean()),
                "macro_f1": classification_metrics(sub["label"], sub["pred"])["macro_f1"],
            }
        )
    out = pd.DataFrame(rows)
    out["error_rate_vs_all"] = out["error_rate"] - overall_err
    return out.round(4)


def confusion_pairs(df: pd.DataFrame) -> pd.DataFrame:
    wrong = df[df["label"] != df["pred"]]
    pairs = wrong.groupby(["label", "pred"]).size().reset_index(name="count")
    pairs["actual"] = pairs["label"].map(dict(enumerate(LABELS)))
    pairs["predicted"] = pairs["pred"].map(dict(enumerate(LABELS)))
    pairs["share_of_errors"] = (pairs["count"] / len(wrong)).round(4)
    return pairs.sort_values("count", ascending=False)[["actual", "predicted", "count", "share_of_errors"]]


def confident_disagreements(df: pd.DataFrame, threshold: float = 0.9, n: int = 50) -> pd.DataFrame:
    """High-confidence errors: the best place to look for possible label noise.
    These need *manual* review; disagreement alone does not prove a label is wrong."""
    wrong = df[(df["label"] != df["pred"]) & (df["confidence"] >= threshold)]
    return wrong.sort_values("confidence", ascending=False).head(n)


# ----------------------------------------------------------------- figures

def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10})
    return plt


def plot_confusion(cm, title: str, path: str | Path, normalize: bool = True) -> None:
    plt = _plt()
    cm = np.asarray(cm, dtype=float)
    shown = cm / cm.sum(1, keepdims=True) if normalize else cm
    fig, ax = plt.subplots(figsize=(4.6, 4))
    from matplotlib.colors import LinearSegmentedColormap

    ramp = LinearSegmentedColormap.from_list(
        "blue_ramp", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    im = ax.imshow(shown, cmap=ramp, vmin=0, vmax=1 if normalize else None)
    for i in range(3):
        for j in range(3):
            txt = f"{shown[i, j]:.0%}\n({int(cm[i, j])})" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if shown[i, j] > 0.55 * shown.max() else "#222")
    ax.set_xticks(range(3), [x.capitalize() for x in LABELS])
    ax.set_yticks(range(3), [x.capitalize() for x in LABELS])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_reliability(probs: np.ndarray, y: np.ndarray, title: str, path: str | Path, n_bins: int = 10) -> None:
    plt = _plt()
    conf, pred = probs.max(1), probs.argmax(1)
    correct = pred == y
    edges = np.linspace(1 / 3, 1, n_bins + 1)
    mids, accs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.sum() >= 5:
            mids.append(conf[m].mean())
            accs.append(correct[m].mean())
            counts.append(m.sum())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.5, 3.6))
    a1.plot([1 / 3, 1], [1 / 3, 1], ls="--", c="#999", lw=1, label="perfect calibration")
    a1.plot(mids, accs, marker="o", c="#2b6cb0", label="model")
    a1.set_xlabel("Mean confidence")
    a1.set_ylabel("Accuracy")
    a1.set_title("Reliability", fontsize=10)
    a1.legend(frameon=False, fontsize=8)
    a2.hist(conf[correct], bins=20, range=(1 / 3, 1), alpha=0.75, color="#2b6cb0", label="correct")
    a2.hist(conf[~correct], bins=20, range=(1 / 3, 1), alpha=0.75, color="#dd6b20", label="incorrect")
    a2.set_xlabel("Top-class confidence")
    a2.set_ylabel("Count")
    a2.set_title("Confidence distribution", fontsize=10)
    a2.legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_learning_curves(results: list[dict], path: str | Path, max_runs: int = 8) -> None:
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ranked = sorted(results, key=lambda r: r["validation"]["macro_f1"], reverse=True)[:max_runs]
    for r in ranked:
        hist = r["log_history"]
        ev = [(h["epoch"], h["eval_macro_f1"]) for h in hist if "eval_macro_f1" in h]
        tr = [(h["epoch"], h["loss"]) for h in hist if "loss" in h]
        if ev:
            a1.plot(*zip(*ev, strict=True), marker=".", label=r["name"])
        if tr:
            a2.plot(*zip(*tr, strict=True), lw=1, label=r["name"])
    a1.set_xlabel("Epoch")
    a1.set_ylabel("Validation macro-F1")
    a1.set_title("Validation macro-F1", fontsize=10)
    a2.set_xlabel("Epoch")
    a2.set_ylabel("Training loss")
    a2.set_title("Training loss", fontsize=10)
    a1.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_model_comparison(table: pd.DataFrame, path: str | Path, baseline: float | None = None) -> None:
    plt = _plt()
    t = table.sort_values("val_macro_f1")
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(t) + 1.2))
    ax.barh(t["experiment"], t["val_macro_f1"], color="#2b6cb0")
    if baseline is not None:
        ax.axvline(baseline, ls="--", c="#dd6b20", lw=1, label="original model (val)")
        ax.legend(frameon=False, fontsize=8)
    lo = max(0.0, t["val_macro_f1"].min() - 0.05)
    ax.set_xlim(lo, min(1.0, t["val_macro_f1"].max() + 0.02))
    ax.set_xlabel("Validation macro-F1")
    for y, v in enumerate(t["val_macro_f1"]):
        ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
