"""Run dataset audits and write reports/data_audit.json.

Usage: python scripts/audit_data.py --data-dir data/raw --gop data/raw/gop/Sentiment.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sentiment.audit import cross_split_overlap, overlap_rows, within_split_duplicates
from sentiment.data import (
    GOP_HIGH_AGREEMENT_THRESHOLD,
    GOP_SHA256,
    TWEETEVAL_SIZES,
    download_tweeteval,
    load_gop_debate,
    load_tweeteval,
)
from sentiment.labels import LABELS


def dist(df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts().reindex(range(3), fill_value=0)
    return {LABELS[i]: {"n": int(c), "pct": round(100 * c / len(df), 2)} for i, c in counts.items()}


def distribution_shift(splits: dict[str, pd.DataFrame]) -> dict:
    """Statistical tests for train/val vs test shift (labels, length, surface features)."""
    from scipy.stats import chi2_contingency, mannwhitneyu

    out: dict = {}
    for a_name, b_name in (("train", "validation"), ("train", "test"), ("validation", "test")):
        a, b = splits[a_name], splits[b_name]
        table = [np.bincount(a["label"], minlength=3), np.bincount(b["label"], minlength=3)]
        chi2, p, _, _ = chi2_contingency(table)
        n = len(a) + len(b)
        cramers_v = float(np.sqrt(chi2 / n / 2))
        la, lb = a["text"].str.split().str.len(), b["text"].str.split().str.len()
        u = mannwhitneyu(la, lb)
        out[f"{a_name}_vs_{b_name}"] = {
            "label_chi2": round(float(chi2), 1), "label_p": float(p), "label_cramers_v": round(cramers_v, 4),
            "len_words_mean": [round(float(la.mean()), 2), round(float(lb.mean()), 2)],
            "len_mannwhitney_p": float(u.pvalue),
        }
    feats = {
        "has_url": r"http",
        "has_user_mention": r"@user",
        "has_hashtag": r"#\w+",
        "starts_with_quote": r'^"',
    }
    out["surface_feature_pct"] = {
        s: {k: round(100 * float(df["text"].str.contains(rx, regex=True).mean()), 2) for k, rx in feats.items()}
        for s, df in splits.items()
    }
    # Majority-class and prior-only reference points
    out["majority_class_accuracy"] = {
        s: round(float(df["label"].value_counts(normalize=True).max()), 4) for s, df in splits.items()
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--gop", default="data/raw/gop/Sentiment.csv")
    p.add_argument("--out", default="reports/data_audit.json")
    a = p.parse_args()

    root = download_tweeteval(a.data_dir)
    splits = load_tweeteval(root)
    report: dict = {"tweeteval": {}, "gop_debate": {}}
    te = report["tweeteval"]
    te["sizes_match_expected"] = {s: len(df) == TWEETEVAL_SIZES[s] for s, df in splits.items()}
    te["label_distribution"] = {s: dist(df) for s, df in splits.items()}
    te["within_split_exact"] = {s: within_split_duplicates(df) for s, df in splits.items()}
    te["within_split_normalized"] = {
        s: within_split_duplicates(df, key="normalized") for s, df in splits.items()
    }
    te["cross_split_exact"] = cross_split_overlap(splits)
    te["cross_split_normalized"] = cross_split_overlap(splits, key="normalized")
    lengths = {s: df["text"].str.split().str.len() for s, df in splits.items()}
    te["whitespace_tokens"] = {
        s: {"mean": round(float(v.mean()), 1), "p99": int(v.quantile(0.99)), "max": int(v.max())}
        for s, v in lengths.items()
    }
    te["raw_handles_remaining"] = {
        s: int(df["text"].str.contains(r"@(?!user\b)\w+", regex=True).sum())
        for s, df in splits.items()
    }

    te["distribution_shift"] = distribution_shift(splits)

    gop_path = Path(a.gop)
    if gop_path.exists():
        g = report["gop_debate"]
        g["sha256"] = hashlib.sha256(gop_path.read_bytes()).hexdigest()
        g["sha256_matches_pinned"] = g["sha256"] == GOP_SHA256
        raw = pd.read_csv(gop_path)
        g["raw_rows"] = int(len(raw))
        g["raw_sentiment_values"] = raw["sentiment"].value_counts().to_dict()
        g["relevant_yn_values"] = raw["relevant_yn"].value_counts().to_dict()
        g["sentiment_confidence_quantiles"] = {
            str(q): round(float(raw["sentiment_confidence"].quantile(q)), 4)
            for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        }
        g["retweets_rows"] = int(raw["text"].str.match(r"^RT @").sum())
        g["gold_rows"] = int(raw["sentiment_gold"].notna().sum())
        full = load_gop_debate(gop_path)
        g["dedup_rows"] = int(len(full))
        g["dedup_label_distribution"] = dist(full)
        g["high_agreement_threshold"] = GOP_HIGH_AGREEMENT_THRESHOLD
        vals = np.sort(raw["sentiment_confidence"].unique())
        g["confidence_values_between_0.71_and_0.999"] = int(((vals > 0.71) & (vals < 0.999)).sum())
        for thr in (0.6, 0.8, 1.0):
            sub = full[full["confidence"] >= thr]
            g[f"dedup_conf_ge_{thr}"] = {"rows": int(len(sub)), "dist": dist(sub)}
        for s, df in splits.items():
            m = overlap_rows(df, full)
            g[f"overlap_with_tweeteval_{s}"] = int(m.sum())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
