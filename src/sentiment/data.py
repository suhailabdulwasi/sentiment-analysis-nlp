"""Dataset loading. All loaders return DataFrames with ``text`` and ``label`` (int)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from sentiment.labels import LABEL2ID, normalize_label

TWEETEVAL_BASE = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment"
TWEETEVAL_SPLITS = {"train": "train", "validation": "val", "test": "test"}
TWEETEVAL_SIZES = {"train": 45_615, "validation": 2_000, "test": 12_284}

GOP_URL = (
    "https://raw.githubusercontent.com/sanjay-raghu/sentiment-analysis-using-LSTM-keras/"
    "master/input/Sentiment.csv"
)
GOP_SHA256 = "83beb7691f876f9a762613e7c262f7a777a529cccfb96478089e912f1292c61c"

# Pre-registered BEFORE any model was evaluated on this dataset (see docs/DATA_AUDIT.md).
# CrowdFlower's ``sentiment_confidence`` is a trust-weighted annotator-agreement score.
# Its distribution is bimodal: 5,367 of 13,871 raw rows are exactly 1.0, almost all
# others are <= 0.75, and only 17 raw rows lie in (0.75, 1.0) (none in [0.8, 1.0)
# after de-duplication). 1.0 ("all weighted judgments agree") is therefore the
# natural high-agreement cut-off and is fixed here in advance.
GOP_HIGH_AGREEMENT_THRESHOLD = 1.0


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed https URLs
    return dest


def download_tweeteval(data_dir: str | Path) -> Path:
    """Fetch the official TweetEval sentiment text/label files from GitHub."""
    root = Path(data_dir) / "tweeteval"
    for prefix in TWEETEVAL_SPLITS.values():
        for kind in ("text", "labels"):
            _download(f"{TWEETEVAL_BASE}/{prefix}_{kind}.txt", root / f"{prefix}_{kind}.txt")
    return root


def load_tweeteval_split(root: str | Path, split: str) -> pd.DataFrame:
    """Load one split from the official line-aligned text/label files."""
    if split not in TWEETEVAL_SPLITS:
        raise ValueError(f"split must be one of {list(TWEETEVAL_SPLITS)}")
    prefix = TWEETEVAL_SPLITS[split]
    root = Path(root)
    # Read line-by-line: tweets may contain quote characters that confuse CSV parsers.
    texts = (root / f"{prefix}_text.txt").read_text(encoding="utf-8").split("\n")
    labels = (root / f"{prefix}_labels.txt").read_text(encoding="utf-8").split("\n")
    texts = texts[:-1] if texts and texts[-1] == "" else texts
    labels = labels[:-1] if labels and labels[-1] == "" else labels
    if len(texts) != len(labels):
        raise ValueError(f"{split}: {len(texts)} texts vs {len(labels)} labels")
    df = pd.DataFrame({"text": [t.strip() for t in texts], "label": [int(x) for x in labels]})
    if not df["label"].isin(LABEL2ID.values()).all():
        raise ValueError(f"{split}: unexpected label ids")
    return df


def load_tweeteval(root: str | Path) -> dict[str, pd.DataFrame]:
    return {split: load_tweeteval_split(root, split) for split in TWEETEVAL_SPLITS}


def load_gop_debate(
    path: str | Path,
    min_confidence: float = 0.0,
    deduplicate: bool = True,
) -> pd.DataFrame:
    """Load the CrowdFlower 'First GOP Debate' (Aug 2015) sentiment tweets.

    Returns ``text``, ``label``, ``confidence``, ``candidate`` and ``dedup_key``.
    Retweets/near-copies are collapsed (keeping the highest-confidence copy) when
    ``deduplicate`` is True; copies with conflicting labels are dropped entirely
    because their gold label is undefined.
    """
    from sentiment.preprocess import dedup_key

    raw = pd.read_csv(path)
    df = pd.DataFrame(
        {
            "text": raw["text"].astype(str),
            "label": raw["sentiment"].map(lambda s: LABEL2ID[normalize_label(s)]),
            "confidence": raw["sentiment_confidence"].astype(float),
            "candidate": raw["candidate"].fillna("Unknown"),
        }
    )
    df["dedup_key"] = df["text"].map(dedup_key)
    if deduplicate:
        n_labels = df.groupby("dedup_key")["label"].transform("nunique")
        df = df[n_labels == 1]
        df = df.sort_values("confidence", ascending=False).drop_duplicates("dedup_key")
        df = df.sort_index()
    df = df[df["confidence"] >= min_confidence]
    return df.reset_index(drop=True)
