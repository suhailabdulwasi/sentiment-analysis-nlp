import pandas as pd
import pytest

from sentiment.data import GOP_HIGH_AGREEMENT_THRESHOLD, load_gop_debate, load_tweeteval, load_tweeteval_split


def test_load_tweeteval_reads_quotes_verbatim(tweeteval_dir):
    splits = load_tweeteval(tweeteval_dir)
    assert {k: len(v) for k, v in splits.items()} == {"train": 6, "validation": 3, "test": 3}
    assert splits["train"]["text"][1] == '"quoted, text" ok'
    assert splits["validation"]["label"].tolist() == [1, 0, 2]


def test_load_tweeteval_misaligned(tweeteval_dir):
    (tweeteval_dir / "val_labels.txt").write_text("1\n0\n")
    with pytest.raises(ValueError, match="texts vs"):
        load_tweeteval_split(tweeteval_dir, "validation")


def test_load_tweeteval_bad_label(tweeteval_dir):
    (tweeteval_dir / "test_labels.txt").write_text("0\n1\n5\n")
    with pytest.raises(ValueError, match="unexpected label"):
        load_tweeteval_split(tweeteval_dir, "test")


def test_bad_split_name(tweeteval_dir):
    with pytest.raises(ValueError):
        load_tweeteval_split(tweeteval_dir, "dev")


def _gop_csv(path):
    pd.DataFrame({
        "text": ["RT @a: Trump rocks", "Trump rocks", "Jeb is boring", "jeb is boring!", "Cruz spoke", "x y"],
        "sentiment": ["Positive", "Positive", "Negative", "Neutral", "Neutral", "Negative"],
        "sentiment_confidence": [0.65, 1.0, 1.0, 1.0, 0.4, 1.0],
        "candidate": ["Donald Trump"] * 2 + ["Jeb Bush"] * 2 + ["Ted Cruz", None],
    }).to_csv(path, index=False)


def test_gop_dedup_and_conflicts(tmp_path):
    p = tmp_path / "s.csv"
    _gop_csv(p)
    df = load_gop_debate(p)
    # retweet copy collapsed keeping the higher-confidence row; conflicting Jeb pair dropped
    assert sorted(df["text"]) == ["Cruz spoke", "Trump rocks", "x y"]
    assert df.set_index("text").loc["Trump rocks", "confidence"] == 1.0
    assert df.set_index("text").loc["x y", "candidate"] == "Unknown"


def test_gop_threshold(tmp_path):
    p = tmp_path / "s.csv"
    _gop_csv(p)
    df = load_gop_debate(p, min_confidence=GOP_HIGH_AGREEMENT_THRESHOLD)
    assert sorted(df["text"]) == ["Trump rocks", "x y"]
    assert GOP_HIGH_AGREEMENT_THRESHOLD == 1.0  # pre-registered; changing it must be deliberate
