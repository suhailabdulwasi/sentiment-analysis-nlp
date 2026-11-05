from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tweeteval_dir(tmp_path):
    """A miniature TweetEval directory in the official line-aligned format."""
    d = tmp_path / "tweeteval"
    d.mkdir()
    data = {
        "train": (['I love it', '"quoted, text" ok', 'bad day', 'meh', 'great', 'awful'], [2, 1, 0, 1, 2, 0]),
        "val": (["fine", "terrible", "wonderful"], [1, 0, 2]),
        "test": (["so sad", "news today", "yay"], [0, 1, 2]),
    }
    for prefix, (texts, labels) in data.items():
        (d / f"{prefix}_text.txt").write_text("\n".join(texts) + "\n")
        (d / f"{prefix}_labels.txt").write_text("\n".join(map(str, labels)) + "\n")
    return d
