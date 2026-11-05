import pytest

from sentiment.labels import ID2LABEL, LABEL2ID, LABELS, normalize_label


def test_tweeteval_order():
    # Official TweetEval mapping.txt: 0 negative, 1 neutral, 2 positive
    assert LABELS == ("negative", "neutral", "positive")
    assert LABEL2ID == {"negative": 0, "neutral": 1, "positive": 2}
    assert all(LABEL2ID[ID2LABEL[i]] == i for i in ID2LABEL)


@pytest.mark.parametrize("raw,expected", [
    (0, "negative"), (2, "positive"), ("Positive", "positive"), (" NEUTRAL ", "neutral"),
    ("neg", "negative"), ("LABEL_1", "neutral"), ("label_2", "positive"),
])
def test_normalize_label(raw, expected):
    assert normalize_label(raw) == expected


def test_normalize_uses_model_mapping():
    # A checkpoint with a different label order must decode by its own id2label
    other = {0: "positive", 1: "negative", 2: "neutral"}
    assert normalize_label(0, other) == "positive"
    assert normalize_label("LABEL_1", other) == "negative"


@pytest.mark.parametrize("bad", [3, -1, "mixed", "", True, "LABEL_9"])
def test_normalize_rejects_unknown(bad):
    with pytest.raises(ValueError):
        normalize_label(bad)
