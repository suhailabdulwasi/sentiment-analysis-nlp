import numpy as np
import pytest

from sentiment.inference import (
    MAX_CHARS,
    Prediction,
    SentimentClassifier,
    probs_to_predictions,
    validate_text,
)


def fake(texts):
    # deterministic: positive if "good" in text, negative if "bad", else neutral
    out = []
    for t in texts:
        p = [0.1, 0.8, 0.1]
        if "good" in t:
            p = [0.05, 0.15, 0.8]
        if "bad" in t:
            p = [0.7, 0.2, 0.1]
        out.append(p)
    return np.array(out)


@pytest.fixture
def clf():
    calls = []

    def spy(texts):
        calls.append(list(texts))
        return fake(texts)

    c = SentimentClassifier(predict_proba_fn=spy)
    c.calls = calls
    return c


def test_prediction_format(clf):
    (p,) = clf.predict(["a good day"])
    assert isinstance(p, Prediction)
    assert p.label == "positive" and p.confidence == pytest.approx(0.8)
    assert list(p.probabilities) == ["negative", "neutral", "positive"]
    assert sum(p.probabilities.values()) == pytest.approx(1.0)
    row = p.as_row()
    assert set(row) == {"predicted_sentiment", "confidence", "p_negative", "p_neutral", "p_positive", "error"}
    assert row["error"] is None


def test_batching_preserves_order_and_sizes(clf):
    texts = [f"good {i}" if i % 3 == 0 else f"bad {i}" for i in range(10)]
    preds = clf.predict(texts, batch_size=4)
    assert [len(c) for c in clf.calls] == [4, 4, 2]
    assert [p.text for p in preds] == texts
    assert [p.label for p in preds] == ["positive" if i % 3 == 0 else "negative" for i in range(10)]


def test_inputs_are_normalised_before_model(clf):
    clf.predict(["@Someone good https://x.y/z"])
    assert clf.calls[0] == ["@user good http"]


def test_invalid_rows_do_not_abort_batch(clf):
    preds = clf.predict(["good", "", None, float("nan"), "x" * (MAX_CHARS + 1), "bad"])
    assert [p.label for p in preds] == ["positive", None, None, None, None, "negative"]
    assert preds[1].error == "empty input" and "longer" in preds[4].error
    assert preds[2].as_row()["confidence"] is None
    assert len(clf.calls) == 1 and clf.calls[0] == ["good", "bad"]


def test_all_invalid_makes_no_model_call(clf):
    assert all(p.error for p in clf.predict(["  ", None]))
    assert clf.calls == []


def test_bad_batch_size(clf):
    with pytest.raises(ValueError):
        clf.predict(["x"], batch_size=0)


def test_label_order_from_model_is_respected():
    # model outputs in (positive, negative, neutral) order
    p = probs_to_predictions(["t"], np.array([[0.7, 0.2, 0.1]]), ("positive", "negative", "neutral"))[0]
    assert p.label == "positive" and p.probabilities["negative"] == pytest.approx(0.2)


def test_probs_shape_checked():
    with pytest.raises(ValueError):
        probs_to_predictions(["a", "b"], np.zeros((1, 3)))


@pytest.mark.parametrize("t,ok", [("hi", True), ("   ", False), (None, False), (0, True)])
def test_validate_text(t, ok):
    assert (validate_text(t) is None) == ok
