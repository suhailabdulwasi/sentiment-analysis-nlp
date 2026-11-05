"""Model-agnostic inference wrapper used by the Streamlit app and scripts.

The heavy ``transformers`` import is deferred so the module (and the
formatting/validation helpers) can be imported and unit-tested without it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from sentiment.labels import LABELS, normalize_label
from sentiment.preprocess import normalize_tweet

DEFAULT_MODEL_ID = "suhailabdulwasi/ai4all-distilroberta-sentiment"
MAX_CHARS = 2_000
MAX_BATCH_ROWS = 5_000


@dataclass(frozen=True)
class Prediction:
    text: str
    label: str | None
    confidence: float | None
    probabilities: dict[str, float]
    error: str | None = None

    def as_row(self) -> dict:
        row = {
            "predicted_sentiment": self.label,
            "confidence": None if self.confidence is None else round(self.confidence, 4),
        }
        row.update({f"p_{k}": round(v, 4) for k, v in self.probabilities.items()})
        row["error"] = self.error
        return row


def validate_text(text: object) -> str | None:
    """Return an error message for unusable input, else None."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return "empty input"
    s = str(text)
    if not s.strip():
        return "empty input"
    if len(s) > MAX_CHARS:
        return f"input longer than {MAX_CHARS} characters"
    return None


def probs_to_predictions(
    texts: Sequence[str], probs: np.ndarray, label_order: Sequence[str] = LABELS
) -> list[Prediction]:
    """Convert an (n, 3) probability matrix into :class:`Prediction` objects."""
    probs = np.asarray(probs, dtype=float)
    if probs.shape != (len(texts), len(label_order)):
        raise ValueError(f"probs shape {probs.shape} != ({len(texts)}, {len(label_order)})")
    names = [normalize_label(n) for n in label_order]
    out = []
    for text, row in zip(texts, probs, strict=True):
        dist = {name: float(row[i]) for i, name in enumerate(names)}
        dist = {k: dist[k] for k in LABELS}  # canonical order
        best = max(dist, key=dist.get)
        out.append(Prediction(text=text, label=best, confidence=dist[best], probabilities=dist))
    return out


class SentimentClassifier:
    """Batch classifier returning full probability distributions.

    ``predict_proba_fn`` may be injected (tests, alternative backends); by
    default a Hugging Face model is loaded from ``model_id``.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        max_length: int = 128,
        device: str | None = None,
        predict_proba_fn: Callable[[list[str]], np.ndarray] | None = None,
        label_order: Sequence[str] = LABELS,
    ) -> None:
        self.model_id = model_id
        self.max_length = max_length
        self.label_order = tuple(label_order)
        if predict_proba_fn is not None:
            self._predict_proba = predict_proba_fn
        else:
            self._load_hf(model_id, device)

    def _load_hf(self, model_id: str, device: str | None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device)
        self.model.eval()
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        # Decode via the checkpoint's own mapping: never assume an order.
        self.label_order = tuple(normalize_label(i, id2label) for i in sorted(id2label))
        if sorted(self.label_order) != sorted(LABELS):
            raise ValueError(f"model labels {self.label_order} are not {LABELS}")
        self._predict_proba = self._hf_proba

    def _hf_proba(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt"
        ).to(self.device)
        with self._torch.inference_mode():
            logits = self.model(**enc).logits
        return self._torch.softmax(logits.float(), dim=-1).cpu().numpy()

    def predict(self, texts: Sequence[object], batch_size: int = 32) -> list[Prediction]:
        """Classify texts. Invalid rows get ``label=None`` and an ``error`` message
        instead of aborting the whole batch."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        results: list[Prediction | None] = [None] * len(texts)
        valid_idx, valid_text = [], []
        for i, t in enumerate(texts):
            err = validate_text(t)
            if err:
                results[i] = Prediction(str(t) if t is not None else "", None, None, {}, err)
            else:
                valid_idx.append(i)
                valid_text.append(str(t))
        for start in range(0, len(valid_text), batch_size):
            chunk = valid_text[start : start + batch_size]
            probs = self._predict_proba([normalize_tweet(t) for t in chunk])
            for j, pred in enumerate(probs_to_predictions(chunk, probs, self.label_order)):
                results[valid_idx[start + j]] = pred
        return results  # type: ignore[return-value]
