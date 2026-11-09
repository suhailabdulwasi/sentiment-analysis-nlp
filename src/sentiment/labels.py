"""Single source of truth for the 3-class label scheme.

TweetEval sentiment uses 0=negative, 1=neutral, 2=positive (see the official
``mapping.txt``). Every training, evaluation and inference path imports these
constants instead of re-declaring them.
"""

from __future__ import annotations

LABELS: tuple[str, ...] = ("negative", "neutral", "positive")
LABEL2ID: dict[str, int] = {name: i for i, name in enumerate(LABELS)}
ID2LABEL: dict[int, str] = dict(enumerate(LABELS))

_ALIASES = {
    "neg": "negative",
    "negative": "negative",
    "neu": "neutral",
    "neutral": "neutral",
    "pos": "positive",
    "positive": "positive",
}


def normalize_label(label: str | int, id2label: dict[int, str] | None = None) -> str:
    """Map any label representation to one of :data:`LABELS`.

    Accepts integer ids, ``"LABEL_0"``-style names produced by a model whose
    config lacks ``id2label``, and case/abbreviation variants ("Positive", "POS").
    ``id2label`` lets callers pass a model's own mapping so that a checkpoint
    with a *different* label order is still decoded correctly.
    """
    mapping = id2label or ID2LABEL
    if isinstance(label, bool):  # bool is an int subclass; reject explicitly
        raise ValueError(f"Invalid label: {label!r}")
    if isinstance(label, int):
        if label not in mapping:
            raise ValueError(f"Unknown label id: {label}")
        return normalize_label(mapping[label])
    text = str(label).strip().lower()
    if text.startswith("label_") and text[6:].isdigit():
        return normalize_label(int(text[6:]), id2label)
    if text not in _ALIASES:
        raise ValueError(f"Unknown label: {label!r}")
    return _ALIASES[text]
