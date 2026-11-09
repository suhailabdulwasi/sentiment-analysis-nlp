"""Text normalisation shared by training and inference.

TweetEval already replaces user handles with ``@user``. Applying the same
normalisation at inference time keeps the app's input distribution consistent
with training (raw handles and URLs are otherwise out-of-distribution tokens).
"""

from __future__ import annotations

import re
import unicodedata

_USER_RE = re.compile(r"(?<!\w)@\w{1,30}")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def normalize_tweet(text: object) -> str:
    """Normalise a tweet the way TweetEval / Cardiff NLP models expect.

    * ``@handle`` -> ``@user``
    * URLs -> ``http``
    * HTML entities that commonly survive scraping (``&amp;`` etc.) decoded
    * whitespace collapsed
    Non-string input (``None``/NaN) becomes an empty string.
    """
    if text is None or (isinstance(text, float) and text != text):
        return ""
    s = str(text)
    s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    s = _USER_RE.sub("@user", s)
    s = _URL_RE.sub("http", s)
    return _WS_RE.sub(" ", s).strip()


def dedup_key(text: object) -> str:
    """Aggressive key for duplicate / leakage detection (not used for modelling).

    Normalises handles and URLs, strips a leading retweet marker, removes
    punctuation and case so that trivially different copies collide.
    """
    s = normalize_tweet(text)
    s = re.sub(r"^rt\s+@user:?\s*", "", s, flags=re.IGNORECASE)
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"\bhttp\b", " ", s)
    s = re.sub(r"[^\w\s#]", " ", s)
    return _WS_RE.sub(" ", s).strip()
