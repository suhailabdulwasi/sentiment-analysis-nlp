import re
from sklearn.base import BaseEstimator, TransformerMixin


# Candidate names to anonymize
CANDIDATE_PATTERNS = [
    r"\bdonald\s+trump\b",
    r"\bjoe\s+biden\b",
    r"\bkamala\s+harris\b",
    r"\btrump\b",
    r"\bbiden\b",
    r"\bkamala\b",
    r"\bharris\b",
]


class CandidateAnonymizer(BaseEstimator, TransformerMixin):
    """
    Replaces candidate names with a common token to reduce
    the model's dependence on specific political names.
    """

    def __init__(self, token="CAND"):
        self.token = token

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        pattern = re.compile(
            "|".join(CANDIDATE_PATTERNS),
            flags=re.IGNORECASE
        )

        return [
            pattern.sub(self.token, str(text))
            for text in X
        ]


# Common negation expressions
NEG_TRIGGERS = {
    "not", "no", "never", "cannot",
    "can't", "won't", "don't", "didn't",
    "isn't", "wasn't", "aren't", "weren't",
    "haven't", "hasn't", "couldn't",
    "shouldn't", "wouldn't"
}


class NegationTagger(BaseEstimator, TransformerMixin):
    """
    Marks words following negation expressions.

    Example:
        "not very good"
        becomes
        "not very_NEG good_NEG"
    """

    def __init__(self, window=3):
        self.window = window

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        results = []

        for text in X:
            tokens = re.findall(
                r"\b[\w']+\b",
                str(text).lower()
            )

            negated = set()

            for i, token in enumerate(tokens):
                if token in NEG_TRIGGERS:
                    end = min(
                        i + self.window + 1,
                        len(tokens)
                    )

                    for j in range(i + 1, end):
                        negated.add(j)

            processed = [
                token + "_NEG" if i in negated else token
                for i, token in enumerate(tokens)
            ]

            results.append(" ".join(processed))

        return results
