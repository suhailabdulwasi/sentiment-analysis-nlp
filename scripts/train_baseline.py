"""Classical TF-IDF baseline, tuned on the validation split only.

Usage: python scripts/train_baseline.py [--gop data/raw/gop/Sentiment.csv]
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sentiment.data import GOP_HIGH_AGREEMENT_THRESHOLD, download_tweeteval, load_gop_debate, load_tweeteval
from sentiment.final_eval import summarize
from sentiment.metrics import classification_metrics
from sentiment.preprocess import normalize_tweet


def featurize(train, others, char: bool):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    mats = [word.fit_transform(train)]
    rest = [[word.transform(o)] for o in others]
    if char:
        ch = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                             max_features=300_000)
        mats.append(ch.fit_transform(train))
        for r, o in zip(rest, others, strict=True):
            r.append(ch.transform(o))
    return hstack(mats).tocsr(), [hstack(r).tocsr() for r in rest]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--gop", default="data/raw/gop/Sentiment.csv")
    p.add_argument("--out", default="reports/baseline_results.json")
    a = p.parse_args()
    splits = load_tweeteval(download_tweeteval(a.data_dir))
    tr, va, te = (splits[s] for s in ("train", "validation", "test"))
    norm = lambda s: s.map(normalize_tweet)  # noqa: E731
    y_tr, y_va = tr["label"].values, va["label"].values

    log = []
    best = None
    for char in (False, True):
        Xtr, (Xva,) = featurize(norm(tr["text"]), [norm(va["text"])], char)
        for algo, C, cw in itertools.product(("logreg", "linsvc"), (0.1, 0.3, 1.0, 3.0), (None, "balanced")):
            t0 = time.time()
            clf = (LogisticRegression(C=C, max_iter=3000, class_weight=cw) if algo == "logreg"
                   else LinearSVC(C=C, class_weight=cw))
            clf.fit(Xtr, y_tr)
            m = classification_metrics(y_va, clf.predict(Xva))
            row = {"features": "word1-2" + ("+char2-5" if char else ""), "algo": algo, "C": C,
                   "class_weight": cw or "none", "val_accuracy": round(m["accuracy"], 4),
                   "val_macro_f1": round(m["macro_f1"], 4), "seconds": round(time.time() - t0, 1)}
            log.append(row)
            print(row)
            if best is None or m["macro_f1"] > best[0]["val_macro_f1"]:
                best = (row, char, algo, C, cw)

    row, char, algo, C, cw = best
    print("Selected on validation:", row)
    others = [norm(va["text"]), norm(te["text"])]
    gop_clean = gop_full = None
    if Path(a.gop).exists():
        gop_full = load_gop_debate(a.gop)
        gop_clean = gop_full[gop_full["confidence"] >= GOP_HIGH_AGREEMENT_THRESHOLD]
        others += [norm(gop_full["text"]), norm(gop_clean["text"])]
    Xtr, Xs = featurize(norm(tr["text"]), others, char)
    clf = (LogisticRegression(C=C, max_iter=3000, class_weight=cw) if algo == "logreg"
           else LinearSVC(C=C, class_weight=cw))
    clf.fit(Xtr, y_tr)

    def probs_of(X):
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(X)
        d = clf.decision_function(X)  # pseudo-probabilities (for argmax/ECE display only)
        e = np.exp(d - d.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    res = {"search": log, "selected": row,
           "validation": summarize("tweeteval_validation", y_va, probs_of(Xs[0])),
           "test": summarize("tweeteval_test", te["label"].values, probs_of(Xs[1]))}
    if gop_full is not None:
        res["gop_full_dedup"] = summarize("gop_full_dedup", gop_full["label"].values, probs_of(Xs[2]))
        res["gop_conf1"] = summarize("gop_conf1", gop_clean["label"].values, probs_of(Xs[3]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    for k in ("validation", "test", "gop_full_dedup", "gop_conf1"):
        if k in res:
            r = res[k]
            print(f"{k:16s} acc={r['accuracy']:.4f} {r['accuracy_ci95']} macroF1={r['macro_f1']:.4f} "
                  f"{r['macro_f1_ci95']} macroRecall={r['macro_recall']:.4f}")


if __name__ == "__main__":
    main()
