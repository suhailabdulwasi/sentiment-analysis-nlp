# Data audit: duplicates, leakage and distribution shift

Reproduce: `PYTHONPATH=src python scripts/audit_data.py` → `reports/data_audit.json`.

## Sources
| Dataset | Source | Integrity |
|---|---|---|
| TweetEval sentiment | Official Cardiff NLP files `github.com/cardiffnlp/tweeteval/datasets/sentiment` (same data as HF `cardiffnlp/tweet_eval`) | Sizes match exactly: 45,615 / 2,000 / 12,284 |
| CrowdFlower "First GOP Debate Twitter Sentiment" (Aug 2015) | Mirror of the CrowdFlower *Data for Everyone* CSV (`input/Sentiment.csv` in `sanjay-raghu/sentiment-analysis-using-LSTM-keras`) | SHA-256 pinned: `83beb769…c61c`, 13,871 rows |

## Exact checks performed

1. **Exact within-split duplicates**: identical raw text strings inside a split.
2. **Normalised within-split duplicates**: key = `dedup_key(text)`: `@handle`→`@user`, URL removed,
   leading `RT @user:` removed, Unicode NFKC, case-folded, punctuation removed (hashtags kept),
   whitespace collapsed. Groups with >1 distinct label are counted as *conflicting*.
3. **Cross-split overlap (exact and normalised)**: for train→validation, train→test and
   validation→test, the number of rows in the later split whose key appears in the earlier
   split, and whether the labels agree.
4. **Label distribution shift**: χ² test of independence on 2×3 label-count tables and Cramér's V.
5. **Length shift**: Mann–Whitney U on whitespace token counts.
6. **Surface-feature shift**: % of tweets with URL, `@user`, hashtag, leading quote.
7. **Residual raw handles**: tweets still containing an `@name` other than `@user`.
8. **GOP benchmark**: label values, annotator-agreement (`sentiment_confidence`) distribution,
   retweet share, de-duplication with conflict removal, and normalised overlap with every TweetEval split.

## Results

### TweetEval duplicates and leakage
| Check | train | validation | test |
|---|---:|---:|---:|
| Exact redundant rows | 29 (3 conflicting groups) | 0 | 0 |
| Normalised redundant rows | 115 (23 conflicting groups) | 0 | 13 (3 conflicting groups) |

| Cross-split overlap | exact | normalised |
|---|---:|---:|
| train → validation | 0 | 10 rows (0.5%), 9 same label |
| train → test | **0** | **0** |
| validation → test | **0** | **0** |

**Conclusion:** there is no train/validation → test leakage. The 10 near-duplicates between train and
validation (0.5%) can shift validation accuracy by at most 0.5 points and are left in place so that
results stay comparable with published TweetEval numbers. The 23 conflicting near-duplicate groups in
train are direct evidence of annotation noise.

### Distribution shift (verified)
| | train | validation | test |
|---|---:|---:|---:|
| negative | 15.55% | 15.60% | **32.33%** |
| neutral | 45.32% | 43.45% | 48.33% |
| positive | 39.13% | 40.95% | **19.33%** |
| mean words | 19.24 | 19.44 | **14.86** |
| has hashtag | 18.7% | 17.6% | **39.3%** |
| starts with `"` | 20.5% | 19.9% | **1.1%** |
| has URL token | 0.19% | 0.15% | 3.07% |

| Comparison | label χ² | p | Cramér's V | length Mann–Whitney p |
|---|---:|---:|---:|---:|
| train vs validation | 3.1 | 0.21 | 0.006 | 0.045 |
| train vs test | 2,531.5 | < 1e-300 | 0.148 | < 1e-300 |
| validation vs test | 528.6 | 1.6e-115 | 0.136 | 3.1e-232 |

**Conclusion:** validation is drawn from the same distribution as train; **test is not** (it is the
SemEval-2017 test set, collected separately). Consequences:
* Expect validation → test drops of several points for every model (the classical baseline drops 7.3 accuracy points).
* Positive-heavy validation rewards predicting *positive*; the test set is negative-heavy. This is a
  plausible contributor to the original model's dominant error (neutral → negative) and to
  positive's low test precision.
* Accuracy is prior-sensitive; macro-F1 / macro-recall are more comparable across these splits.

### GOP debate benchmark
See [ELECTION_BENCHMARK.md](ELECTION_BENCHMARK.md). Normalised overlap with TweetEval: train 1 row, validation 0, test 0.
