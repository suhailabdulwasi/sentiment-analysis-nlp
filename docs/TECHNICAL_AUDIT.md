# Technical audit (Phase 1)

Audited on 2026-11-05 against `main` @ `9d1f452`, the Colab notebook *Sentiment Analysis Codes*
(73 cells, read in full incl. outputs) and the Hugging Face model card / usage in the notebook.
The Hugging Face Hub itself was **not reachable** from the audit environment (network policy),
so the hosted checkpoint was audited through the notebook code that created, uploaded and
re-loaded it (cells 16–27, 41–42), not by downloading it.

## Findings

| # | Severity | Finding | Evidence | Action |
|---|---|---|---|---|
| 1 | **High** | `metrics.json` (97.5% accuracy) describes the *old* TF-IDF/LinearSVC model on a 40-tweet random split of 197 unique texts — not the deployed transformer and not an external benchmark. Easy to mistake for the project's headline number. | `metrics.json` keys `test_rows: 40`, `unique_texts: 197` | Moved to `archive/ai4all_tfidf/` with an explanatory README. |
| 2 | **High** | The archived TF-IDF model predicted **"positive" for every input** in the notebook (10/10 test sentences, 8/8 single words). Its vocabulary (1,111 features from 157 training rows) lacked words such as *hate, terrible, bad, good*. The 97.5% figure therefore reflected a tiny, positive-heavy split. | Notebook cells 50–54 | Archived; not used by the app. |
| 3 | **High** | The transformer training/evaluation code that produced the deployed model existed only in a non-linear Colab notebook (cells run out of order, cells referencing undefined variables, e.g. cell 12 uses `classifier`, `true_labels`, `np` before definition). Not reproducible from the repo. | Notebook cells 12, 16–20 | New `src/sentiment/` package + config-driven training + generated Colab notebook. |
| 4 | Medium | Root `train.py`/`preprocessors.py` train the legacy TF-IDF model from `data/*.csv` files that are not in the repo; running them fails. | `train.py` `FILES` | Archived. |
| 5 | Medium | The TweetEval **test label distribution differs strongly from train/validation** (negative 32.3% vs 15.5%; χ² = 2,531, Cramér's V = 0.148) and test tweets are shorter (14.9 vs 19.2 words) with 2× more hashtags. Validation scores therefore over-estimate test scores, and decisions tuned to the validation class prior (e.g. a neutral-class weight) do not transfer. | `reports/data_audit.json` → `distribution_shift` | Documented; report test & validation separately; macro-F1 as primary metric. |
| 6 | Medium | The official TweetEval sentiment metric is **macro-averaged recall**, not macro-F1 or accuracy. The repo reported only accuracy/macro-F1, so numbers were not comparable with the literature. | Barbieri et al. 2020 | All reports now include macro-recall. |
| 7 | Medium | App inference used `max_length=512` while training used 128, and applied no tweet normalisation (raw `@handles`/URLs vs `@user` in training data). Harmless for most tweets, but a train/serve skew. | `app.py` | Shared `normalize_tweet()` + `max_length=128` in `sentiment.inference`. |
| 8 | Medium | App decoded predictions with `result["label"].lower()` — correct only because the checkpoint's `id2label` happens to be set. A checkpoint saved without it would emit `LABEL_0…` and break silently. | `app.py` | `normalize_label()` decodes via the checkpoint's own `id2label` and validates the label set. |
| 9 | Low | App showed only the arg-max label: no probabilities, no uncertainty, metrics hard-coded in the UI. | `app.py` | Probabilities, low-confidence warning, metrics loaded from `reports/model_card_metrics.json`. |
| 10 | Low | Batch analysis aborted on any bad row and required a column literally named `tweet_text`. | `app.py` | Per-row validation/errors, column picker, row cap, progress bar. |
| 11 | Low | No tests, no dependency pins for training, no lint config. | repo | 65 tests (unit + integration), `pyproject.toml`, `requirements-dev.txt`. |
| 12 | Info | Reported 70.41% / 70.44% test metrics are **verified**: recomputing from the notebook's confusion matrix gives accuracy 0.7041, macro-F1 0.7044 (macro-recall 0.7222). | `tests/test_metrics.py::test_metrics_match_sklearn_and_hand_values` | Kept. |
| 13 | Info | Label mapping is consistent across training (`id2label={0:negative,1:neutral,2:positive}`), TweetEval `mapping.txt`, and app decoding. | notebook cell 16, `mapping.txt` | Centralised in `sentiment.labels`. |
| 14 | Info | The neutral-weight experiment (cell 15) was correctly done on validation halves and correctly rejected (72.60 → 72.40 acc). | notebook cell 15 | Kept as a documented negative result. |
| 15 | Info | The TweetEval test set has been looked at repeatedly during the original work (cells 12, 20, 70–71). It remains usable for a *single* final comparison, but further iteration on it would be tuning on test. | notebook | Test evaluation is locked behind a frozen, validation-only selection (`test_lock.json`). |

## Environment of the original baseline
Colab, Tesla T4 (15.64 GB), Python 3.13, PyTorch + Transformers (versions not recorded in the notebook),
fp16, `distilbert/distilroberta-base`, lr 2e-5, 2 epochs, batch 16, max length 128, weight decay 0.01,
best epoch by validation macro-F1 (checkpoint-5702 = end of epoch 2).
