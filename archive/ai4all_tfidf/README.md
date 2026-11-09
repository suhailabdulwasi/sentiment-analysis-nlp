# Archived: original AI4ALL Ignite TF-IDF model (collaborative work)

These files come from the **collaborative AI4ALL Ignite project** that this repository grew out of.
They are kept for attribution and history only; nothing in the current app or pipeline uses them.

| File | What it is |
|---|---|
| `train.py` | TF-IDF (1–2-grams, 2,000 features) + LinearSVC with candidate-name anonymisation and negation tagging, trained on a small election-statement CSV (`data/*.csv`, not in this repo). |
| `preprocessors.py` | `CandidateAnonymizer` and `NegationTagger` scikit-learn transformers. |
| `metrics.json` | **Read the caveat.** 97.5% accuracy on a *40-example* random split of 197 unique texts. |

## Why the 97.5% figure should not be quoted
* The test split had 40 rows (3 negative, 11 neutral, 26 positive), so the 95% CI spans roughly 87–100%.
* The saved model predicted **positive for every input** checked in the Colab notebook ("I hate this
  candidate!", "That debate was terrible." etc.). Its vocabulary contained 1,111 features and lacked
  common sentiment words. The score reflects a tiny, positive-skewed split, not real ability.
* On TweetEval, a TF-IDF + LinearSVC retrained by the original author scored **57.38%** test accuracy.
  A validation-tuned word+char TF-IDF baseline in this repo (`scripts/train_baseline.py`) scores **63.16%**.
