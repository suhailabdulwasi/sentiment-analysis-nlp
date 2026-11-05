# Experiment log

Selection metric: **validation macro-F1** (TweetEval validation, n = 2,000; 95% CI half-width ≈ ±2 points,
so differences < ~1 point between single runs are not meaningful). Accuracy and macro-recall (the
official TweetEval metric) are secondary. Test-set numbers appear only for (a) previously published
results and (b) the single, locked final evaluation.

Status legend: ✅ completed and verified · ⏳ queued in the Colab notebook (not yet run) · ❌ rejected.

## Completed

| ID | Architecture | Config | Val acc | Val macro-F1 | Test acc | Test macro-F1 | Hardware | Status / interpretation |
|---|---|---|---:|---:|---:|---:|---|---|
| P1 | DistilRoBERTa (deployed) | lr 2e-5, 2 ep, bs 16, len 128 | 72.60\* | 70.85\* | 70.41 | 70.44 | Colab T4 | ✅ Original model. \*Validation on a random 1,000-tweet half (notebook cell 15). |
| P2 | P1 + neutral-logit weight 0.95 | weight tuned on val half A | 72.40\* | 70.71\* | – | – | Colab T4 | ❌ No gain on val half B. Not deployed (correct decision). |
| P3 | TF-IDF + LinearSVC (author's retrain) | 1–2-grams, 20k feats, balanced | – | – | 57.38 | – | Colab CPU | ✅ Original classical reference. Not tuned on validation. |
| B1 | TF-IDF word1–2 + char2–5 + LinearSVC | C = 0.1, balanced; best of 32 configs on val | 70.45 | 67.97 | 63.16 | 62.62 | CPU (4 vCPU), 2.4 s fit | ✅ Retained as the classical baseline. Validation-only grid: 2 feature sets × {LogReg, LinearSVC} × C ∈ {0.1, 0.3, 1, 3} × {none, balanced}. Char n-grams add about 3 F1 points; class balancing about 2. Test evaluated once after selection. |

B1 detail (95% bootstrap CIs, `reports/baseline_results.json`): test accuracy [62.25, 64.03], macro-F1 [61.68, 63.54], macro-recall 63.28.
Hypothesis "a properly tuned classical model closes most of the gap to transformers" was **rejected**:
B1 is +5.8 points over P3 but still 7.3 accuracy points below P1 on test.

## Queued for GPU (`notebooks/colab_t4_experiments.ipynb`)
All runs: seed 42, AdamW, linear schedule, 6% warmup, wd 0.01, fp16, max len 128, eval every ½ epoch,
early stopping (patience 1 epoch) on validation macro-F1, best checkpoint restored.

| ID | Stage | Architecture | LR | Max epochs | Hypothesis |
|---|---|---|---:|---:|---|
| s0 | reproduce | distilroberta-base | 2e-5 | 2 (bs 16, no warmup) | Pipeline reproduces P1 (sanity check) |
| s1 ×6 | screen | distilroberta, roberta-base, twitter-roberta-base-2022-154m, bertweet-base, deberta-v3-base, twitter-roberta-base-sentiment-latest | 2e-5 (1e-5 last) | 4 (2 last) | See `configs/experiments.yaml` |
| s2 ×6 | lr_sweep | top-2 distinct s1 architectures | 1e-5, 3e-5, + class weights | 4 | LR sensitivity; inverse-frequency weights for the 15.5% negative class |
| s3 ×2 | large | large versions of the best base families (e.g. bertweet-large, twitter-roberta-large-2022) | 1e-5 | 3 (bs 16×2) | Scale adds ≥ 1 F1 point |
| s4 ×2 | seeds | best config, seeds 13 & 2024 | – | – | Seed variance (mean ± sd reported) |
| final | ensemble | top-k distinct architectures (k ≤ 4), probability averaging | – | – | Accepted only if +0.5 pt val macro-F1 with paired-bootstrap CI > 0 |

Each run writes `result.json` (config, environment, GPU, duration, peak memory, epochs run, full
log history, validation metrics and ECE) and `val_probs.npy`. `select()` regenerates
`experiment_log.csv/.md` from these files, so this table is filled from records, never by hand.

## Approaches considered and not queued (with reasons)
* **Domain-adaptive pre-training**: already provided by twitter-roberta-2022 and BERTweet (154M–850M
  tweets); repeating it on a T4 would cost days for a smaller corpus.
* **Pseudo-labelling on the test inputs**: excluded on integrity grounds.
* **Prior-shift correction (EM re-weighting on unlabelled test inputs)**: it is label-free and would
  likely help accuracy given the documented 15% → 32% negative shift. It is transductive, though, and
  should be reported only as a separate, clearly labelled analysis. Recommended next step, not implemented.
* **Back-translation / EDA augmentation**: small or negative gains are typical with 45k training examples.
  Low priority.
