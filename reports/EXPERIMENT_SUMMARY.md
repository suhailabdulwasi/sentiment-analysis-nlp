# Completed transformer experiment — 2026-11-08

## Model selection
The model was selected using TweetEval validation macro-F1 before the held-out test evaluation. The selected run was `s1_twitter_roberta_sentiment_latest`, a fine-tuned Cardiff Twitter-RoBERTa sentiment checkpoint.

| Metric | Validation | TweetEval held-out test |
|---|---:|---:|
| Accuracy | 77.05% | 71.96% |
| Macro-F1 | 75.71% | 72.03% |
| Macro-recall | — | 73.64% |

## Paired comparison on TweetEval test (n=12,284)
| Metric | Original DistilRoBERTa | Selected Cardiff model |
|---|---:|---:|
| Accuracy | 70.48% | 71.96% |
| Macro-F1 | 70.52% | 72.03% |

Macro-F1 difference: **+1.51 percentage points**; paired bootstrap 95% confidence interval **+0.81 to +2.21 points**.

## Election-domain transfer
On 9,222 deduplicated GOP debate tweets, the original checkpoint achieved **59.62% accuracy / 57.07% macro-F1** and the selected Cardiff checkpoint achieved **60.91% accuracy / 59.09% macro-F1**.

## Reproducibility and serving caveat
Source: Kaggle Version 2, `results_bundle.zip`, with `selection.json`, `final_results.json`, experiment logs and diagnostic figures. The archive contains reports but **does not include the selected fine-tuned model weights**. The live Streamlit analyzer therefore uses the separately published `suhailabdulwasi/ai4all-distilroberta-sentiment` checkpoint, **not** the research-selected Cardiff checkpoint. These metrics must not be represented as live-app measurements. The public deployment is not yet verified.
