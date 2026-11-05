# AI4ALL Sentiment Analysis NLP

An interactive sentiment analysis application powered by a fine-tuned DistilRoBERTa transformer.

## Features
- Classify text as positive, negative, or neutral
- Analyze individual statements
- Upload CSV files for batch predictions
- Download prediction results
- Interactive Streamlit interface

## Model Performance

Fine-tuned on the TweetEval sentiment dataset.

| Metric | Result |
|---|---|
| Test accuracy | 70.41% |
| Test macro F1 | 70.44% |
| Test samples | 12,284 |
| Training samples | 45,615 |

These results are from TweetEval, not a separate election-specific evaluation.

## Model

[View the fine-tuned model on Hugging Face](https://huggingface.co/suhailabdulwasi/ai4all-distilroberta-sentiment)

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application downloads the model from Hugging Face.

## Project Background

This repository extends a collaborative AI4ALL Ignite sentiment analysis project originally developed using traditional machine learning methods, including TF-IDF and LinearSVC.

The transformer fine-tuning, evaluation, Hugging Face hosting, and updated Streamlit application are part of the independent extension.

## Limitations

- Election-specific performance has not been independently evaluated.
- Sarcasm, ambiguous wording, and domain shifts may affect predictions.
- Sentiment predictions do not establish political affiliation or factual accuracy.
