"""Streamlit app: tweet / election-statement sentiment with calibrated transparency.

Run: streamlit run app.py
Environment:
    SENTIMENT_MODEL_ID  Hugging Face id or local path (default: production model)
    SENTIMENT_FAKE_MODEL=1  deterministic keyword backend for tests/offline demos
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sentiment.dashboard import held_out_table, load_experiment_log, load_json  # noqa: E402
from sentiment.inference import (  # noqa: E402
    DEFAULT_MODEL_ID,
    MAX_BATCH_ROWS,
    SentimentClassifier,
    validate_text,
)
from sentiment.labels import LABELS  # noqa: E402

MODEL_ID = os.environ.get("SENTIMENT_MODEL_ID", DEFAULT_MODEL_ID)
METRICS_PATH = ROOT / "reports" / "model_card_metrics.json"
EXPERIMENTS_DIR = Path(os.environ.get("SENTIMENT_REPORTS_DIR", ROOT / "reports" / "experiments"))
LABEL_STYLE = {"negative": ("🔴", "error"), "neutral": ("⚪", "info"), "positive": ("🟢", "success")}

st.set_page_config(page_title="Election Sentiment Analyzer", page_icon="🗳️", layout="centered")


def _fake_proba(texts: list[str]) -> np.ndarray:
    """Offline keyword backend (tests / demos only — not a real model)."""
    out = []
    for t in texts:
        t = t.lower()
        pos = sum(w in t for w in ("love", "great", "amazing", "win", "good"))
        neg = sum(w in t for w in ("hate", "terrible", "awful", "disaster", "bad"))
        logits = np.array([neg, 0.5, pos], dtype=float)
        e = np.exp(logits - logits.max())
        out.append(e / e.sum())
    return np.array(out)


@st.cache_resource(show_spinner="Loading model…")
def load_classifier(model_id: str) -> SentimentClassifier:
    if os.environ.get("SENTIMENT_FAKE_MODEL") == "1":
        return SentimentClassifier(model_id="fake-keyword-model", predict_proba_fn=_fake_proba)
    return SentimentClassifier(model_id)


def load_model_card() -> dict:
    try:
        return json.loads(METRICS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


card = load_model_card()
LOW_CONF = float(card.get("low_confidence_threshold", 0.6))

st.title("🗳️ Election Sentiment Analyzer")
st.write("Classifies the **overall sentiment** of a tweet or short statement as negative, neutral or positive.")
st.caption("Live predictions use the fine-tuned Cardiff Twitter-RoBERTa checkpoint. Its held-out TweetEval test accuracy is 71.87% and macro-F1 is 72.02%.")
st.caption(
    "Sentiment ≠ stance: a negative tweet may still support a candidate (e.g. attacking an opponent). "
    "The model does not judge political affiliation, factual accuracy or candidate support."
)

try:
    clf = load_classifier(MODEL_ID)
except Exception as err:  # network, missing weights, bad id
    st.error(f"Could not load model `{MODEL_ID}`: {type(err).__name__}: {err}")
    st.info("Check your internet connection or set SENTIMENT_MODEL_ID to a local checkpoint directory.")
    st.stop()

tab_single, tab_batch, tab_compare, tab_info = st.tabs(
    ["Analyze text", "Batch CSV", "Model comparison", "Model information"]
)

with tab_single:
    text = st.text_area(
        "Tweet or statement", placeholder="I really liked the candidate's speech!", height=120, max_chars=2000
    )
    if st.button("Analyze sentiment", type="primary"):
        err = validate_text(text)
        if err:
            st.warning(f"Please enter some text ({err}).")
        else:
            pred = clf.predict([text])[0]
            icon, kind = LABEL_STYLE[pred.label]
            getattr(st, kind)(f"{icon} **{pred.label.capitalize()}** — model probability {pred.confidence:.0%}")
            st.bar_chart(
                pd.DataFrame({"probability": [pred.probabilities[k] for k in LABELS]},
                             index=[k.capitalize() for k in LABELS]),
                horizontal=True,
            )
            if pred.confidence < LOW_CONF:
                st.warning(
                    f"Low confidence (< {LOW_CONF:.0%}). The text may be ambiguous, mixed, sarcastic or "
                    "unlike the training data — treat this prediction with caution."
                )
            st.caption(
                "Probabilities are the model's softmax outputs, not guarantees. Transformer classifiers are "
                "often over-confident; see the calibration notes under *Model information*."
            )

with tab_batch:
    st.write(f"Upload a CSV (max {MAX_BATCH_ROWS:,} rows). Choose the column that contains the text.")
    upload = st.file_uploader("CSV file", type=["csv"])
    if upload is not None:
        try:
            df = pd.read_csv(upload)
        except Exception as err:
            st.error(f"Could not read CSV: {err}")
            st.stop()
        if df.empty:
            st.warning("The CSV has no rows.")
        else:
            text_cols = list(df.columns)
            default = text_cols.index("tweet_text") if "tweet_text" in text_cols else 0
            col = st.selectbox("Text column", text_cols, index=default)
            if len(df) > MAX_BATCH_ROWS:
                st.warning(f"Only the first {MAX_BATCH_ROWS:,} of {len(df):,} rows will be analysed.")
                df = df.head(MAX_BATCH_ROWS)
            if st.button("Run batch analysis", type="primary"):
                texts = df[col].tolist()
                preds, bar = [], st.progress(0.0, text="Classifying…")
                step = 64
                for i in range(0, len(texts), step):
                    preds.extend(clf.predict(texts[i : i + step]))
                    bar.progress(min(1.0, (i + step) / len(texts)), text=f"{min(i + step, len(texts))}/{len(texts)}")
                bar.empty()
                result = pd.concat([df.reset_index(drop=True), pd.DataFrame([p.as_row() for p in preds])], axis=1)
                n_err = int(result["error"].notna().sum())
                ok = result[result["error"].isna()]
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows analysed", len(ok))
                c2.metric("Skipped (invalid)", n_err)
                c3.metric("Low confidence", int((ok["confidence"] < LOW_CONF).sum()))
                if len(ok):
                    st.bar_chart(ok["predicted_sentiment"].value_counts().reindex(list(LABELS), fill_value=0))
                st.dataframe(result, use_container_width=True)
                st.download_button(
                    "Download predictions (CSV)", result.to_csv(index=False), "sentiment_predictions.csv", "text/csv"
                )

with tab_compare:
    st.subheader("Model comparison")
    st.info("Research results are from a separately evaluated Cardiff Twitter-RoBERTa checkpoint. The live analyzer above uses the published DistilRoBERTa model; do not interpret the research model\u2019s scores as live-app performance.")
    log = load_experiment_log(EXPERIMENTS_DIR)
    if log is None:
        st.info("No experiment results found yet. Run the GPU notebook and import its results with "
                "`python scripts/ingest_results.py results_bundle.zip`.")
    else:
        st.caption(
            "Validation set (TweetEval, n = 2,000) — used for model selection only. "
            "Differences below ~1–2 points are within noise (95% CI half-width ≈ 2 points)."
        )
        best = log.groupby("model", as_index=False)["val_macro_f1"].max().sort_values("val_macro_f1")
        st.bar_chart(best.set_index("model")["val_macro_f1"], horizontal=True)
        st.dataframe(log, use_container_width=True, hide_index=True)
        selection = load_json(EXPERIMENTS_DIR, "selection.json")
        if selection:
            st.success(f"Selected (frozen before testing): {', '.join(selection['members'])} — "
                       f"validation macro-F1 {selection['val_macro_f1']:.3f}")
        final = load_json(EXPERIMENTS_DIR, "final_results.json")
        if final:
            st.markdown("**Held-out results (evaluated once, after selection was frozen)**")
            st.dataframe(held_out_table(final), use_container_width=True, hide_index=True)
            for fig in ("model_comparison", "confusion_tweeteval_test", "reliability_tweeteval_test"):
                path = EXPERIMENTS_DIR / "figures" / f"{fig}.png"
                if path.exists():
                    st.image(str(path))
        else:
            st.warning("Held-out (test / election) results are not available yet.")

with tab_info:
    st.subheader("Model")
    st.write(f"**Model:** `{MODEL_ID}`")
    if card and card.get("model_id") == MODEL_ID:
        st.write(f"**Architecture:** {card['architecture']}")
        st.write(f"**Training data:** {card['training_data']}")
        st.write(f"**Recipe:** {card['training_recipe']}")
        for ev in card.get("evaluations", []):
            st.markdown(f"**{ev['dataset']}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{ev['accuracy']:.2%}")
            c2.metric("Macro F1", f"{ev['macro_f1']:.2%}")
            c3.metric("Macro recall", f"{ev['macro_recall']:.2%}" if "macro_recall" in ev else "Not recorded")
            st.caption(f"{ev['note']} Source: {ev['source']}.")
        st.info(f"Election-domain evaluation: {card.get('election_domain_status', 'not available')}")
    else:
        st.warning("No verified evaluation metrics are recorded for this model id. Check the model-card metrics file and deployment version.")
    st.subheader("Limitations")
    st.markdown(
        "- Trained on general tweets (TweetEval, ~2013–2017); election language, new slang and events may differ.\n"
        "- Neutral vs. negative is the most frequent confusion, especially for factual news about negative events.\n"
        "- Sarcasm, irony and mixed sentiment are frequently misclassified.\n"
        "- Human annotators themselves disagree on many tweets, so 100% accuracy is not attainable on these labels.\n"
        "- English only. Do not use for decisions about individuals."
    )
    st.caption(
        "Independent extension of a collaborative AI4ALL Ignite sentiment-analysis project "
        "([source](https://github.com/suhailabdulwasi/sentiment-analysis-nlp))."
    )
