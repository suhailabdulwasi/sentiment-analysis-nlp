import pandas as pd
import streamlit as st
from transformers import pipeline

MODEL_ID = "suhailabdulwasi/ai4all-distilroberta-sentiment"

st.set_page_config(
    page_title="Election Sentiment Analyzer",
    page_icon="🗳️",
    layout="centered",
)

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        device=-1,
    )

def predict_sentiments(texts):
    results = model(
        texts,
        truncation=True,
        max_length=512,
        batch_size=16,
    )
    return [result["label"].lower() for result in results]

st.title("Election Sentiment Analyzer")

st.write(
    "Analyze text using a fine-tuned DistilRoBERTa "
    "sentiment classification model."
)

st.caption(
    "This model predicts positive, negative, or neutral sentiment. "
    "It does not determine political affiliation, factual accuracy, "
    "or candidate support."
)

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()

tab1, tab2, tab3 = st.tabs(
    ["Analyze Text", "Batch Analysis", "Model Information"]
)

with tab1:
    st.subheader("Analyze a statement")

    user_text = st.text_area(
        "Enter a tweet or political statement:",
        placeholder="I really liked the candidate's speech!",
        height=130,
    )

    if st.button("Analyze Sentiment", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            prediction = predict_sentiments([user_text])[0]

            st.subheader("Prediction")

            if prediction == "positive":
                st.success("Positive sentiment")
            elif prediction == "negative":
                st.error("Negative sentiment")
            else:
                st.info("Neutral sentiment")

            st.caption(
                "Predictions may be incorrect, particularly "
                "for sarcasm or ambiguous language."
            )

with tab2:
    st.subheader("Analyze multiple statements")

    uploaded_file = st.file_uploader(
        "Upload a CSV containing a tweet_text column",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if "tweet_text" not in df.columns:
                st.error("Your CSV must contain a tweet_text column.")
            else:
                if st.button("Run Batch Analysis"):
                    texts = df["tweet_text"].fillna("").astype(str).tolist()

                    if not texts:
                        st.warning("The CSV contains no rows.")
                    else:
                        df["predicted_sentiment"] = predict_sentiments(texts)

                        st.dataframe(df, use_container_width=True)

                        st.download_button(
                            "Download Predictions",
                            data=df.to_csv(index=False),
                            file_name="sentiment_predictions.csv",
                            mime="text/csv",
                        )
        except Exception as error:
            st.error(f"Could not process CSV: {error}")

with tab3:
    st.subheader("Model Information")

    st.write("**Architecture:** DistilRoBERTa")
    st.write("**Training dataset:** TweetEval sentiment")
    st.write("**Classification:** Positive, negative, neutral")
    st.write("**Training:** Fine-tuned for 2 epochs")

    st.metric("Held-out test accuracy", "70.41%")
    st.metric("Held-out test macro F1", "70.44%")
    st.write("**Test samples:** 12,284")

    st.info(
        "These metrics come from the TweetEval sentiment test set. "
        "Performance on election-specific statements has not "
        "been independently evaluated."
    )

    st.markdown(
        "[View model on Hugging Face]"
        "(https://huggingface.co/suhailabdulwasi/"
        "ai4all-distilroberta-sentiment)"
    )

    st.write(
        "An independent extension of a collaborative "
        "AI4ALL Ignite sentiment analysis project."
    )
