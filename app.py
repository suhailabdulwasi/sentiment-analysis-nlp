
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.joblib"
METRICS_PATH = ROOT / "metrics.json"


st.set_page_config(
    page_title="Election Sentiment Analyzer",
    page_icon="🗳️",
    layout="centered",
)


@st.cache_resource
def load_model():
    # Only load model files created by you or trusted sources.
    return joblib.load(MODEL_PATH)


st.title("2024 U.S. Election Sentiment Analyzer")

st.write(
    "Analyze political text using a machine learning "
    "model trained with TF-IDF and LinearSVC."
)

st.caption(
    "This application classifies sentiment, not political "
    "affiliation, factual accuracy, or candidate support."
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
            prediction = str(model.predict([user_text])[0])

            st.subheader("Prediction")

            if prediction == "positive":
                st.success("Positive sentiment")
            elif prediction == "negative":
                st.error("Negative sentiment")
            else:
                st.info("Neutral sentiment")

            st.caption(
                "Predictions may be incorrect, especially "
                "for sarcasm or unfamiliar language."
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
                st.error(
                    "Your CSV must contain a tweet_text column."
                )
            else:
                texts = df["tweet_text"].fillna("").astype(str)

                if st.button("Run Batch Analysis"):
                    df["predicted_sentiment"] = model.predict(
                        texts.tolist()
                    )

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

    st.write("**Algorithm:** Linear Support Vector Classifier")
    st.write("**Feature extraction:** TF-IDF")
    st.write("**Preprocessing:** Candidate anonymization and negation tagging")

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as file:
            metrics = json.load(file)

        accuracy = metrics.get("accuracy")
        if accuracy is not None:
            st.metric("Held-out test accuracy", f"{accuracy:.1%}")

        st.write(
            "**Evaluation dataset:** "
            f"{metrics.get('test_rows', 'Unknown')} examples"
        )

        st.info(
            "This accuracy comes from a small deduplicated "
            "dataset and is not an independent external benchmark."
        )

    st.write(
        "Built as an extension of a collaborative "
        "AI4ALL Ignite sentiment analysis project."
    )
