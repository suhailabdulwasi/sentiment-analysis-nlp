"""Train and evaluate a 3-class tweet sentiment model without duplicate leakage.

Run from the project directory: python train.py
Requires: pandas, scikit-learn, joblib
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessors import CandidateAnonymizer, NegationTagger

ROOT = Path(__file__).resolve().parent
FILES = [ROOT / 'data' / name for name in ('train.csv', 'val.csv', 'test.csv')]


def main():
    missing = [str(path) for path in FILES if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing CSV files: {missing}')
    df = pd.concat([pd.read_csv(path) for path in FILES], ignore_index=True)
    df = df[['tweet_text', 'sentiment']].dropna().copy()
    df['tweet_text'] = df['tweet_text'].astype(str).str.strip()
    df['sentiment'] = df['sentiment'].astype(str).str.strip().str.lower()
    df = df[df['tweet_text'].ne('')]
    allowed = {'positive', 'neutral', 'negative'}
    if not set(df['sentiment']).issubset(allowed):
        raise ValueError(f'Unexpected labels: {set(df.sentiment) - allowed}')
    df['text_key'] = df['tweet_text'].str.casefold()
    inconsistent = df.groupby('text_key')['sentiment'].nunique()
    if (inconsistent > 1).any():
        raise ValueError('Identical tweets have conflicting labels; review before training.')
    original = len(df)
    df = df.drop_duplicates('text_key')
    print(f'Rows: {original}; unique texts: {len(df)}; removed duplicates: {original - len(df)}')
    X_train, X_test, y_train, y_test = train_test_split(
        df['tweet_text'], df['sentiment'], test_size=0.2,
        stratify=df['sentiment'], random_state=42
    )
    model = Pipeline([
        ('anonymize', CandidateAnonymizer()),
        ('negation', NegationTagger()),
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=2000)),
        ('classifier', LinearSVC(class_weight='balanced', random_state=42)),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    labels = ['negative', 'neutral', 'positive']
    print('Accuracy:', round(accuracy_score(y_test, pred), 4))
    print(classification_report(y_test, pred, labels=labels, zero_division=0))
    print('Confusion matrix (negative, neutral, positive):')
    print(confusion_matrix(y_test, pred, labels=labels))
    report = {
        'data_rows': original, 'unique_texts': len(df),
        'train_rows': len(X_train), 'test_rows': len(X_test),
        'accuracy': accuracy_score(y_test, pred),
        'classification_report': classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0),
        'caveat': 'Single random split of a small deduplicated dataset; not an independent external benchmark.'
    }
    (ROOT / 'metrics.json').write_text(json.dumps(report, indent=2))
    joblib.dump(model, ROOT / 'model.joblib')
    print('Saved model.joblib and metrics.json')


if __name__ == '__main__':
    main()
