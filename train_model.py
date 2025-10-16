# train_model.py
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = 'model.pkl'
CSV_PATH = 'dataset.csv'

def load_data():
    if os.path.exists(CSV_PATH):
        print("Found dataset.csv — loading it.")
        df = pd.read_csv(CSV_PATH)
        # expect columns: text,label  (label: 1 positive, 0 negative)
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("dataset.csv must have columns: text,label")
        texts = df['text'].astype(str).tolist()
        labels = df['label'].astype(int).tolist()
    else:
        print("dataset.csv not found — using built-in tiny dataset.")
        texts = [
            "I love this product, it's fantastic!",
            "This is the best thing I've bought this year.",
            "Absolutely wonderful, I am so happy with it.",
            "Terrible experience, I hate it.",
            "This is the worst product I've ever used.",
            "I am very disappointed and upset.",
            "Not bad, works as expected.",
            "It is okay, nothing special.",
            "Pretty good overall, satisfied.",
            "Extremely poor quality, do not buy."
        ]
        labels = [1,1,1,0,0,0,1,1,1,0]
    return texts, labels

def train_and_save():
    texts, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.25, random_state=42)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=2000)),
        ('clf', LogisticRegression(max_iter=200))
    ])
    print("Training model...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    # retrain on full data for better final model
    pipeline.fit(texts, labels)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == '__main__':
    train_and_save()
