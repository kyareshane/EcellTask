import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv(
    "data/processed/labeled_dataset.csv"
)

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = vectorizer.fit_transform(
    df["text"]
)

joblib.dump(
    vectorizer,
    "models/tfidf.pkl"
)

joblib.dump(
    X,
    "models/features.pkl"
)

print(X.shape)