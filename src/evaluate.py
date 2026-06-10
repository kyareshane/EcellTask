import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "data/processed/labeled_dataset.csv"
)

vectorizer = joblib.load(
    "models/tfidf.pkl"
)

model = joblib.load(
    "models/CatBoost.pkl"
)

X = vectorizer.transform(
    df["text"]
)

y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

preds = model.predict(X_test)

cm = confusion_matrix(
    y_test,
    preds
)

print(cm)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("CatBoost Confusion Matrix")

plt.savefig(
    "reports/confusion_matrix.png"
)

plt.show()