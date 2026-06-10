from fastapi import FastAPI
from pydantic import BaseModel

import joblib

app = FastAPI()

vectorizer = joblib.load(
    "models/tfidf.pkl"
)

model = joblib.load(
    "models/CatBoost.pkl"
)

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputText):

    X = vectorizer.transform(
        [data.text]
    )

    pred = model.predict(X)
    print("Prediction raw:", pred)
    prediction = int(pred[0][0])

    probability = float(
    max(model.predict_proba(X)[0])
    )

    labels = {
        0: "Low Complexity",
        1: "Medium Complexity",
        2: "High Complexity"
    }

    return {
        "label": labels[prediction],
        "confidence": float(probability)
    }