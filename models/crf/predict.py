import json
import joblib
from feature_extraction import sentence_features


MODEL_PATH = "models/crf/crf_model.pkl"


def predict_tokens(tokens):
    model = joblib.load(MODEL_PATH)

    X = [sentence_features(tokens)]

    preds = model.predict(X)

    return list(zip(tokens, preds[0]))


if __name__ == "__main__":
    sample = [
        "Nama",
        "saya",
        "Farras",
        "email",
        "abc@gmail.com"
    ]

    result = predict_tokens(sample)

    for token, label in result:
        print(token, "->", label)