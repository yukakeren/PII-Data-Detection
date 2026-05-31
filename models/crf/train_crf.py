import os
import json
import joblib
import pandas as pd
import sklearn_crfsuite
from sklearn.metrics import classification_report
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
)
from collections import defaultdict

from feature_extraction import sentence_features


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

TRAIN_PATH = os.path.join(DATA_DIR, "train.json")
VAL_PATH = os.path.join(DATA_DIR, "val.json")
TEST_PATH = os.path.join(DATA_DIR, "test_internal.json")

LABEL_SCHEMA_PATH = os.path.join(
    BASE_DIR,
    "configs",
    "label_schema.json"
)

with open(
    LABEL_SCHEMA_PATH,
    "r",
    encoding="utf-8"
) as f:
    label_schema = json.load(f)

VALID_LABELS = set(
    label_schema.keys()
)

ALL_LABELS = sorted(
    label for label in VALID_LABELS
    if label != "O"
)

MODEL_DIR = os.path.join(BASE_DIR, "models", "crf")
RESULT_DIR = os.path.join(BASE_DIR, "results")
PRED_DIR = os.path.join(RESULT_DIR, "predictions")
METRIC_DIR = os.path.join(RESULT_DIR, "metrics")

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_dataset(data):
    X = []
    y = []

    for doc in data:

        filtered_tokens = []
        filtered_labels = []

        for token, label in zip(
            doc["tokens"],
            doc["labels"]
        ):

            if token.strip() == "":
                continue

            # VALIDASI LABEL
            if label not in VALID_LABELS:
                raise ValueError(
                    f"Unknown label: {label}"
                )

            filtered_tokens.append(token)
            filtered_labels.append(label)

        X.append(
            sentence_features(filtered_tokens)
        )

        y.append(
            filtered_labels
        )

    return X, y


print("Loading dataset...")

train_data = load_json(TRAIN_PATH)
val_data = load_json(VAL_PATH)
test_data = load_json(TEST_PATH)

X_train, y_train = prepare_dataset(train_data)
X_val, y_val = prepare_dataset(val_data)
X_test, y_test = prepare_dataset(test_data)

print("Training CRF model...")

crf = sklearn_crfsuite.CRF(
    algorithm="lbfgs",
    c1=0.01,
    c2=0.01,
    max_iterations=300,
    all_possible_transitions=True,
    verbose=True
)

crf.fit(X_train, y_train)

model_path = os.path.join(MODEL_DIR, "crf_model.pkl")
joblib.dump(crf, model_path)

print(f"Model saved: {model_path}")

print("Evaluating validation set...")

val_preds = crf.predict(X_val)

from seqeval.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_val, val_preds))


def compute_token_metrics(y_true, y_pred):
    stats = defaultdict(
        lambda: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        }
    )

    total_tp = total_fp = total_fn = 0

    for true_seq, pred_seq in zip(y_true, y_pred):
        for true_label, pred_label in zip(
            true_seq,
            pred_seq
        ):

            if true_label == pred_label:
                if true_label != "O":
                    stats[true_label]["tp"] += 1
                    total_tp += 1

            else:
                if pred_label != "O":
                    stats[pred_label]["fp"] += 1
                    total_fp += 1

                if true_label != "O":
                    stats[true_label]["fn"] += 1
                    total_fn += 1

    per_class = {}

    for label in ALL_LABELS:

        values = stats.get(
            label,
            {
                "tp": 0,
                "fp": 0,
                "fn": 0
            }
        )

        tp = values["tp"]
        fp = values["fp"]
        fn = values["fn"]

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0 else 0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0 else 0
        )

        f1 = (
            2 * precision * recall /
            (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }

    token_precision = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0
    )

    token_recall = (
        total_tp / (total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 0
    )

    token_f1 = (
        2 * token_precision * token_recall /
        (token_precision + token_recall)
        if (token_precision + token_recall) > 0
        else 0
    )

    return {
        "precision": token_precision,
        "recall": token_recall,
        "f1": token_f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "per_class": per_class
    }


token_metrics = compute_token_metrics(
    y_val,
    val_preds
)

metrics = {
    "model": "crf",

    "total_tokens": sum(
        len(seq) for seq in y_val
    ),

    "token_level": token_metrics,

    "entity_level": {
        "precision": float(
            precision_score(
                y_val,
                val_preds
            )
        ),
        "recall": float(
            recall_score(
                y_val,
                val_preds
            )
        ),
        "f1": float(
            f1_score(
                y_val,
                val_preds
            )
        ),
        "total_tp": 0,
        "total_fp": 0,
        "total_fn": 0
    }
}

metric_path = os.path.join(
    METRIC_DIR,
    "crf_metrics.json"
)

with open(metric_path, "w") as f:
    json.dump(
        metrics,
        f,
        indent=4
    )

print("Metrics:")
print(metrics)

print("Predicting test set...")

test_preds = crf.predict(X_test)

rows = []

for doc, pred_labels in zip(test_data, test_preds):
    doc_id = doc["document"]

    filtered_tokens = []
    filtered_labels = []

    for token, label in zip(
        doc["tokens"],
        doc["labels"]
    ):
        if token.strip() == "":
            continue

        filtered_tokens.append(token)
        filtered_labels.append(label)

    for token, true_label, pred_label in zip(
        filtered_tokens,
        filtered_labels,
        pred_labels
    ):
        rows.append({
            "document_id": doc_id,
            "token": token,
            "true_label": true_label,
            "pred_label": pred_label
        })

df = pd.DataFrame(rows)

csv_path = os.path.join(
    PRED_DIR,
    "crf_predictions.csv"
)

df.to_csv(csv_path, index=False)

print(len(y_val))
print(sum(len(seq) for seq in y_val))
print(f"Prediction saved: {csv_path}")
print("Done.")