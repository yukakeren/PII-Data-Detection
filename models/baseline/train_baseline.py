import os
import sys
import csv
import re
import time
import joblib
import warnings

from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)

from src.data_loader import DataLoader
from src.evaluate import evaluate_from_csv, save_metrics

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train.json")
VAL_PATH = os.path.join(BASE_DIR, "data", "processed", "val.json")
TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test_internal.json")

MODEL_DIR = os.path.join(BASE_DIR, "models", "baseline")
PRED_DIR = os.path.join(BASE_DIR, "results", "predictions")
METRIC_DIR = os.path.join(BASE_DIR, "results", "metrics")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)


def progress(message):
    print(message, flush=True)


def elapsed(start_time):
    return f"{time.time() - start_time:.2f}s"


def is_email(token):
    return bool(re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", token))


def is_url(token):
    token = token.lower()
    return (
        "http" in token
        or "www" in token
        or ".com" in token
        or ".org" in token
        or ".net" in token
    )


def token_shape(token):
    result = ""

    for char in token:
        if char.isupper():
            result += "X"
        elif char.islower():
            result += "x"
        elif char.isdigit():
            result += "d"
        else:
            result += char

    return result


def extract_features(tokens, i):
    token = tokens[i]
    lower = token.lower()

    features = {
        "token_lower": lower,
        "token_shape": token_shape(token),
        "token_length": len(token),
        "is_digit": token.isdigit(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(char.isdigit() for char in token),
        "has_alpha": any(char.isalpha() for char in token),
        "has_at": "@" in token,
        "has_dot": "." in token,
        "has_dash": "-" in token,
        "is_email": is_email(token),
        "is_url": is_url(token),
        "prefix_1": lower[:1],
        "prefix_2": lower[:2],
        "prefix_3": lower[:3],
        "suffix_1": lower[-1:],
        "suffix_2": lower[-2:],
        "suffix_3": lower[-3:],
    }

    if i > 0:
        prev_token = tokens[i - 1]
        features["prev_token"] = prev_token.lower()
        features["prev_shape"] = token_shape(prev_token)
        features["prev_is_title"] = prev_token.istitle()
    else:
        features["BOS"] = True

    if i < len(tokens) - 1:
        next_token = tokens[i + 1]
        features["next_token"] = next_token.lower()
        features["next_shape"] = token_shape(next_token)
        features["next_is_title"] = next_token.istitle()
    else:
        features["EOS"] = True

    return features


def prepare_data(data, dataset_name):
    X_features = []
    X_tokens = []
    y = []
    meta = []

    total_docs = len(data)
    start_time = time.time()

    progress(f"Preparing {dataset_name} data...")
    progress(f"{dataset_name}: total documents = {total_docs}")

    for doc_idx, doc in enumerate(data, start=1):
        doc_id = doc["document"]
        tokens = doc["tokens"]
        labels = doc["labels"]

        for i, token in enumerate(tokens):
            X_features.append(extract_features(tokens, i))
            X_tokens.append(token.lower())
            y.append(labels[i])
            meta.append({
                "document_id": doc_id,
                "token": token,
                "true_label": labels[i],
            })

        if doc_idx % 500 == 0 or doc_idx == total_docs:
            progress(
                f"{dataset_name}: processed {doc_idx}/{total_docs} documents "
                f"| tokens = {len(X_tokens)} "
                f"| elapsed = {elapsed(start_time)}"
            )

    progress(f"{dataset_name}: feature extraction done")
    progress(f"{dataset_name}: total tokens = {len(X_tokens)}")

    return X_features, X_tokens, y, meta


def save_predictions(meta, predictions, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["document_id", "token", "true_label", "pred_label"]
        )

        writer.writeheader()

        for row, pred in zip(meta, predictions):
            writer.writerow({
                "document_id": row["document_id"],
                "token": row["token"],
                "true_label": row["true_label"],
                "pred_label": pred,
            })


def print_simple_metrics(metrics):
    token = metrics["token_level"]
    entity = metrics["entity_level"]

    progress(f"{metrics['model']} result:")
    progress(f"total tokens: {metrics['total_tokens']}")

    progress(
        f"token-level: "
        f"precision = {token['precision']:.4f}, "
        f"recall = {token['recall']:.4f}, "
        f"f1 = {token['f1']:.4f}"
    )

    progress(
        f"entity-level: "
        f"precision = {entity['precision']:.4f}, "
        f"recall = {entity['recall']:.4f}, "
        f"f1 = {entity['f1']:.4f}"
    )


def build_feature_matrix(
    dict_vectorizer,
    tfidf_vectorizer,
    X_features,
    X_tokens,
    fit=False
):
    if fit:
        dict_matrix = dict_vectorizer.fit_transform(X_features)
        tfidf_matrix = tfidf_vectorizer.fit_transform(X_tokens)
    else:
        dict_matrix = dict_vectorizer.transform(X_features)
        tfidf_matrix = tfidf_vectorizer.transform(X_tokens)

    return hstack([dict_matrix, tfidf_matrix])


def train_model(
    model_name,
    classifier,
    X_train,
    y_train,
    X_test,
    meta_test,
    dict_vectorizer,
    tfidf_vectorizer
):
    start_time = time.time()

    progress("")
    progress(f"{model_name}: training started")
    progress(f"{model_name}: fitting classifier...")

    classifier.fit(X_train, y_train)

    progress(f"{model_name}: training finished in {elapsed(start_time)}")

    progress(f"{model_name}: predicting test_internal data...")
    predictions = classifier.predict(X_test)

    pred_path = os.path.join(PRED_DIR, f"{model_name}_predictions.csv")
    save_predictions(meta_test, predictions, pred_path)

    progress(f"{model_name}: evaluating test_internal predictions...")
    metrics = evaluate_from_csv(pred_path, model_name)

    metric_path = os.path.join(METRIC_DIR, f"{model_name}_metrics.json")
    save_metrics(metrics, metric_path)

    model_object = {
        "model_name": model_name,
        "classifier": classifier,
        "dict_vectorizer": dict_vectorizer,
        "tfidf_vectorizer": tfidf_vectorizer,
        "features": [
            "token length",
            "character features",
            "capitalization pattern",
            "digit pattern",
            "email pattern",
            "URL pattern",
            "prefix and suffix",
            "previous token",
            "next token",
            "TF-IDF token representation",
        ],
    }

    model_path = os.path.join(MODEL_DIR, f"{model_name}_model.joblib")
    joblib.dump(model_object, model_path)

    progress(f"{model_name}: predictions saved")
    progress(f"{model_name}: metrics saved")
    progress(f"{model_name}: model saved")

    print_simple_metrics(metrics)


def main():
    total_start = time.time()

    progress("Baseline training started")

    loader = DataLoader()

    progress("Loading train data with DataLoader...")
    train_data = loader.load_raw_json(TRAIN_PATH)
    progress(f"Train data loaded: {len(train_data)} documents")

    progress("Loading validation data with DataLoader...")
    val_data = loader.load_raw_json(VAL_PATH)
    progress(f"Validation data loaded: {len(val_data)} documents")

    progress("Loading test_internal data with DataLoader...")
    test_data = loader.load_raw_json(TEST_PATH)
    progress(f"Test internal data loaded: {len(test_data)} documents")

    X_train_features, X_train_tokens, y_train, _ = prepare_data(train_data, "train")
    X_val_features, X_val_tokens, _, _ = prepare_data(val_data, "val")
    X_test_features, X_test_tokens, _, meta_test = prepare_data(test_data, "test_internal")

    progress("")
    progress("Vectorizing handcrafted features and TF-IDF features...")
    vector_start = time.time()

    dict_vectorizer = DictVectorizer(sparse=True)
    tfidf_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=2
    )

    X_train = build_feature_matrix(
        dict_vectorizer,
        tfidf_vectorizer,
        X_train_features,
        X_train_tokens,
        fit=True
    )

    X_val = build_feature_matrix(
        dict_vectorizer,
        tfidf_vectorizer,
        X_val_features,
        X_val_tokens,
        fit=False
    )

    X_test = build_feature_matrix(
        dict_vectorizer,
        tfidf_vectorizer,
        X_test_features,
        X_test_tokens,
        fit=False
    )

    progress(f"Train matrix shape: {X_train.shape}")
    progress(f"Val matrix shape: {X_val.shape}")
    progress(f"Test matrix shape: {X_test.shape}")
    progress(f"Vectorizing finished in {elapsed(vector_start)}")

    logistic_regression = LogisticRegression(
        class_weight="balanced",
        max_iter=150,
        solver="saga",
        tol=1e-3,
        verbose=0
    )

    linear_svm = LinearSVC(
        class_weight="balanced",
        max_iter=1000,
        tol=1e-3
    )

    train_model(
        "logistic_regression",
        logistic_regression,
        X_train,
        y_train,
        X_test,
        meta_test,
        dict_vectorizer,
        tfidf_vectorizer
    )

    train_model(
        "linear_svm",
        linear_svm,
        X_train,
        y_train,
        X_test,
        meta_test,
        dict_vectorizer,
        tfidf_vectorizer
    )

    progress("")
    progress("Baseline training completed")
    progress(f"Total elapsed time: {elapsed(total_start)}")
    progress("Generated required files:")
    progress("results/predictions/logistic_regression_predictions.csv")
    progress("results/predictions/linear_svm_predictions.csv")
    progress("results/metrics/logistic_regression_metrics.json")
    progress("results/metrics/linear_svm_metrics.json")
    progress("models/baseline/logistic_regression_model.joblib")
    progress("models/baseline/linear_svm_model.joblib")


if __name__ == "__main__":
    main()