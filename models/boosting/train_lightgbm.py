import numpy as np
from collections import Counter
from lightgbm import LGBMClassifier

from models.boosting.feature_extraction import (
    compute_class_weights,
    get_sample_weights,
    quick_f1_pii,
)


def train_lightgbm_imbalance(X_train, y_train, y_train_str, X_val, y_val, le):
    print("\n=== Training LightGBM (Imbalance) ===")
    print(f"  Data: data/processed/imbalance/")
    print(f"  Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")

    # Compute sample weights
    class_weights = compute_class_weights(y_train, power=0.5)
    sample_weights = get_sample_weights(y_train, class_weights)

    model = LGBMClassifier(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.06,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.5,
        reg_lambda=0.5,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)

    val_f1 = quick_f1_pii(
        le.inverse_transform(y_val),
        le.inverse_transform(model.predict(X_val)),
    )
    print(f"  Val F1 (PII): {val_f1:.4f}")

    return model, val_f1, sample_weights


def train_lightgbm_balance(X_train, y_train, y_train_str, X_val, y_val, le):
    print("\n=== Training LightGBM (Balance) ===")
    print(f"  Data: data/processed/balance/")
    print(f"  Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")

    # Log distribusi label
    label_counts = Counter(y_train_str)
    print("  Distribusi label training:")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    {lbl}: {cnt}")

    class_weights = compute_class_weights(y_train, power=0.3)
    sample_weights = get_sample_weights(y_train, class_weights)

    model = LGBMClassifier(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.06,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=50,
        reg_alpha=0.5,
        reg_lambda=0.5,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)

    val_f1 = quick_f1_pii(
        le.inverse_transform(y_val),
        le.inverse_transform(model.predict(X_val)),
    )
    print(f"  Val F1 (PII): {val_f1:.4f}")

    return model, val_f1, sample_weights
