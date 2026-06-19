import numpy as np
from collections import Counter
from xgboost import XGBClassifier

from models.boosting.feature_extraction import (
    compute_class_weights,
    get_sample_weights,
    quick_f1_pii,
)


def create_label_remap(y_train):
    unique_labels = sorted(np.unique(y_train))
    remap = {old: new for new, old in enumerate(unique_labels)}
    reverse_remap = {new: old for old, new in remap.items()}
    y_remapped = np.array([remap[label] for label in y_train])

    print(f"  Label remap: {len(unique_labels)} classes, contiguous 0..{len(unique_labels)-1}")
    return remap, reverse_remap, y_remapped


def train_xgboost_imbalance(X_train, y_train, y_train_str, X_val, y_val, le,
                             remap, reverse_remap, y_train_remapped, sample_weights):
    print("\n=== Training XGBoost (Imbalance) ===")
    print(f"  Data: data/processed/imbalance/")
    print(f"  Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="mlogloss",
        tree_method="hist",
        device="cuda",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )

    model.fit(X_train, y_train_remapped, sample_weight=sample_weights)

    # Predict dan reverse remap
    y_val_pred_remapped = model.predict(X_val)
    y_val_pred_original = np.array([reverse_remap[p] for p in y_val_pred_remapped])
    val_f1 = quick_f1_pii(
        le.inverse_transform(y_val),
        le.inverse_transform(y_val_pred_original),
    )
    print(f"  Val F1 (PII): {val_f1:.4f}")

    return model, val_f1


def train_xgboost_balance(X_train, y_train, y_train_str,
                           X_val, y_val, le, o_index):
    print("\n=== Training XGBoost (Balance) ===")
    print(f"  Data: data/processed/balance/")
    print(f"  Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")

    # Log distribusi label
    label_counts = Counter(y_train_str)
    print("  Distribusi label training:")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    {lbl}: {cnt}")

    # Remap labels untuk balanced data
    remap_bal, reverse_remap_bal, y_train_bal_remapped = create_label_remap(y_train)

    class_weights = compute_class_weights(y_train, power=0.3)
    sample_weights = get_sample_weights(y_train, class_weights)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.5,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="mlogloss",
        tree_method="hist",
        device="cuda",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )

    model.fit(X_train, y_train_bal_remapped, sample_weight=sample_weights)

    # Predict dan reverse remap
    y_val_pred_remapped = model.predict(X_val)
    y_val_pred_original = np.array([reverse_remap_bal[p] for p in y_val_pred_remapped])
    val_f1 = quick_f1_pii(
        le.inverse_transform(y_val),
        le.inverse_transform(y_val_pred_original),
    )
    print(f"  Val F1 (PII): {val_f1:.4f}")

    return model, val_f1, remap_bal, reverse_remap_bal
