"""
Training script untuk XGBoost model pada PII Detection.
Mendukung dua mode: imbalance (data penuh) dan balance (data balanced).

Note: XGBoost memerlukan label contiguous 0..N-1, sehingga diperlukan
label remapping sebelum training.
"""

import numpy as np
from xgboost import XGBClassifier

from models.boosting.feature_extraction import (
    compute_class_weights,
    get_sample_weights,
    quick_f1_pii,
)


def create_label_remap(y_train):
    """
    Buat remapping label agar contiguous 0..N-1 (requirement XGBoost).
    
    Args:
        y_train: encoded integer labels
        
    Returns:
        remap: Dict[old_label -> new_label]
        reverse_remap: Dict[new_label -> old_label]
        y_remapped: remapped labels
    """
    unique_labels = sorted(np.unique(y_train))
    remap = {old: new for new, old in enumerate(unique_labels)}
    reverse_remap = {new: old for old, new in remap.items()}
    y_remapped = np.array([remap[label] for label in y_train])

    print(f"  Label remap: {len(unique_labels)} classes, contiguous 0..{len(unique_labels)-1}")
    return remap, reverse_remap, y_remapped


def train_xgboost_imbalance(X_train, y_train, y_train_str, X_val, y_val, le,
                             remap, reverse_remap, y_train_remapped, sample_weights):
    """
    Train XGBoost dengan data imbalance (full dataset).
    
    Menggunakan GPU acceleration (device="cuda") dan tree_method="hist"
    untuk training lebih cepat.
    
    Args:
        X_train: sparse feature matrix training
        y_train: encoded integer labels training (original)
        y_train_str: string labels training
        X_val: sparse feature matrix validation
        y_val: encoded integer labels validation
        le: LabelEncoder instance
        remap: label remapping dict
        reverse_remap: reverse label remapping dict
        y_train_remapped: remapped labels untuk XGBoost
        sample_weights: per-sample weights
        
    Returns:
        model: trained XGBClassifier
        val_f1: validation F1 score (PII only)
    """
    print("\n=== Training XGBoost (Imbalance) ===")

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


def train_xgboost_balance(X_train_bal, y_train_bal, y_train_bal_str,
                           X_val, y_val, le, o_index):
    """
    Train XGBoost dengan data balanced.
    
    Membuat remapping khusus untuk balanced data dan menggunakan
    min_child_weight=3 (lebih rendah dari imbalance) karena data lebih kecil.
    
    Args:
        X_train_bal: balanced sparse feature matrix
        y_train_bal: balanced encoded integer labels
        y_train_bal_str: balanced string labels
        X_val: sparse feature matrix validation
        y_val: encoded integer labels validation
        le: LabelEncoder instance
        o_index: index label "O" dalam LabelEncoder
        
    Returns:
        model: trained XGBClassifier
        val_f1: validation F1 score (PII only)
        remap_bal: remapping dict untuk balanced data
        reverse_remap_bal: reverse remapping dict
    """
    print("\n=== Training XGBoost (Balance) ===")

    # Remap labels untuk balanced data
    remap_bal, reverse_remap_bal, y_train_bal_remapped = create_label_remap(y_train_bal)

    class_weights = compute_class_weights(y_train_bal, power=0.3)
    sample_weights = get_sample_weights(y_train_bal, class_weights)

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

    model.fit(X_train_bal, y_train_bal_remapped, sample_weight=sample_weights)

    # Predict dan reverse remap
    y_val_pred_remapped = model.predict(X_val)
    y_val_pred_original = np.array([reverse_remap_bal[p] for p in y_val_pred_remapped])
    val_f1 = quick_f1_pii(
        le.inverse_transform(y_val),
        le.inverse_transform(y_val_pred_original),
    )
    print(f"  Val F1 (PII): {val_f1:.4f}")

    return model, val_f1, remap_bal, reverse_remap_bal
