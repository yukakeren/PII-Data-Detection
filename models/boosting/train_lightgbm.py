"""
Training script untuk LightGBM model pada PII Detection.
Mendukung dua mode: imbalance (data penuh) dan balance (undersampled O + oversampled PII).
"""

import numpy as np
from collections import Counter
from lightgbm import LGBMClassifier

from models.boosting.feature_extraction import (
    compute_class_weights,
    get_sample_weights,
    quick_f1_pii,
)


def train_lightgbm_imbalance(X_train, y_train, y_train_str, X_val, y_val, le):
    """
    Train LightGBM dengan data imbalance (full dataset).
    
    Menggunakan sample weights untuk menangani class imbalance
    tanpa mengubah distribusi data.
    
    Args:
        X_train: sparse feature matrix training
        y_train: encoded integer labels training
        y_train_str: string labels training (untuk compute weights)
        X_val: sparse feature matrix validation
        y_val: encoded integer labels validation
        le: LabelEncoder instance
        
    Returns:
        model: trained LGBMClassifier
        val_f1: validation F1 score (PII only)
    """
    print("\n=== Training LightGBM (Imbalance) ===")

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


def balance_dataset(X_train, y_train, y_train_str, target_o_ratio=3, target_per_label=500, seed=42):
    """
    Balance dataset dengan undersample class O dan oversample minority PII.
    
    Strategy:
    - Undersample O: target = jumlah_PII_tokens * target_o_ratio
    - Oversample minority PII: minimum target_per_label samples per class
    
    Args:
        X_train: sparse feature matrix
        y_train: encoded integer labels
        y_train_str: string labels
        target_o_ratio: rasio O terhadap PII tokens
        target_per_label: minimum samples per PII class
        seed: random seed
        
    Returns:
        X_bal: balanced feature matrix
        y_bal: balanced integer labels
        y_bal_str: balanced string labels
    """
    print("\n📊 Balancing dataset...")
    label_counts = Counter(y_train_str)

    print("Distribusi sebelum balance:")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {lbl}: {cnt}")

    pii_indices = [i for i, lbl in enumerate(y_train_str) if lbl != "O"]
    o_indices = [i for i, lbl in enumerate(y_train_str) if lbl == "O"]
    target_o = len(pii_indices) * target_o_ratio

    np.random.seed(seed)
    o_sampled = np.random.choice(o_indices, size=min(target_o, len(o_indices)), replace=False)

    # Oversample minority PII classes
    extra_indices = []
    for lbl, cnt in label_counts.items():
        if lbl == "O":
            continue
        lbl_idx = [i for i, l in enumerate(y_train_str) if l == lbl]
        if cnt < target_per_label:
            extra = np.random.choice(lbl_idx, size=target_per_label - cnt, replace=True)
            extra_indices.extend(extra)

    balanced_indices = np.array(list(o_sampled) + pii_indices + extra_indices)
    np.random.shuffle(balanced_indices)

    X_bal = X_train[balanced_indices]
    y_bal = y_train[balanced_indices]
    y_bal_str = [y_train_str[i] for i in balanced_indices]

    print(f"\nDistribusi sesudah balance:")
    for lbl, cnt in sorted(Counter(y_bal_str).items(), key=lambda x: -x[1]):
        print(f"  {lbl}: {cnt}")
    print(f"\nTotal: {len(y_bal)} token (sebelumnya: {len(y_train_str)})")

    return X_bal, y_bal, y_bal_str


def train_lightgbm_balance(X_train_bal, y_train_bal, y_train_bal_str, X_val, y_val, le):
    """
    Train LightGBM dengan data balanced.
    
    Menggunakan min_child_samples=50 (lebih tinggi dari imbalance)
    untuk mengurangi overfitting pada data yang lebih kecil.
    
    Args:
        X_train_bal: balanced sparse feature matrix
        y_train_bal: balanced encoded integer labels
        y_train_bal_str: balanced string labels (untuk weights)
        X_val: sparse feature matrix validation
        y_val: encoded integer labels validation
        le: LabelEncoder instance
        
    Returns:
        model: trained LGBMClassifier
        val_f1: validation F1 score (PII only)
    """
    print("\n=== Training LightGBM (Balance) ===")

    class_weights = compute_class_weights(y_train_bal, power=0.3)
    sample_weights = get_sample_weights(y_train_bal, class_weights)

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

    model.fit(X_train_bal, y_train_bal, sample_weight=sample_weights)

    val_f1 = quick_f1_pii(
        le.inverse_transform(y_val),
        le.inverse_transform(model.predict(X_val)),
    )
    print(f"  Val F1 (PII): {val_f1:.4f}")

    return model, val_f1, sample_weights
