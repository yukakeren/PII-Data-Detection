import os
import csv
import json
import numpy as np

from models.boosting.feature_extraction import quick_f1_pii


# ============================================================
# Threshold-based Prediction
# ============================================================

def predict_with_threshold(model, X, le, o_idx_remapped, threshold, reverse_remap):
    proba = model.predict_proba(X)
    pred_idx = np.argmax(proba, axis=1)

    proba_no_o = proba.copy()
    proba_no_o[:, o_idx_remapped] = -1
    best_non_o_idx = np.argmax(proba_no_o, axis=1)
    best_non_o_proba = proba_no_o[np.arange(len(proba)), best_non_o_idx]

    final_idx_remapped = np.where(
        (pred_idx == o_idx_remapped) & (best_non_o_proba >= threshold),
        best_non_o_idx,
        pred_idx,
    )
    final_idx_original = np.array([reverse_remap[i] for i in final_idx_remapped])
    return le.inverse_transform(final_idx_original)


def predict_with_threshold_xgb(model, X, le, o_idx_original, threshold, reverse_remap):
    proba = model.predict_proba(X)
    pred_idx_remapped = np.argmax(proba, axis=1)

    # Cari index O di remapped space
    o_idx_remapped = [k for k, v in reverse_remap.items() if v == o_idx_original][0]

    proba_no_o = proba.copy()
    proba_no_o[:, o_idx_remapped] = -1
    best_non_o_idx = np.argmax(proba_no_o, axis=1)
    best_non_o_proba = proba_no_o[np.arange(len(proba)), best_non_o_idx]

    final_idx_remapped = np.where(
        (pred_idx_remapped == o_idx_remapped) & (best_non_o_proba >= threshold),
        best_non_o_idx,
        pred_idx_remapped,
    )
    final_idx_original = np.array([reverse_remap[i] for i in final_idx_remapped])
    return le.inverse_transform(final_idx_original)


# ============================================================
# Threshold Tuning
# ============================================================

def tune_threshold(model, X_val, y_val_str, le, o_idx_remapped, reverse_remap,
                   thresholds=None, predict_fn=None):
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    if predict_fn is None:
        predict_fn = predict_with_threshold

    best_f1, best_th = -1, 0.5
    for th in thresholds:
        pred = predict_fn(model, X_val, le, o_idx_remapped, th, reverse_remap)
        f1 = quick_f1_pii(y_val_str, pred)
        print(f"  threshold={th:.2f} → Val F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_th = f1, th

    print(f"  ✅ Best threshold: {best_th:.2f} (F1: {best_f1:.4f})")
    return best_th, best_f1


# ============================================================
# Save Predictions & Metrics
# ============================================================

def save_predictions_csv(meta, pred_labels_str, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["document_id", "token", "true_label", "pred_label"]
        )
        writer.writeheader()
        for (doc_id, token, true_lbl), pred_lbl in zip(meta, pred_labels_str):
            writer.writerow({
                "document_id": doc_id,
                "token": token,
                "true_label": true_lbl,
                "pred_label": pred_lbl,
            })
    print(f"  ✅ Saved predictions → {output_path}")


def save_metrics_json(csv_path, model_name, output_path):
    from src.evaluate import evaluate_from_csv, print_metrics, save_metrics

    metrics = evaluate_from_csv(csv_path, model_name)
    print_metrics(metrics)

    save_metrics(metrics, output_path)
    print(f"  ✅ Saved metrics → {output_path}")

    return metrics


# ============================================================
# Full Prediction Pipeline
# ============================================================

def run_prediction_pipeline(
    models_dict,
    X_test_imb, X_val_imb, y_val_imb_str, meta_test_imb,
    X_test_bal, X_val_bal, y_val_bal_str, meta_test_bal,
    le, o_index,
    remap_imb, reverse_remap_imb, remap_bal, reverse_remap_bal,
):
    o_idx_remapped_imb = remap_imb[o_index]
    o_idx_remapped_bal = remap_bal[o_index]

    all_metrics = {}

    # ── 1. LightGBM Imbalance ──
    print("\n" + "=" * 60)
    print("🔹 LightGBM Imbalance — Threshold Tuning")
    print("  Data: imbalance/val.json → imbalance/test.json")
    print("=" * 60)
    best_th, _ = tune_threshold(
        models_dict["lgb_imb"], X_val_imb, y_val_imb_str, le,
        o_idx_remapped_imb, reverse_remap_imb,
        predict_fn=predict_with_threshold,
    )
    pred = predict_with_threshold(
        models_dict["lgb_imb"], X_test_imb, le,
        o_idx_remapped_imb, best_th, reverse_remap_imb,
    )
    save_predictions_csv(meta_test_imb, pred, "results/predictions/lightgbm_imbalance_predictions.csv")
    m = save_metrics_json(
        "results/predictions/lightgbm_imbalance_predictions.csv",
        "lightgbm_imbalance",
        "results/metrics/lightgbm_imbalance_metrics.json",
    )
    all_metrics["lightgbm_imbalance"] = m

    # ── 2. XGBoost Imbalance ──
    print("\n" + "=" * 60)
    print("🔹 XGBoost Imbalance — Threshold Tuning")
    print("  Data: imbalance/val.json → imbalance/test.json")
    print("=" * 60)
    best_th, _ = tune_threshold(
        models_dict["xgb_imb"], X_val_imb, y_val_imb_str, le,
        o_index, reverse_remap_imb,
        predict_fn=predict_with_threshold_xgb,
    )
    pred = predict_with_threshold_xgb(
        models_dict["xgb_imb"], X_test_imb, le,
        o_index, best_th, reverse_remap_imb,
    )
    save_predictions_csv(meta_test_imb, pred, "results/predictions/xgboost_imbalance_predictions.csv")
    m = save_metrics_json(
        "results/predictions/xgboost_imbalance_predictions.csv",
        "xgboost_imbalance",
        "results/metrics/xgboost_imbalance_metrics.json",
    )
    all_metrics["xgboost_imbalance"] = m

    # ── 3. LightGBM Balance ──
    print("\n" + "=" * 60)
    print("🔹 LightGBM Balance — Threshold Tuning")
    print("  Data: balance/val.json → balance/test.json")
    print("=" * 60)
    best_th, _ = tune_threshold(
        models_dict["lgb_bal"], X_val_bal, y_val_bal_str, le,
        o_idx_remapped_bal, reverse_remap_bal,
        predict_fn=predict_with_threshold,
    )
    pred = predict_with_threshold(
        models_dict["lgb_bal"], X_test_bal, le,
        o_idx_remapped_bal, best_th, reverse_remap_bal,
    )
    save_predictions_csv(meta_test_bal, pred, "results/predictions/lightgbm_balance_predictions.csv")
    m = save_metrics_json(
        "results/predictions/lightgbm_balance_predictions.csv",
        "lightgbm_balance",
        "results/metrics/lightgbm_balance_metrics.json",
    )
    all_metrics["lightgbm_balance"] = m

    # ── 4. XGBoost Balance ──
    print("\n" + "=" * 60)
    print("🔹 XGBoost Balance — Threshold Tuning")
    print("  Data: balance/val.json → balance/test.json")
    print("=" * 60)
    best_th, _ = tune_threshold(
        models_dict["xgb_bal"], X_val_bal, y_val_bal_str, le,
        o_index, reverse_remap_bal,
        predict_fn=predict_with_threshold_xgb,
    )
    pred = predict_with_threshold_xgb(
        models_dict["xgb_bal"], X_test_bal, le,
        o_index, best_th, reverse_remap_bal,
    )
    save_predictions_csv(meta_test_bal, pred, "results/predictions/xgboost_balance_predictions.csv")
    m = save_metrics_json(
        "results/predictions/xgboost_balance_predictions.csv",
        "xgboost_balance",
        "results/metrics/xgboost_balance_metrics.json",
    )
    all_metrics["xgboost_balance"] = m

    # ── Summary ──
    print("\n" + "=" * 60)
    print("📊 SUMMARY — All Models")
    print("=" * 60)
    print(f"{'Model':<25} {'Token F1':>10} {'Entity F1':>10}")
    print("-" * 47)
    for name, m in all_metrics.items():
        tf1 = m["token_level"]["f1"]
        ef1 = m["entity_level"]["f1"]
        print(f"{name:<25} {tf1:>10.4f} {ef1:>10.4f}")

    return all_metrics
