import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import Counter

sns.set_theme(style="whitegrid")
os.makedirs("assets", exist_ok=True)
print("Generating comparative plots...")

# ==========================================
# 1. Dataset Balance vs Imbalance
# ==========================================
def get_label_counts(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return Counter()
    with open(filepath, 'r') as f:
        data = json.load(f)
    labels = [lbl for doc in data for lbl in doc['labels'] if lbl != "O"]
    return Counter(labels)

counts_imb = get_label_counts("data/processed/imbalance/train.json")
counts_bal = get_label_counts("data/processed/balance/train.json")

all_labels = sorted(list(set(counts_imb.keys()) | set(counts_bal.keys())))
if all_labels:
    imb_vals = [counts_imb.get(l, 0) for l in all_labels]
    bal_vals = [counts_bal.get(l, 0) for l in all_labels]

    x = np.arange(len(all_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, imb_vals, width, label='Imbalance (Original)', color='#3498db')
    rects2 = ax.bar(x + width/2, bal_vals, width, label='Balance (Oversampled)', color='#e74c3c')

    ax.set_ylabel('Jumlah Token (Log Scale)', fontsize=12)
    ax.set_title('Perbandingan Distribusi Token PII: Dataset Imbalance vs Balance', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=12)
    ax.set_yscale('log')

    fig.tight_layout()
    plt.savefig("assets/dataset_balance_vs_imbalance.png", dpi=300)
    plt.close()
    print("Saved assets/dataset_balance_vs_imbalance.png")

# ==========================================
# 2. FP & FN Distribution per class
# ==========================================
imb_metrics_path = "results/metrics/distilbert_imbalance_metrics.json"
bal_metrics_path = "results/metrics/distilbert_balance_metrics.json"

if os.path.exists(imb_metrics_path) and os.path.exists(bal_metrics_path):
    with open(imb_metrics_path, "r") as f:
        metrics_imb = json.load(f).get("token_level", {}).get("per_class", {})
        
    with open(bal_metrics_path, "r") as f:
        metrics_bal = json.load(f).get("token_level", {}).get("per_class", {})

    labels = sorted([l for l in metrics_imb.keys() if l != "O"])

    fp_imb = [metrics_imb.get(l, {}).get("fp", 0) for l in labels]
    fp_bal = [metrics_bal.get(l, {}).get("fp", 0) for l in labels]

    fn_imb = [metrics_imb.get(l, {}).get("fn", 0) for l in labels]
    fn_bal = [metrics_bal.get(l, {}).get("fn", 0) for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    # Plot False Positive
    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, fp_imb, width, label='DistilBERT Imbalance', color='#2ecc71')
    rects2 = ax.bar(x + width/2, fp_bal, width, label='DistilBERT Balance', color='#e67e22')

    ax.set_ylabel('Jumlah False Positive (Kesalahan Tebak)', fontsize=12)
    ax.set_title('Distribusi False Positive per Token (DistilBERT: Imbalance vs Balance)', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=12)

    fig.tight_layout()
    plt.savefig("assets/fp_per_label.png", dpi=300)
    plt.close()
    print("Saved assets/fp_per_label.png")

    # Plot False Negative
    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, fn_imb, width, label='DistilBERT Imbalance', color='#9b59b6')
    rects2 = ax.bar(x + width/2, fn_bal, width, label='DistilBERT Balance', color='#f1c40f')

    ax.set_ylabel('Jumlah False Negative (Gagal Deteksi)', fontsize=12)
    ax.set_title('Distribusi False Negative per Token (DistilBERT: Imbalance vs Balance)', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=12)

    fig.tight_layout()
    plt.savefig("assets/fn_per_label.png", dpi=300)
    plt.close()
    print("Saved assets/fn_per_label.png")
else:
    print("Metrics files not found. Skipping FP/FN plots.")

print("Done.")
