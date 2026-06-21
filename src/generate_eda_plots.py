import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import Counter

# Set style
sns.set_theme(style="whitegrid")

print("Loading data...")
with open("data/raw/imbalance/train.json", "r") as f:
    data = json.load(f)

# 1. Document lengths
doc_lengths = [len(doc["tokens"]) for doc in data]

plt.figure(figsize=(10, 6))
sns.histplot(doc_lengths, bins=50, kde=True, color="skyblue")
plt.title("Distribusi Panjang Dokumen (Jumlah Token per Esai)", fontsize=14)
plt.xlabel("Jumlah Token", fontsize=12)
plt.ylabel("Frekuensi", fontsize=12)
plt.axvline(np.mean(doc_lengths), color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {np.mean(doc_lengths):.0f}')
plt.axvline(np.median(doc_lengths), color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {np.median(doc_lengths):.0f}')
plt.legend()
plt.tight_layout()
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/doc_length_dist.png", dpi=300)
plt.close()

# 2. Label Distribution (excluding O)
all_labels = [label for doc in data for label in doc["labels"] if label != "O"]
label_counts = Counter(all_labels)

# Sort by count
labels, counts = zip(*label_counts.most_common())

plt.figure(figsize=(12, 6))
sns.barplot(x=list(counts), y=list(labels), hue=list(labels), palette="viridis", legend=False)
plt.title("Distribusi Label PII (Tanpa Kelas 'O')", fontsize=14)
plt.xlabel("Jumlah Token (Log Scale)", fontsize=12)
plt.ylabel("Label", fontsize=12)
plt.xscale('log') # Log scale because of extreme imbalance even among PIIs
for i, v in enumerate(counts):
    plt.text(v, i, f" {v}", va='center', fontsize=10)
plt.tight_layout()
plt.savefig("assets/label_dist_log.png", dpi=300)
plt.close()

print("Plots saved to assets/")
