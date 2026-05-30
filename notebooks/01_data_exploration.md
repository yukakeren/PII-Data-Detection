# PII Shield - Data Exploration Template Notebook

Gunakan notebook ini untuk explorasi data sebelum membangun model.

## 1. Import Libraries

```python
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.insert(0, '/home/ata/school/fp-ml/src')
from data_loader import DataLoader
from utils import print_dataset_stats
```

## 2. Load Data

```python
loader = DataLoader()

# Load splits
train_data = loader.load_raw_json("data/processed/train.json")
val_data = loader.load_raw_json("data/processed/val.json")
test_data = loader.load_raw_json("data/processed/test_internal.json")

print(f"Train: {len(train_data)} documents")
print(f"Val: {len(val_data)} documents")
print(f"Test: {len(test_data)} documents")
```

## 3. Explore Dataset

```python
# Print statistics
print_dataset_stats(train_data, "TRAIN")
print_dataset_stats(val_data, "VAL")
print_dataset_stats(test_data, "TEST")
```

## 4. Look at Sample Documents

```python
# First document
doc = train_data[0]
print(f"Document ID: {doc['document']}")
print(f"Number of tokens: {len(doc['tokens'])}")
print(f"Number of labels: {len(doc['labels'])}")
print()

# Print first 10 tokens and labels
print("First 10 tokens and labels:")
for i, (token, label) in enumerate(zip(doc['tokens'][:10], doc['labels'][:10])):
    print(f"  {i}: {token:20s} → {label}")
```

## 5. Label Distribution

```python
# Aggregate all labels from train set
label_counts = {}
for doc in train_data:
    for label in doc['labels']:
        label_counts[label] = label_counts.get(label, 0) + 1

# Sort by count
sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

print("Label distribution (Train):")
for label, count in sorted_labels:
    pct = count / sum(label_counts.values()) * 100
    print(f"  {label:20s}: {count:7,} ({pct:6.2f}%)")
```

## 6. Document Length Distribution

```python
# Document lengths
train_lengths = [len(doc['tokens']) for doc in train_data]
val_lengths = [len(doc['tokens']) for doc in val_data]

print(f"Train lengths: min={min(train_lengths)}, max={max(train_lengths)}, mean={np.mean(train_lengths):.1f}")
print(f"Val lengths: min={min(val_lengths)}, max={max(val_lengths)}, mean={np.mean(val_lengths):.1f}")

# Plot distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(train_lengths, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Document length (tokens)')
axes[0].set_ylabel('Count')
axes[0].set_title('Train set document lengths')

axes[1].hist(val_lengths, bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[1].set_xlabel('Document length (tokens)')
axes[1].set_ylabel('Count')
axes[1].set_title('Val set document lengths')

plt.tight_layout()
plt.show()
```

## 7. Find PII Examples

```python
# Find documents dengan PII
pii_examples = []
for doc in train_data[:1000]:  # Check first 1000
    if any(label != 'O' for label in doc['labels']):
        pii_examples.append(doc)

print(f"Found {len(pii_examples)} documents with PII in first 1000")

# Show one example
if pii_examples:
    doc = pii_examples[0]
    print(f"\nExample document with PII:")
    print(f"ID: {doc['document']}")
    print(f"Text: {doc['full_text'][:200]}...")
    
    # Find PII tokens
    pii_tokens = [(token, label) for token, label in zip(doc['tokens'], doc['labels']) if label != 'O']
    print(f"\nPII entities:")
    for token, label in pii_tokens[:10]:
        print(f"  {token:30s} → {label}")
```

## 8. Feature Ideas

```python
# Contoh features yang bisa digunakan:
print("Feature ideas untuk token classification:")
print()
print("1. Token features:")
print("   - Token length")
print("   - Is digit: token.isdigit()")
print("   - Is upper: token.isupper()")
print("   - Is title: token.istitle()")
print("   - Contains special char: any(c in token for c in '@.-_')")
print()
print("2. Context features:")
print("   - Previous token")
print("   - Next token")
print("   - Previous label (for sequence models)")
print()
print("3. Linguistic features:")
print("   - POS tags")
print("   - Named entity tags")
print("   - Word embeddings (GloVe, Word2Vec, FastText)")
print()
print("4. Pattern matching:")
print("   - Email pattern: contains @")
print("   - URL pattern: contains ://")
print("   - Phone pattern: mostly digits")
print("   - ID number: sequence of digits")
```

## 9. Class Imbalance Analysis

```python
# Calculate class weights (untuk handling imbalance)
from collections import Counter

label_counts = Counter()
for doc in train_data:
    for label in doc['labels']:
        label_counts[label] += 1

total = sum(label_counts.values())
class_weights = {label: total / (len(label_counts) * count) 
                 for label, count in label_counts.items()}

print("Class weights (untuk imbalanced data):")
for label, weight in sorted(class_weights.items(), key=lambda x: x[1], reverse=True):
    print(f"  {label:20s}: {weight:.2f}")
```

## 10. Save Processed Data

```python
# Jika Anda create custom processed data
import os

# Example: Extract features dan save
processed_data = []
for doc in train_data:
    processed_doc = {
        'document': doc['document'],
        'tokens': doc['tokens'],
        'labels': doc['labels'],
        'features': [...]  # Your custom features
    }
    processed_data.append(processed_doc)

# Save
output_path = "data/processed/train_with_features.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(processed_data, f)

print(f"Saved to {output_path}")
```

---

**Tips:**
- Gunakan notebook ini untuk eksperimen
- Jangan modify data di `data/processed/`
- Save hasil eksperimen ke folder model (e.g., `models/baseline/`)
- Commit notebook Anda ke git

**Next:** Mulai build model di folder model Anda!
