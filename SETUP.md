# PII Shield - Setup & Usage Guide

## 📋 Overview

Struktur project sudah siap. Berikut adalah panduan penggunaan untuk setiap anggota tim.

---

## ✅ Struktur Folder

```
pii-shield/
├── configs/
│   └── label_schema.json              ← Label mapping (DO NOT CHANGE)
├── data/
│   ├── raw/                           ← Original Kaggle data
│   │   ├── train.json                 (sudah ada)
│   │   └── test.json                  (sudah ada)
│   └── processed/                     ← Processed data (siap untuk training)
│       ├── train.json                 ✓ Siap
│       ├── val.json                   ✓ Siap
│       └── test_internal.json         ✓ Siap
├── src/
│   ├── data_loader.py                 ← Load dataset
│   ├── preprocessing.py               ← Preprocessing (ringan, opsional)
│   ├── evaluate.py                    ← Evaluasi metrics
│   ├── utils.py                       ← Shared utilities
│   ├── split_data.py                  ← Script split (sudah dijalankan)
│   └── verify_split.py                ← Script verify (sudah dijalankan)
├── models/
│   ├── baseline/                      ← Untuk Naura (Logistic Reg, SVM)
│   ├── crf/                           ← Untuk Farras
│   ├── boosting/                      ← Untuk Erica
│   └── transformer/                   ← Untuk Naufal
├── notebooks/
│   └── [empty]                        ← Jupyter notebooks per model
├── results/
│   ├── metrics/
│   │   └── metrics_template.json      ← Template untuk metrics
│   ├── predictions/                   ← CSV predictions per model
│   └── final_comparison.md            ← Perbandingan final
├── app/
│   └── streamlit_app.py               ← Demo web app
├── README.md                          ← Dokumentasi project
├── SETUP.md                           ← File ini
└── requirements.txt                   ← Dependencies
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Data Split

Dataset sudah di-split menjadi train/val/test. Untuk verify:

```bash
python3 src/verify_split.py
```

Output:
- Train: 4,764 documents (69.7% tokens)
- Val: 1,021 documents (15.0% tokens)
- Test: 1,022 documents (15.3% tokens)
- **Penting**: Label `O` sangat dominan (99.94%)!

### 3. Load Data untuk Model

```python
from src.data_loader import DataLoader

loader = DataLoader()

# Load processed data
train_data = loader.load_raw_json("data/processed/train.json")
val_data = loader.load_raw_json("data/processed/val.json")
test_data = loader.load_raw_json("data/processed/test_internal.json")

print(f"Train: {len(train_data)} documents")
print(f"Val: {len(val_data)} documents")
print(f"Test: {len(test_data)} documents")
```

Struktur setiap document:
```json
{
  "document": 7,
  "full_text": "...",
  "tokens": [...],
  "labels": [...],       # BIO format
  "trailing_whitespace": [...]
}
```

---

## 📊 Dataset Statistics

**Keseimbangan label sangat tidak seimbang:**

| Label | Count | Percentage |
|-------|-------|------------|
| O | 3,476,574 | 99.94% |
| B-NAME_STUDENT | 961 | 0.03% |
| I-NAME_STUDENT | 760 | 0.02% |
| B-URL_PERSONAL | 70 | 0.00% |
| Lainnya | 152 | 0.01% |

**Implikasi:**
- ✅ Jangan gunakan **accuracy** sebagai metrik (bias terhadap O)
- ✅ Gunakan **precision, recall, F1-score**
- ✅ **PII labels sangat jarang**, jadi model perlu careful tuning

---

## 🔧 Preprocessing Pipeline

Ada dua tingkat preprocessing:

### Level 1: Ringan (Default)
```python
from src.preprocessing import Preprocessor

text = "My email is john@example.com"
cleaned = Preprocessor.basic_preprocessing(text)
```

**Yang dilakukan:**
- Normalize whitespace
- Normalize HTML entities

**Yang TIDAK dilakukan:**
- Lowercasing
- Punctuation removal
- Stopword removal

### Level 2: Opsional per Model
```python
# Jika model Anda butuh lowercase:
text = Preprocessor.lowercase(text)

# ⚠️ BERBAHAYA untuk PII - jangan gunakan untuk default:
# text = Preprocessor.remove_punctuation(text)  # Hapus @ . dari email!
```

**Penting:**
- Email: butuh `@` dan `.`
- URL: butuh `/`, `.`, `:`
- Nama: sensitif case
- Alamat: butuh angka dan tanda baca

---

## 📈 Evaluation Metrics

Gunakan script `src/evaluate.py` untuk evaluasi:

```python
from src.evaluate import Evaluator, print_metrics

y_true = [...] # True labels
y_pred = [...] # Predicted labels

metrics = Evaluator.evaluate(y_true, y_pred)
print_metrics({**metrics, 'model': 'my_model', 'total_tokens': len(y_true)})
```

Output metrics:
- **Token-level**: Precision, Recall, F1 per label
- **Entity-level**: Exact match untuk entities

---

## 📝 Output Format Wajib

### 1. Prediction File (CSV)

Setiap model HARUS output file CSV dengan format:

```csv
document_id,token,true_label,pred_label
1,My,O,O
1,name,O,O
1,is,O,O
1,Michael,B-NAME_STUDENT,B-NAME_STUDENT
1,Jordan,I-NAME_STUDENT,I-NAME_STUDENT
```

**Simpan ke:** `results/predictions/[model_name]_predictions.csv`

Contoh:
```
results/predictions/logistic_regression_predictions.csv
results/predictions/linear_svm_predictions.csv
results/predictions/crf_predictions.csv
results/predictions/xgboost_predictions.csv
results/predictions/distilbert_predictions.csv
```

### 2. Metrics File (JSON)

```json
{
  "model": "logistic_regression",
  "precision": 0.75,
  "recall": 0.68,
  "f1": 0.71,
  "per_class_f1": {
    "NAME_STUDENT": 0.75,
    "EMAIL": 0.50,
    ...
  }
}
```

**Simpan ke:** `results/metrics/[model_name]_metrics.json`

Gunakan template di `results/metrics_template.json`.

---

## 🎯 Per-Model Instructions

### Naura (Baseline ML)

**Models:** Logistic Regression, Linear SVM

**Output folder:** `models/baseline/`

Steps:
1. Load train/val/test dari `data/processed/`
2. Extract token-level features (TF-IDF, embeddings, dll)
3. Train model
4. Predict pada test set
5. Output ke `results/predictions/logistic_regression_predictions.csv` dan `.../linear_svm_predictions.csv`
6. Evaluasi dengan `src/evaluate.py`

---

### Farras (NER / CRF)

**Model:** CRF (Conditional Random Field)

**Output folder:** `models/crf/`

Steps:
1. Load train/val/test dari `data/processed/`
2. Extract sequence features
3. Train CRF
4. Predict pada test set
5. Output ke `results/predictions/crf_predictions.csv`
6. Evaluasi dengan `src/evaluate.py`

---

### Erica (Boosting)

**Models:** XGBoost, LightGBM

**Output folder:** `models/boosting/`

Steps:
1. Load train/val/test dari `data/processed/`
2. Feature engineering (token features, contextual features, dll)
3. Train XGBoost/LightGBM
4. Predict pada test set
5. Output ke `results/predictions/xgboost_predictions.csv` dan `.../lightgbm_predictions.csv`
6. Evaluasi dengan `src/evaluate.py`

---

### Naufal (Deep Learning + Hybrid)

**Models:** DistilBERT, DeBERTa, Hybrid Rule+ML

**Output folder:** `models/transformer/`

Steps:
1. Load train/val/test dari `data/processed/`
2. Fine-tune transformer model
3. Predict pada test set
4. Output ke `results/predictions/distilbert_predictions.csv` dan `.../hybrid_predictions.csv`
5. Evaluasi dengan `src/evaluate.py`

---

## 🧪 Testing Your Code

### Test loading data:
```bash
python3 -c "from src.data_loader import DataLoader; loader = DataLoader(); data = loader.load_raw_json('data/processed/train.json'); print(f'Loaded {len(data)} documents')"
```

### Test evaluation:
```bash
python3 src/evaluate.py
```

### Test preprocessing:
```bash
python3 src/preprocessing.py
```

---

## 📦 Integration Checklist

Sebelum submit hasil, pastikan:

- [ ] Data dari `data/processed/` (jangan split sendiri)
- [ ] Label schema dari `configs/label_schema.json` (jangan hardcode)
- [ ] Prediction CSV format benar: `document_id, token, true_label, pred_label`
- [ ] Metrics JSON lengkap dengan per-class metrics
- [ ] File disimpan di folder yang benar:
  - `models/[type]/` untuk kode
  - `results/predictions/` untuk CSV
  - `results/metrics/` untuk metrics JSON
- [ ] Tidak ada data leakage (test set jangan di-train)
- [ ] Random seed konsisten untuk reproducibility

---

## 🚦 Next Steps (untuk Qurrata)

1. ✅ Setup repository
2. ✅ Create label schema
3. ✅ Split train/val/test
4. ✅ Create evaluation framework
5. [ ] **Kirim instruksi ke team** dengan link ke file ini
6. [ ] Collect predictions dari semua anggota
7. [ ] Jalankan evaluasi terpadu
8. [ ] Buat tabel perbandingan model
9. [ ] Integrate best model ke Streamlit app

---

## 🔗 Link Penting

- [README.md](README.md) - Full project documentation
- [configs/label_schema.json](configs/label_schema.json) - Label mapping
- [results/metrics_template.json](results/metrics_template.json) - Metrics template
- [Kaggle Competition](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data)

---

## ❓ FAQ

**Q: Boleh saya split data sendiri?**
A: **TIDAK**. Gunakan split yang sudah di-generate di `data/processed/`. Ini memastikan evaluasi fair dan reproducible.

**Q: Boleh saya pakai preprocessing agresif?**
A: **BOLEH**, tapi **opsional per model**. Jangan set sebagai default karena bisa merusak PII info (email, URL, dll).

**Q: Berapa seed untuk reproducibility?**
A: `RANDOM_SEED = 42` (sudah di-set di split_data.py)

**Q: Model saya F1-nya rendah, apa yang salah?**
A: Kemungkinan:
1. Label tidak seimbang (O dominan) → gunakan weighted metrics
2. Feature tidak cukup baik → improve feature engineering
3. Hyperparameter tidak optimal → tune hyperparameter
4. Data leakage → cek split dengan verify_split.py

**Q: Saya perlu input dari Qurrata, siapa yang hubungi?**
A: Hubungi Qurrata (5025241031) atau discuss di group.

---

**Terakhir diupdate:** 30 Mei 2026
**Status:** ✓ Setup Complete - Ready for Model Training
