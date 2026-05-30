# HANDOVER — PII Shield

**Status:** Setup selesai. Tim siap mulai implementasi model.

**Tanggal:** 30 Mei 2026

---

## 📢 Ringkasan Status

✅ **Sudah Siap:**
- Repository struktur lengkap
- Dataset train/val/test internal sudah tersplit (70/15/15 by document)
- Label schema fix di `configs/label_schema.json`
- Script evaluasi tersedia (`src/evaluate.py`)
- Aturan format output sudah ditetapkan
- Prototype Streamlit dasar tersedia

📋 **Perlu Dikerjakan Anggota:**
- Model implementation (baseline, CRF, boosting, transformer)
- Output CSV prediksi per model
- Output metrics JSON per model
- Integrasi hasil dan laporan final

---

## 🚀 Step 1: Setup Awal

### 1.1 Clone/Sync Repository

```bash
git pull origin main
```

### 1.2 Install Dependencies

```bash
pip install -r requirements.txt
```
Download train.json dari kaggle lalu masukan ke data/processed

### 1.3 Verifikasi Data

```bash
python3 src/verify_split.py
```

Output yang diharapkan: 
- Train: 4,764 documents
- Val: 1,021 documents
- Test: 1,022 documents
- ✓ Split verification PASSED

**Jika ada error**, jangan lanjut — hubungi Qurrata.

---

## ⚠️ Aturan Global

### Larangan Keras

❌ **JANGAN** split dataset sendiri-sendiri
❌ **JANGAN** modifikasi `data/processed/` atau `configs/label_schema.json`
❌ **JANGAN** ubah format BIO label
❌ **JANGAN** hanya pakai accuracy sebagai metrik

### Wajib

✅ **HARUS** pakai data dari `data/processed/` (train/val/test_internal)
✅ **HARUS** output CSV prediksi format: `document_id,token,true_label,pred_label`
✅ **HARUS** output metrics JSON sesuai template
✅ **HARUS** pakai label schema dari `configs/label_schema.json`
✅ **HARUS** gunakan evaluasi script dari `src/evaluate.py`

---

## 👤 Instruksi Per Anggota

### Nau — Baseline ML

**Models:** Logistic Regression, Linear SVM

**Folder:** `models/baseline/`

#### Setup

1. **Ekstrak features dari tokens:**
   - Token length
   - Character features (isdigit, isupper, istitle, etc)
   - Embedding (Word2Vec, GloVe, FastText, atau TF-IDF)
   - Context tokens (prev/next token)

   **Tips:** Lihat `notebooks/01_data_exploration.md` untuk contoh feature engineering

2. **Load data:**
   ```python
   from src.data_loader import DataLoader
   
   loader = DataLoader()
   train_data = loader.load_raw_json("data/processed/train.json")
   val_data = loader.load_raw_json("data/processed/val.json")
   test_data = loader.load_raw_json("data/processed/test_internal.json")
   ```

3. **Train model:**
   - Logistic Regression → token-level classifier
   - Linear SVM → token-level classifier

4. **Prediksi pada test set:**
   - Output format: (document_id, token, true_label, pred_label)

#### Output Wajib

**CSV Prediksi:**
```
results/predictions/logistic_regression_predictions.csv
results/predictions/linear_svm_predictions.csv
```

**Metrics JSON:**
```
results/metrics/logistic_regression_metrics.json
results/metrics/linear_svm_metrics.json
```

**Deskripsi model:**
```
models/baseline/README.md
```
Berisi: method, feature engineering, hyperparameter, hasil singkat

#### Evaluasi

```bash
python3 -c "
from src.evaluate import evaluate_from_csv, print_metrics
metrics = evaluate_from_csv('results/predictions/logistic_regression_predictions.csv', 'logistic_regression')
print_metrics(metrics)
"
```

---

### NER / CRF

**Model:** CRF (Conditional Random Field)

**Folder:** `models/crf/`

#### Setup

1. **CRF membutuhkan sequence features:**
   - Token features (dari Naura)
   - Context window (prev/next 2 tokens)
   - Word shape patterns
   - POS tags (opsional)

2. **Load data:**
   ```python
   from src.data_loader import DataLoader
   
   loader = DataLoader()
   train_data = loader.load_raw_json("data/processed/train.json")
   val_data = loader.load_raw_json("data/processed/val.json")
   test_data = loader.load_raw_json("data/processed/test_internal.json")
   ```

3. **Library Options:**
   - `python-crfsuite` (simple, recommended)
   - `pycrfsuite`

4. **Training:**
   - Train pada train.json
   - Tune hyperparameter di val.json
   - Test pada test_internal.json

#### Output Wajib

**CSV Prediksi:**
```
results/predictions/crf_predictions.csv
```

**Metrics JSON:**
```
results/metrics/crf_metrics.json
```

**Deskripsi model:**
```
models/crf/README.md
```

#### Evaluasi

```bash
python3 -c "
from src.evaluate import evaluate_from_csv, print_metrics
metrics = evaluate_from_csv('results/predictions/crf_predictions.csv', 'crf')
print_metrics(metrics)
"
```

---

### Boosting Models

**Models:** XGBoost / LightGBM

**Folder:** `models/boosting/`

#### Setup

1. **Feature engineering** (bisa pakai dari Naura + tambahan):
   - Token features
   - Context features
   - Linguistic features
   - Pattern matching (email, URL, number patterns)

2. **Load data:**
   ```python
   from src.data_loader import DataLoader
   
   loader = DataLoader()
   train_data = loader.load_raw_json("data/processed/train.json")
   val_data = loader.load_raw_json("data/processed/val.json")
   test_data = loader.load_raw_json("data/processed/test_internal.json")
   ```

3. **Handle Class Imbalance:**
   - Gunakan `scale_pos_weight` atau `class_weight`
   - Jangan pakai accuracy, fokus F1/precision/recall

4. **Training:**
   - Train pada train.json
   - Early stopping di val.json
   - Test pada test_internal.json

#### Output Wajib

**CSV Prediksi:**
```
results/predictions/xgboost_predictions.csv
results/predictions/lightgbm_predictions.csv
```

**Metrics JSON:**
```
results/metrics/xgboost_metrics.json
results/metrics/lightgbm_metrics.json
```

**Deskripsi model:**
```
models/boosting/README.md
```

#### Evaluasi

```bash
python3 -c "
from src.evaluate import evaluate_from_csv, print_metrics
metrics = evaluate_from_csv('results/predictions/xgboost_predictions.csv', 'xgboost')
print_metrics(metrics)
"
```

---

### Transformer + Hybrid

**Models:** DistilBERT / DeBERTa + Hybrid Rule+ML

**Folder:** `models/transformer/`

#### Setup: DistilBERT / DeBERTa

1. **Library:**
   - `transformers` (HuggingFace)
   - `torch`

2. **Model Architecture:**
   - Use `DistilBERT` atau `DeBERTa` sebagai encoder
   - Add token classification head
   - Fine-tune pada PII task

3. **Load data:**
   ```python
   from src.data_loader import DataLoader
   
   loader = DataLoader()
   train_data = loader.load_raw_json("data/processed/train.json")
   val_data = loader.load_raw_json("data/processed/val.json")
   test_data = loader.load_raw_json("data/processed/test_internal.json")
   ```

4. **Training:**
   - Tokenize dengan model's tokenizer
   - Handle BIO labels properly
   - Use weighted loss untuk class imbalance
   - Early stopping pada val set

#### Setup: Hybrid Rule + ML

1. **Rule-based layer:**
   - Regex untuk email pattern (`.*@.*\..*`)
   - Regex untuk URL pattern (`https?://.*`)
   - Regex untuk phone pattern (`\d{3}-\d{3}-\d{4}`)
   - Regex untuk ID number pattern (`\d{5,}`)

2. **Kombinasi dengan model:**
   - If rule matches → use rule prediction
   - Else → use model prediction
   - Evaluasi mana yang lebih baik

#### Output Wajib

**CSV Prediksi:**
```
results/predictions/distilbert_predictions.csv
results/predictions/hybrid_predictions.csv
```

**Metrics JSON:**
```
results/metrics/distilbert_metrics.json
results/metrics/hybrid_metrics.json
```

**Deskripsi model:**
```
models/transformer/README.md
```

#### Evaluasi

```bash
python3 -c "
from src.evaluate import evaluate_from_csv, print_metrics
metrics = evaluate_from_csv('results/predictions/distilbert_predictions.csv', 'distilbert')
print_metrics(metrics)
"
```

---

## 📊 Format Output Yang BENAR

### CSV Prediksi

**Lokasi:** `results/predictions/[model_name]_predictions.csv`

**Format:**
```csv
document_id,token,true_label,pred_label
1,My,O,O
1,name,O,O
1,is,O,O
1,Michael,B-NAME_STUDENT,B-NAME_STUDENT
1,Jordan,I-NAME_STUDENT,I-NAME_STUDENT
1,and,O,O
1,my,O,O
1,email,O,O
1,is,O,O
1,mjordan@nba.com,B-EMAIL,B-EMAIL
```

**Penting:**
- Header HARUS ada
- Satu token per row
- Urut per document_id
- Jangan ada baris kosong

### Metrics JSON

**Lokasi:** `results/metrics/[model_name]_metrics.json`

**Format:** Gunakan template dari `results/metrics_template.json`

Contoh:
```json
{
  "model": "logistic_regression",
  "total_tokens": 764041,
  "token_level": {
    "precision": 0.65,
    "recall": 0.58,
    "f1": 0.61,
    "per_class": {
      "B-NAME_STUDENT": {"precision": 0.75, "recall": 0.68, "f1": 0.71},
      "I-NAME_STUDENT": {"precision": 0.70, "recall": 0.62, "f1": 0.66},
      ...
    }
  },
  "entity_level": {
    "precision": 0.60,
    "recall": 0.52,
    "f1": 0.56
  }
}
```

---

## 🔗 Script Evaluasi Buat Semua

Setelah output CSV dan JSON, bisa evaluate dengan:

```bash
python3 src/integrate_predictions.py
```

Script ini akan:
1. Cari semua file CSV di `results/predictions/`
2. Jalankan evaluasi untuk masing-masing
3. Generate tabel perbandingan final
4. Output ke `results/final_comparison.md`

---

## 🆘 Troubleshooting & FAQ

### Q: Data shape tidak cocok?
A: Pastikan jumlah tokens = jumlah labels di setiap dokumen.
```python
for doc in data:
    assert len(doc['tokens']) == len(doc['labels']), \
        f"Doc {doc['document']}: {len(doc['tokens'])} tokens vs {len(doc['labels'])} labels"
```

### Q: Model training sangat lambat?
A: 
- Gunakan subset untuk debugging awal (misalnya first 100 docs)
- Pakai GPU jika ada
- Reduce feature dimensionality

### Q: F1 score sangat rendah?
A: Kemungkinan:
- Label O terlalu dominan (99.94%) → gunakan weighted metrics
- Feature tidak cukup baik
- Hyperparameter belum optimal

### Q: Format output salah?
A: Gunakan script helper:
```python
import csv
def save_predictions_csv(predictions, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['document_id', 'token', 'true_label', 'pred_label'])
        writer.writeheader()
        writer.writerows(predictions)
```

### Q: Preprocessing apa yang boleh dipakai?
A: 
- ✅ Ringan: normalisasi whitespace, HTML entities
- ⚠️ Hati-hati: lowercasing, punctuation removal
- ❌ Jangan: stopword removal (bisa hapus PII info)

Gunakan `src/preprocessing.py` sebagai referensi.

### Q: Bagaimana kalau ada bug di script evaluasi?
A: Hubungi Qurrata (5025241031) atau cek `src/evaluate.py` untuk validasi script.

---

## 📚 Dokumen Penting

- **README.md** — Overview project, struktur repo, alasan desain
- **readme-chore.md** — Progress, timeline, Definition of Done
- **SETUP.md** — Panduan setup lengkap + per-model
- **src/data_loader.py** — Dokumentasi load/save data
- **src/evaluate.py** — Dokumentasi evaluasi metrics
- **notebooks/01_data_exploration.md** — Contoh explorasi data

---

## 📅 Timeline Singkat

**Week 1 (Hari 1-7):**
- Hari 1-2: Setup, verifikasi data (semua anggota)
- Hari 3-4: Baseline ML + CRF awal
- Hari 5-7: Boosting + Transformer start

**Week 2 (Hari 8-14):**
- Hari 8-10: Tuning + eksperimen lanjutan
- Hari 11-12: Integrasi hasil + komparasi final
- Hari 13-14: Laporan + finalisasi

---

## ✅ Checklist Sebelum Submit

Setiap anggota sebelum kumpulkan hasil, pastikan:

- [ ] CSV prediksi sudah di-generate
- [ ] CSV format benar: `document_id,token,true_label,pred_label`
- [ ] CSV sudah di-cek (tidak ada baris kosong)
- [ ] Metrics JSON sudah di-generate
- [ ] Metrics JSON sesuai template
- [ ] Script evaluasi bisa dijalankan tanpa error
- [ ] F1 score sudah dicek (jangan hanya accuracy)
- [ ] Deskripsi model sudah ditulis (method + kendala)

---

## 🎯 Kontak

- **Qurrata (Project Lead):** 5025241031
- **Group Chat:** [sesuai dengan platform komunikasi tim]

Jika ada pertanyaan atau error, jangan ragu tanya di group atau langsung ke Qurrata.

---

**Good luck! 🚀**

Last updated: 30 Mei 2026
