# PII Shield: Deteksi dan Redaksi Data Pribadi pada Teks Pendidikan

## 1. Overview Project

Project ini bertujuan untuk membangun sistem deteksi dan redaksi Personally Identifiable Information (PII) pada teks pendidikan, khususnya teks esai siswa.

Dataset utama yang digunakan berasal dari Kaggle Competition:

**PII Detection Removal from Educational Data**

Task utama project ini adalah:

- Token Classification
- Named Entity Recognition (NER)
- PII Detection
- PII Redaction / Anonymization

Contoh input:

```text
My name is Michael Jordan and my email is mjordan@nba.com
```

Contoh output token classification:

```text
Michael B-NAME_STUDENT
Jordan I-NAME_STUDENT
mjordan@nba.com B-EMAIL
```

Contoh output redacted:

```text
My name is [NAME_STUDENT] and my email is [EMAIL]
```

---

## 2. Current Project Status

### Sudah Selesai

Beberapa keputusan awal project sudah ditentukan:

* [x] Topik project sudah ditentukan
* [x] Dataset utama sudah ditentukan
* [x] Task sudah ditentukan sebagai Token Classification / NER
* [x] Label PII mengikuti label dari Kaggle competition
* [x] Model yang akan diuji sudah ditentukan
* [x] Pembagian jobdesk anggota sudah dibuat
* [x] Scope project sudah dibatasi agar realistis untuk 2 minggu
* [x] Output akhir sudah ditentukan: highlight PII dan redacted text
* [x] Metrik utama sudah diarahkan ke precision, recall, dan F1-score

---

## 3. Dataset

### Dataset Utama

Dataset utama:

```text
Kaggle Competition: PII Detection Removal from Educational Data
```

Dataset ini sudah menyediakan data teks esai siswa yang diberi anotasi PII pada level token.

### Catatan Penting Dataset

Kaggle menyediakan train dan test data. Namun, test set Kaggle biasanya tidak memiliki label ground truth publik.

Karena itu, untuk eksperimen internal dan laporan, kita tetap perlu membuat split sendiri dari data training yang berlabel.

Skema yang digunakan:

```text
Kaggle train data berlabel
→ train_internal
→ validation_internal
→ test_internal
```

Kaggle test data digunakan untuk:

* inference
* demo
* simulasi prediksi
* prototype aplikasi

Bukan sebagai evaluasi utama, kecuali label ground truth tersedia.

---

## 4. Target Label

Label mengikuti competition Kaggle.

Daftar kelas PII:

```text
NAME_STUDENT
EMAIL
USERNAME
ID_NUM
PHONE_NUM
URL_PERSONAL
STREET_ADDRESS
```

Selain itu terdapat label:

```text
O
```

untuk token non-PII.

### Format Label

Project ini menggunakan format BIO tagging:

```text
B-NAME_STUDENT
I-NAME_STUDENT
B-EMAIL
I-EMAIL
B-USERNAME
I-USERNAME
B-ID_NUM
I-ID_NUM
B-PHONE_NUM
I-PHONE_NUM
B-URL_PERSONAL
I-URL_PERSONAL
B-STREET_ADDRESS
I-STREET_ADDRESS
O
```

Catatan:

* `B-` berarti beginning of entity
* `I-` berarti inside of entity
* `O` berarti outside / bukan PII

---

## 5. Scope Project

Agar project realistis dikerjakan dalam waktu sekitar 2 minggu, scope dibatasi sebagai berikut:

| Bagian       | Scope                                             |
| ------------ | ------------------------------------------------- |
| Dataset      | Dataset Kaggle + opsional data buatan             |
| Tipe Data    | Text                                              |
| Task         | Token Classification / Named Entity Recognition   |
| Model        | ML baseline, CRF, boosting, deep learning, hybrid |
| Implementasi | Prototype Streamlit sederhana                     |
| Output       | Highlight PII dan redacted text                   |
| Evaluasi     | Precision, Recall, F1-score                       |

---

## 6. Model yang Digunakan

Model yang akan diuji:

| No | Model                  | Jenis          | Keterangan                             |
| -- | ---------------------- | -------------- | -------------------------------------- |
| 1  | Logistic Regression    | Baseline ML    | Model baseline sederhana               |
| 2  | Linear SVM             | Baseline ML    | Cocok untuk fitur token/text           |
| 3  | CRF                    | Sequence Model | Cocok untuk NER/token sequence         |
| 4  | XGBoost / LightGBM     | Boosting       | Model berbasis boosting                |
| 5  | DistilBERT / DeBERTaV3 | Deep Learning  | Transformer-based token classification |
| 6  | Hybrid Rule + ML       | Hybrid         | Regex/rule-based + ML                  |
| 7  | Voting Ensemble        | Ensemble       | Kombinasi beberapa model terbaik       |

---

## 7. Pembagian Jobdesk

| Anggota                | NRP        | Role                                         | Tugas Teknis                                                                                                              |
| ---------------------- | ---------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Qurrata A'yun Kamil    | 5025241031 | Project Lead, Pipeline, Integrasi Eksperimen | Dataset processing, tokenization, label processing, split internal, integrasi hasil eksperimen, evaluasi akhir, prototype |
| Naura Taqiyya Mazi     | 5025241119 | Baseline Machine Learning                    | Logistic Regression, Linear SVM, analisis baseline model                                                                  |
| Farras Nizar           | 5025241143 | NER / Token Classification                   | CRF, sequence labeling, evaluasi token-level                                                                              |
| Naufal Daffa Alfa Zain | 5025241066 | Deep Learning / Anonymization                | DistilBERT/DeBERTa, Hybrid Rule + ML, anonymization                                                                       |
| Erica Triana           | 5025241069 | Boosting-based Model                         | XGBoost/LightGBM, hyperparameter tuning sederhana                                                                         |

---

## 8. Tugas Utama Qurrata

Sebagai project lead, Qurrata bertanggung jawab memastikan semua anggota bisa bekerja dengan format data, label, dan evaluasi yang sama.

### 8.1 Dataset Preparation

Yang perlu dilakukan:

* [ ] Download dataset Kaggle
* [ ] Simpan dataset ke folder `data/raw/`
* [ ] Pahami struktur file Kaggle
* [ ] Pastikan field token, label, dan document id terbaca dengan benar
* [ ] Buat script loading dataset

Output yang diharapkan:

```text
data/raw/
data/processed/
src/data_loader.py
```

---

### 8.2 Label Processing

Yang perlu dilakukan:

* [ ] Ambil label dari dataset Kaggle
* [ ] Pastikan format BIO tagging benar
* [ ] Buat mapping label ke id
* [ ] Buat mapping id ke label
* [ ] Simpan label schema

Output yang diharapkan:

```text
configs/label_schema.json
```

Contoh:

```json
{
  "O": 0,
  "B-NAME_STUDENT": 1,
  "I-NAME_STUDENT": 2,
  "B-EMAIL": 3,
  "I-EMAIL": 4
}
```

---

### 8.3 Internal Train/Validation/Test Split

Walaupun Kaggle sudah menyediakan train dan test, project ini tetap perlu split internal dari train data berlabel.

Yang perlu dilakukan:

* [ ] Gunakan train data Kaggle sebagai sumber utama
* [ ] Split menjadi train, validation, dan internal test
* [ ] Pastikan split dilakukan berdasarkan document id, bukan token random
* [ ] Gunakan random seed tetap agar eksperimen reproducible
* [ ] Semua anggota wajib memakai split yang sama

Rekomendasi split:

```text
Train      : 70%
Validation : 15%
Test       : 15%
```

Output yang diharapkan:

```text
data/processed/train.json
data/processed/val.json
data/processed/test_internal.json
```

atau:

```text
data/processed/train.conll
data/processed/val.conll
data/processed/test_internal.conll
```

---

### 8.4 Preprocessing Pipeline

Preprocessing dilakukan ringan agar informasi PII tidak hilang.

Yang boleh dilakukan:

* [ ] Tokenization
* [ ] Extra whitespace removal
* [ ] Newline normalization
* [ ] HTML/special character normalization jika diperlukan

Yang perlu hati-hati:

* Lowercasing
* Punctuation removal
* Stopword removal

Catatan penting:

Untuk task PII detection, jangan sembarangan menghapus kapital, tanda baca, atau stopword karena bisa merusak informasi penting.

Contoh:

* Email membutuhkan `@` dan `.`
* URL membutuhkan `/`, `.`, `:`
* Nama orang sering bergantung pada kapitalisasi
* Alamat bisa bergantung pada angka dan tanda baca

Jadi, preprocessing harus disesuaikan per model.

**⚠️ PENTING:** Lowercasing, punctuation removal, dan stopword removal bukan preprocessing wajib global. Fitur-fitur tersebut harus dijadikan **opsional per model**, bukan default. Beberapa model mungkin memerlukan teks asli, sementara model lain bisa memanfaatkan preprocessing lebih agresif. Keputusan preprocessing sebaiknya diambil per model berdasarkan eksperimen.

Output yang diharapkan:

```text
src/preprocessing.py
```

---

### 8.5 Evaluation Script

Semua model harus dievaluasi dengan metrik yang sama.

Metrik utama:

* Precision
* Recall
* F1-score

Metrik tambahan:

* Token-level F1
* Entity-level F1
* Per-class F1

Accuracy tidak dijadikan metrik utama karena label `O` sangat dominan.

Yang perlu dilakukan:

* [ ] Buat script evaluasi umum
* [ ] Input berupa true labels dan predicted labels
* [ ] Output berupa metrics JSON
* [ ] Semua anggota wajib menghasilkan format output yang kompatibel

Output yang diharapkan:

```text
src/evaluate.py
results/metrics_template.json
```

Contoh output:

```json
{
  "model": "linear_svm",
  "precision": 0.0,
  "recall": 0.0,
  "f1": 0.0,
  "per_class_f1": {
    "NAME_STUDENT": 0.0,
    "EMAIL": 0.0,
    "USERNAME": 0.0,
    "ID_NUM": 0.0,
    "PHONE_NUM": 0.0,
    "URL_PERSONAL": 0.0,
    "STREET_ADDRESS": 0.0
  }
}
```

---

### 8.6 Experiment Integration

Yang perlu dilakukan:

* [ ] Tentukan format output prediksi dari setiap anggota
* [ ] Kumpulkan hasil prediksi semua model
* [ ] Jalankan evaluasi dengan script yang sama
* [ ] Buat tabel perbandingan model
* [ ] Buat analisis hasil akhir

Format prediksi yang disarankan:

```csv
document_id,token,true_label,pred_label
1,My,O,O
1,name,O,O
1,is,O,O
1,Michael,B-NAME_STUDENT,B-NAME_STUDENT
1,Jordan,I-NAME_STUDENT,I-NAME_STUDENT
```

Output yang diharapkan:

```text
results/
├── logistic_regression_predictions.csv
├── linear_svm_predictions.csv
├── crf_predictions.csv
├── xgboost_predictions.csv
├── distilbert_predictions.csv
├── metrics_summary.csv
└── final_comparison.md
```

---

### 8.7 Prototype Streamlit

Prototype digunakan untuk demo akhir.

Fitur minimal:

* [ ] User bisa input teks manual
* [ ] Sistem mendeteksi PII
* [ ] Sistem menampilkan teks dengan highlight PII
* [ ] Sistem menampilkan redacted text
* [ ] Sistem menampilkan daftar entitas PII yang ditemukan

Fitur opsional:

* [ ] User bisa input link web
* [ ] Sistem scraping teks dari link
* [ ] Sistem melakukan privacy filtering pada teks web

Output yang diharapkan:

```text
app/streamlit_app.py
```

---

## 9. Struktur Repository

Struktur repository yang disarankan:

```text
pii-shield/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── label_schema.json
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── train.json
│       ├── val.json
│       └── test_internal.json
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_ml.ipynb
│   ├── 03_crf_model.ipynb
│   ├── 04_boosting_model.ipynb
│   └── 05_transformer_model.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── evaluate.py
│   └── utils.py
│
├── models/
│   ├── baseline/
│   ├── crf/
│   ├── boosting/
│   └── transformer/
│
├── results/
│   ├── predictions/
│   ├── metrics/
│   └── final_comparison.md
│
└── app/
    └── streamlit_app.py
```

---

## 10. Output yang Harus Dikumpulkan Setiap Anggota

Setiap anggota wajib mengumpulkan:

```text
1. Source code / notebook
2. File prediksi
3. Metrics JSON
4. Penjelasan singkat model
5. Kendala / catatan eksperimen
```

Format minimal:

```text
results/
├── predictions/
│   └── nama_model_predictions.csv
│
└── metrics/
    └── nama_model_metrics.json
```

---

## 11. Format Prediksi Wajib

Setiap model harus menghasilkan file prediksi dengan format:

```csv
document_id,token,true_label,pred_label
```

Contoh:

```csv
document_id,token,true_label,pred_label
1,My,O,O
1,name,O,O
1,is,O,O
1,Michael,B-NAME_STUDENT,B-NAME_STUDENT
1,Jordan,I-NAME_STUDENT,I-NAME_STUDENT
```

Format ini penting agar semua model bisa dievaluasi dengan script yang sama.

---

## 12. Timeline Pengerjaan

### Fase 1: Planning & Literature Review

Target:

* [x] Menentukan topik
* [x] Menentukan paper referensi
* [x] Menentukan dataset
* [x] Menentukan jobdesk
* [x] Menentukan scope

Status:

```text
Done
```

---

### Fase 2: Dataset & Pipeline Setup

Target:

* [ ] Download dataset
* [ ] Buat struktur folder project
* [ ] Buat loader dataset
* [ ] Buat label schema
* [ ] Buat internal split train/val/test
* [ ] Buat preprocessing awal
* [ ] Buat template evaluasi

Status:

```text
To Do
```

---

### Fase 3: Basic Model

Target:

* [ ] Logistic Regression selesai
* [ ] Linear SVM selesai
* [ ] CRF selesai
* [ ] Evaluasi awal selesai

PIC:

* Naura
* Farras
* Qurrata sebagai integrator

Status:

```text
To Do
```

---

### Fase 4: Boosting, Deep Learning, Hybrid

Target:

* [ ] XGBoost / LightGBM selesai
* [ ] DistilBERT / DeBERTa selesai
* [ ] Hybrid Rule + ML selesai
* [ ] Perbandingan dengan baseline selesai

PIC:

* Erica
* Naufal
* Qurrata sebagai integrator

Status:

```text
To Do
```

---

### Fase 5: Prototype

Target:

* [ ] Streamlit app selesai
* [ ] Input teks manual
* [ ] Highlight PII
* [ ] Redacted text
* [ ] List PII entity

Status:

```text
To Do
```

---

### Fase 6: Final Report

Target:

* [ ] Tabel hasil eksperimen
* [ ] Analisis performa model
* [ ] Penjelasan metode
* [ ] Screenshot prototype
* [ ] Kesimpulan
* [ ] Dokumen final

Status:

```text
To Do
```

---

## 13. Immediate Next Steps untuk Qurrata

Urutan kerja paling penting:

### Step 1: Setup Repository

Buat struktur folder:

```bash
mkdir -p configs
mkdir -p data/raw
mkdir -p data/processed
mkdir -p notebooks
mkdir -p src
mkdir -p models/baseline
mkdir -p models/crf
mkdir -p models/boosting
mkdir -p models/transformer
mkdir -p results/predictions
mkdir -p results/metrics
mkdir -p app
```

---

### Step 2: Download Dataset Kaggle

Download dataset dari Kaggle competition dan simpan di:

```text
data/raw/
```

Pastikan file utama dataset tersedia.

---

### Step 3: Buat Label Schema

Buat file:

```text
configs/label_schema.json
```

Isi awal:

```json
{
  "O": 0,
  "B-NAME_STUDENT": 1,
  "I-NAME_STUDENT": 2,
  "B-EMAIL": 3,
  "I-EMAIL": 4,
  "B-USERNAME": 5,
  "I-USERNAME": 6,
  "B-ID_NUM": 7,
  "I-ID_NUM": 8,
  "B-PHONE_NUM": 9,
  "I-PHONE_NUM": 10,
  "B-URL_PERSONAL": 11,
  "I-URL_PERSONAL": 12,
  "B-STREET_ADDRESS": 13,
  "I-STREET_ADDRESS": 14
}
```

---

### Step 4: Buat Dataset Loader

Buat file:

```text
src/data_loader.py
```

Fungsi minimal:

```python
load_raw_data()
load_processed_data()
save_processed_data()
```

---

### Step 5: Buat Internal Split

Buat script untuk split dataset:

```text
src/split_data.py
```

Target output:

```text
data/processed/train.json
data/processed/val.json
data/processed/test_internal.json
```

Gunakan seed tetap:

```python
RANDOM_SEED = 42
```

---

### Step 6: Buat Evaluation Template

Buat file:

```text
src/evaluate.py
```

Fungsi minimal:

```python
evaluate_token_classification(y_true, y_pred)
save_metrics(metrics, output_path)
```

---

### Step 7: Kirim Instruksi ke Anggota

Setelah data dan format siap, kirim ke grup:

```text
Dataset, label schema, dan split internal sudah fix.

Semua model wajib:
1. Menggunakan data dari data/processed/
2. Mengikuti label schema dari configs/label_schema.json
3. Menghasilkan prediction file dengan format:
   document_id, token, true_label, pred_label
4. Menghasilkan metrics JSON
5. Tidak split dataset sendiri-sendiri
```

---

## 14. Notes Penting

### Jangan Split Dataset Per Token

Untuk NER, split sebaiknya berdasarkan dokumen, bukan token random.

Salah:

```text
Token dari dokumen yang sama tersebar ke train dan test
```

Benar:

```text
Satu dokumen hanya masuk ke salah satu split: train, val, atau test
```

Tujuannya agar evaluasi lebih fair.

---

### Jangan Menjadikan Accuracy sebagai Metrik Utama

Karena mayoritas token adalah `O`, model bisa terlihat bagus hanya dengan sering memprediksi `O`.

Metrik utama:

```text
Precision
Recall
F1-score
```

---

### Preprocessing Jangan Terlalu Agresif

Untuk PII detection, preprocessing agresif bisa merusak informasi.

Contoh yang berisiko:

```text
Lowercasing semua token
Menghapus tanda baca
Menghapus stopword
```

Gunakan preprocessing ringan terlebih dahulu.

---

## 15. Definition of Done

Project dianggap siap untuk implementasi model jika:

* [ ] Dataset sudah ada di `data/raw/`
* [ ] Dataset processed sudah ada di `data/processed/`
* [ ] Label schema sudah ada di `configs/label_schema.json`
* [ ] Split train/val/test sudah fix
* [ ] Script evaluasi sudah tersedia
* [ ] Format prediksi sudah disepakati
* [ ] Semua anggota sudah tahu file mana yang harus digunakan

Project dianggap selesai jika:

* [ ] Semua model sudah dijalankan
* [ ] Semua model menghasilkan prediksi
* [ ] Semua model dievaluasi dengan script yang sama
* [ ] Tabel perbandingan model tersedia
* [ ] Prototype Streamlit tersedia
* [ ] Laporan akhir selesai
