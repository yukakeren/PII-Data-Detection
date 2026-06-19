# Boosting Models for PII Detection

Folder ini berisi implementasi dan hasil eksperimen boosting machine learning untuk deteksi **Personally Identifiable Information (PII)** pada level token.

Model yang digunakan:

- LightGBM
- XGBoost

Notebook utama yang digunakan untuk eksperimen:

```text
notebooks/05_boosting.ipynb
```

Notebook ini membandingkan dua skenario dataset:

- `balance`
- `imbalance`

## Method

Boosting model dibuat sebagai **token-level classifier**. Setiap token pada dokumen diubah menjadi fitur numerik, kemudian diklasifikasikan ke salah satu dari 15 BIO label PII atau label non-PII (`O`):

```text
O, B-NAME_STUDENT, I-NAME_STUDENT, B-EMAIL, I-EMAIL, B-USERNAME, I-USERNAME,
B-ID_NUM, I-ID_NUM, B-PHONE_NUM, I-PHONE_NUM, B-URL_PERSONAL, I-URL_PERSONAL,
B-STREET_ADDRESS, I-STREET_ADDRESS
```

Eksperimen dilakukan pada dua skenario dataset:

```text
data/processed/
├── balance/
│   ├── train.json
│   ├── val.json
│   └── test.json
└── imbalance/
    ├── train.json
    ├── val.json
    └── test.json
```

Pada setiap skenario:

- `train.json` digunakan untuk melatih model.
- `val.json` digunakan untuk validasi, threshold tuning, dan pemilihan model.
- `test.json` digunakan untuk evaluasi akhir skenario tersebut.

Evaluasi dilakukan dengan membandingkan `true_label` dan `pred_label` pada hasil prediksi test set.

> **Penting:** Data **TIDAK** di-balance secara manual di dalam kode. Data balance sudah tersedia sebagai pre-split sebelum training dimulai.

## Preprocessing

Tahap preprocessing terdiri dari:

- Tokenization yang sudah disediakan oleh dataset
- Lowercasing pada fitur token
- Token-level feature extraction secara manual
- Context feature extraction menggunakan token sebelum dan sesudah (window ±3)
- Character n-gram vectorization menggunakan HashingVectorizer (2–4 gram)

Stopword removal, stemming, dan lemmatization tidak digunakan karena task ini adalah PII detection / token classification. Proses tersebut dapat menghilangkan informasi penting seperti nama, email, username, ID number, URL, atau alamat.

## Feature Engineering

Setiap token diubah menjadi gabungan fitur manual (hand-crafted) dan fitur character n-gram.

Total: **106 hand-crafted features** + **32,768 character n-gram features** (HashingVectorizer)

### Handcrafted Features (106 dim)

| Kategori | Jumlah | Deskripsi |
|---|---|---|
| Char features (token utama) | 12 | length, isupper, islower, istitle, isdigit, isalpha, isalnum, has @, has ., has -/\_, has //:, has digit |
| Regex pattern matching | 12 | email, URL, phone, phone2, ID\_NUM, zip, username, username2, street\_kw, URL search, @+., starts\_digit |
| Prefix/suffix hash | 2 | hash 3-char prefix & suffix |
| Context char features | 72 | 12 char features × 6 context positions (-3,-2,-1,+1,+2,+3) |
| Keyword distance | 6 | jarak ke keyword email/phone/name/address/username/id terdekat (window=8) |
| Colon pattern | 2 | has colon before, has colon 2-back |

### Character N-gram Features (32,768 dim)

Character-level HashingVectorizer membantu model mengenali pola dalam token, misalnya:

- pola email seperti `@`, `gmail`, `.com`
- pola URL seperti `http`, `www`, `.com`
- pola ID number yang banyak mengandung angka
- pola nama atau username

Konfigurasi HashingVectorizer:

```python
HashingVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 4),
    n_features=2**15,
    norm="l2"
)
```

## Handling Class Imbalance

Dua strategi yang dibandingkan menggunakan data yang sudah di-split:

1. **Imbalance** (`data/processed/imbalance/`): Data distribusi asli. Class O sangat dominan (~99.94%). Menggunakan `sample_weight` berbasis inverse frequency saat training untuk memberi bobot lebih ke kelas minoritas.

2. **Balance** (`data/processed/balance/`): Data yang sudah di-balance secara offline. Class O sudah di-undersample dan class PII sudah di-oversample sebelum split. Data ini lebih kecil namun distribusi label lebih seimbang.

## Threshold Tuning

Semua model menggunakan **probability threshold tuning** pada validation set masing-masing.

Jika model memprediksi O tetapi probabilitas class non-O terbaik ≥ threshold, prediksi di-override ke class non-O tersebut. Strategi ini membantu model mendeteksi lebih banyak PII pada data yang sangat imbalanced.

Nilai threshold yang dicoba:

```text
[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
```

Threshold terbaik dipilih berdasarkan Val F1 (PII) tertinggi, lalu digunakan untuk prediksi test set.

## Hyperparameter

Hyperparameter adalah pengaturan model yang ditentukan sebelum training.

### LightGBM

```python
LGBMClassifier(
    n_estimators=600,
    max_depth=8,
    learning_rate=0.06,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,   # imbalance: 20, balance: 50
    reg_alpha=0.5,
    reg_lambda=0.5,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)
```

Penjelasan:

- `n_estimators=600` menentukan jumlah pohon yang dibangun. Lebih banyak pohon meningkatkan performa tetapi memperlambat training.
- `num_leaves=63` mengontrol kompleksitas setiap pohon. Nilai lebih besar menghasilkan model lebih ekspresif.
- `min_child_samples=20/50` mencegah overfitting dengan mensyaratkan minimum sampel per leaf. Diset lebih besar (50) pada data balance yang lebih kecil.
- `sample_weight` dengan power `0.5` (imbalance) dan `0.3` (balance) digunakan untuk memberi bobot lebih pada kelas minoritas PII.

### XGBoost

```python
XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,     # imbalance: 5, balance: 3
    reg_alpha=0.5,
    reg_lambda=1.0,
    tree_method="hist",
    device="cuda",
    n_jobs=-1,
    random_state=42,
    verbosity=0,
)
```

Penjelasan:

- `tree_method="hist"` dan `device="cuda"` mengaktifkan GPU acceleration untuk mempercepat training secara signifikan.
- `min_child_weight=5/3` berfungsi serupa dengan `min_child_samples` pada LightGBM untuk mencegah overfitting.
- XGBoost memerlukan **label remapping** ke rentang kontiguous `0..N-1` sebelum training karena keterbatasan internal library.
- `sample_weight` dengan power `0.5` (imbalance) dan `0.3` (balance) digunakan untuk menangani class imbalance.

## Run Training

Jalankan notebook berikut di Kaggle (GPU enabled) untuk menjalankan seluruh eksperimen dan menghasilkan model, prediksi, dan metrics:

```text
notebooks/05_boosting.ipynb
```

Pastikan dataset balance dan imbalance tersedia sebagai Kaggle dataset input dengan path:

```text
/kaggle/input/pii-balance-data/
/kaggle/input/pii-imbalance-data/
```

## Outputs

### Trained Models

Model tersimpan dalam memori sesi Kaggle selama notebook berjalan. Untuk menyimpan model ke disk, gunakan `joblib`:

```python
import joblib
joblib.dump(lgb_imb, "models/boosting/lightgbm_imbalance_model.joblib")
```

### Predictions

Output prediksi yang dihasilkan notebook:

```text
results/predictions/lightgbm_imbalance_predictions.csv
results/predictions/xgboost_imbalance_predictions.csv
results/predictions/lightgbm_balance_predictions.csv
results/predictions/xgboost_balance_predictions.csv
```

Format CSV:

```text
document_id,token,true_label,pred_label
```

### Metrics

Output metrics yang dihasilkan notebook:

```text
results/metrics/lightgbm_imbalance_metrics.json
results/metrics/xgboost_imbalance_metrics.json
results/metrics/lightgbm_balance_metrics.json
results/metrics/xgboost_balance_metrics.json
```

## Alur Data per Model

| Model | Train | Val | Test |
|---|---|---|---|
| LightGBM Imbalance | `imbalance/train.json` | `imbalance/val.json` | `imbalance/test.json` |
| XGBoost Imbalance | `imbalance/train.json` | `imbalance/val.json` | `imbalance/test.json` |
| LightGBM Balance | `balance/train.json` | `balance/val.json` | `balance/test.json` |
| XGBoost Balance | `balance/train.json` | `balance/val.json` | `balance/test.json` |

## Experiment Results

### Imbalance Scenario

| Model | Token Precision | Token Recall | Token F1 | Entity Precision | Entity Recall | Entity F1 |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM Imbalance | 0.7561 | 0.8009 | 0.7779 | 0.6031 | 0.7177 | 0.6554 |
| XGBoost Imbalance | 0.4385 | 0.8997 | 0.5896 | 0.3421 | 0.8549 | 0.4887 |

Pada skenario imbalance, **LightGBM** menjadi model terbaik dengan token F1 sebesar `0.7779` dan entity F1 sebesar `0.6554`.

### Balance Scenario

| Model | Token Precision | Token Recall | Token F1 | Entity Precision | Entity Recall | Entity F1 |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM Balance | 0.2535 | 0.6063 | 0.3575 | 0.1915 | 0.5338 | 0.2818 |
| XGBoost Balance | 0.2650 | 0.6026 | 0.3682 | 0.2044 | 0.5434 | 0.2970 |

Pada skenario balance, **XGBoost** sedikit lebih unggul dengan token F1 sebesar `0.3682` dan entity F1 sebesar `0.2970`.

## Per-Class F1 (Model Terbaik: LightGBM Imbalance)

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| B-EMAIL | 1.0000 | 1.0000 | 1.0000 |
| B-ID\_NUM | 0.8400 | 0.8400 | 0.8400 |
| B-URL\_PERSONAL | 0.7931 | 0.9200 | 0.8519 |
| I-NAME\_STUDENT | 0.7460 | 0.8345 | 0.7878 |
| B-NAME\_STUDENT | 0.7666 | 0.7594 | 0.7630 |
| B-USERNAME | 0.0000 | 0.0000 | 0.0000 |
| I-ID\_NUM | 0.0000 | 0.0000 | 0.0000 |
| I-URL\_PERSONAL | 0.0000 | 0.0000 | 0.0000 |

## Short Analysis

Berdasarkan hasil eksperimen, **LightGBM pada skenario imbalance** menjadi model terbaik secara keseluruhan dengan entity F1 sebesar `0.6554`. Model ini menghasilkan keseimbangan terbaik antara precision dan recall.

**Skenario imbalance lebih unggul dari skenario balance** pada kedua model. Hal ini terjadi karena:

- Data balance mengandung banyak false positive pada kelas yang sebenarnya tidak ada di test set (misalnya `PHONE_NUM`, `STREET_ADDRESS`, `USERNAME`). Precision sangat rendah (~0.25) menunjukkan model terlalu agresif menandai token sebagai PII.
- Data imbalance yang dikombinasikan dengan `sample_weight` lebih efektif dalam memberi bobot pada kelas minoritas tanpa membuat model terlalu agresif.

**XGBoost cenderung menghasilkan recall lebih tinggi** tetapi precision lebih rendah dibanding LightGBM pada skenario yang sama. XGBoost lebih agresif dalam mendeteksi PII (lebih banyak TP sekaligus lebih banyak FP), sedangkan LightGBM lebih konservatif dan presisi.

**Kelas yang sulit dideteksi** pada semua model adalah `USERNAME`, `PHONE_NUM`, dan `STREET_ADDRESS`. Hal ini disebabkan karena jumlah sampel kelas tersebut sangat sedikit di test set sehingga model tidak cukup belajar pola kelas tersebut.

## Conclusion

**LightGBM Imbalance** adalah model boosting terbaik pada eksperimen ini dengan entity F1 `0.6554` dan token F1 `0.7779`. Model ini dipilih sebagai model boosting final karena menghasilkan precision, recall, dan F1-score yang paling seimbang.

Skenario imbalance terbukti lebih efektif dibanding skenario balance, menunjukkan bahwa pendekatan `sample_weight` pada data distribusi asli lebih cocok untuk task PII detection dibandingkan data yang di-balance secara offline.
