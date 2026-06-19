# Boosting Models — PII Data Detection

## Metode

Token-level classification menggunakan **XGBoost** dan **LightGBM** untuk mendeteksi PII (Personally Identifiable Information) dalam teks.

Setiap token diklasifikasikan ke salah satu dari 15 BIO label:
`O, B-NAME_STUDENT, I-NAME_STUDENT, B-EMAIL, I-EMAIL, B-USERNAME, I-USERNAME, B-ID_NUM, I-ID_NUM, B-PHONE_NUM, I-PHONE_NUM, B-URL_PERSONAL, I-URL_PERSONAL, B-STREET_ADDRESS, I-STREET_ADDRESS`

## Sumber Data

Data diambil dari **2 folder terpisah** yang sudah di-split sebelumnya:

| Folder | Deskripsi | File |
|---|---|---|
| `data/processed/imbalance/` | Data distribusi asli (imbalance, class O dominan) | `train.json`, `val.json`, `test.json` |
| `data/processed/balance/` | Data yang sudah di-balance (undersample O + oversample PII) | `train.json`, `val.json`, `test.json` |

> **Penting:** Data **TIDAK** di-balance secara manual di dalam kode. Data balance sudah tersedia sebagai pre-split.

### Alur Data per Model

| Model | Train | Val | Test |
|---|---|---|---|
| LightGBM Imbalance | `imbalance/train.json` | `imbalance/val.json` | `imbalance/test.json` |
| XGBoost Imbalance | `imbalance/train.json` | `imbalance/val.json` | `imbalance/test.json` |
| LightGBM Balance | `balance/train.json` | `balance/val.json` | `balance/test.json` |
| XGBoost Balance | `balance/train.json` | `balance/val.json` | `balance/test.json` |

## Feature Engineering

Total: **106 hand-crafted features** + **32,768 character n-gram features** (HashingVectorizer)

### Hand-crafted Features (106 dim)
| Kategori | Jumlah | Deskripsi |
|---|---|---|
| Char features (token utama) | 12 | length, isupper, islower, istitle, isdigit, isalpha, isalnum, has @, has ., has -/_, has //:, has digit |
| Regex pattern matching | 12 | email, URL, phone, phone2, ID_NUM, zip, username, username2, street_kw, URL search, @+., starts_digit |
| Prefix/suffix hash | 2 | hash 3-char prefix & suffix |
| Context char features | 72 | 12 char features × 6 context positions (-3,-2,-1,+1,+2,+3) |
| Keyword distance | 6 | jarak ke keyword email/phone/name/address/username/id terdekat (window=8) |
| Colon pattern | 2 | has colon before, has colon 2-back |

### Character N-gram Features (32,768 dim)
- `HashingVectorizer(analyzer="char_wb", ngram_range=(2,4), n_features=2**15)`
- L2 normalized

## Hyperparameters

### LightGBM
| Parameter | Imbalance | Balance |
|---|---|---|
| n_estimators | 600 | 600 |
| max_depth | 8 | 8 |
| learning_rate | 0.06 | 0.06 |
| num_leaves | 63 | 63 |
| subsample | 0.8 | 0.8 |
| colsample_bytree | 0.8 | 0.8 |
| min_child_samples | 20 | 50 |
| reg_alpha | 0.5 | 0.5 |
| reg_lambda | 0.5 | 0.5 |
| sample_weight power | 0.5 | 0.3 |

### XGBoost
| Parameter | Imbalance | Balance |
|---|---|---|
| n_estimators | 500 | 500 |
| max_depth | 7 | 7 |
| learning_rate | 0.08 | 0.08 |
| subsample | 0.8 | 0.8 |
| colsample_bytree | 0.8 | 0.8 |
| min_child_weight | 5 | 3 |
| reg_alpha | 0.5 | 0.5 |
| reg_lambda | 1.0 | 1.0 |
| tree_method | hist | hist |
| device | cuda | cuda |
| sample_weight power | 0.5 | 0.3 |

## Handling Class Imbalance

Dua strategi yang dibandingkan menggunakan data yang sudah di-split:

1. **Imbalance** (`data/processed/imbalance/`): Data distribusi asli. Class O sangat dominan (~99.94%). Menggunakan `sample_weight` berbasis inverse frequency saat training untuk memberi bobot lebih ke kelas minoritas.

2. **Balance** (`data/processed/balance/`): Data yang sudah di-balance secara offline. Class O sudah di-undersample dan class PII sudah di-oversample sebelum split. Data ini lebih kecil namun distribusi label lebih seimbang.

## Threshold Tuning

Semua model menggunakan **probability threshold tuning** pada validation set masing-masing:
- Imbalance models → tune di `imbalance/val.json`
- Balance models → tune di `balance/val.json`
- Jika model memprediksi O tetapi probabilitas non-O terbaik ≥ threshold, override ke non-O
- Threshold di-tune dari `[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]`

## Output

### CSV Predictions
```
results/predictions/lightgbm_imbalance_predictions.csv
results/predictions/xgboost_imbalance_predictions.csv
results/predictions/lightgbm_balance_predictions.csv
results/predictions/xgboost_balance_predictions.csv
```

### Metrics JSON
```
results/metrics/lightgbm_imbalance_metrics.json
results/metrics/xgboost_imbalance_metrics.json
results/metrics/lightgbm_balance_metrics.json
results/metrics/xgboost_balance_metrics.json
```

## Hasil

| Model | Token F1 | Entity F1 | Token Precision | Token Recall |
|---|---|---|---|---|
| LightGBM Imbalance | - | - | - | - |
| XGBoost Imbalance | - | - | - | - |
| LightGBM Balance | - | - | - | - |
| XGBoost Balance | - | - | - | - |

> **Note:** Tabel hasil akan diisi setelah notebook dijalankan di Kaggle.

## Cara Menjalankan

1. Upload notebook `notebooks/05_boosting.ipynb` ke Kaggle
2. Pastikan dataset balance dan imbalance tersedia sebagai Kaggle dataset
3. Enable GPU accelerator di Kaggle settings
4. Jalankan semua cell secara berurutan
5. Hasil akan tersedia di `results/predictions/` dan `results/metrics/`

## Evaluasi

```bash
python3 -c "
from src.evaluate import evaluate_from_csv, print_metrics
metrics = evaluate_from_csv('results/predictions/xgboost_imbalance_predictions.csv', 'xgboost_imbalance')
print_metrics(metrics)
"
```
