# Boosting Models — PII Data Detection

## Metode

Token-level classification menggunakan **XGBoost** dan **LightGBM** untuk mendeteksi PII (Personally Identifiable Information) dalam teks.

Setiap token diklasifikasikan ke salah satu dari 12 BIO label:
`O, B-NAME_STUDENT, I-NAME_STUDENT, B-EMAIL, B-USERNAME, B-ID_NUM, I-ID_NUM, B-PHONE_NUM, I-PHONE_NUM, B-URL_PERSONAL, B-STREET_ADDRESS, I-STREET_ADDRESS`

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

Dua strategi yang dibandingkan:

1. **Imbalance**: Menggunakan data penuh (~3M tokens) dengan `sample_weight` berbasis inverse frequency. Class O mendominasi ~99.94%.

2. **Balance**: Undersample class O (target = 3× jumlah PII tokens) + oversample minority PII classes (minimum 500 samples per class). Menghasilkan ~270K tokens.

## Threshold Tuning

Semua model menggunakan **probability threshold tuning** pada validation set:
- Jika model memprediksi O tetapi probabilitas non-O terbaik ≥ threshold, override ke non-O
- Threshold di-tune dari `[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]`

## Hasil

| Model | Token F1 | Entity F1 | Token Precision | Token Recall |
|---|---|---|---|---|
| LightGBM Imbalance | 0.3369 | 0.2732 | 0.2317 | 0.6170 |
| XGBoost Imbalance | 0.3481 | 0.2795 | 0.2387 | 0.6429 |
| LightGBM Balance | 0.2478 | 0.1846 | 0.1501 | 0.7097 |
| XGBoost Balance | 0.2435 | 0.1824 | 0.1475 | 0.6976 |

**Insight**: Model imbalance menghasilkan F1 lebih tinggi. Model balance menghasilkan recall lebih tinggi tetapi precision lebih rendah.

