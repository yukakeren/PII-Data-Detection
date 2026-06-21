# Laporan Final Project: PII Shield
## Deteksi dan Redaksi Personally Identifiable Information pada Data Pendidikan

---

## 1. Pendahuluan

Proyek ini bertujuan mendeteksi dan meredaksi **Personally Identifiable Information (PII)** dari esai siswa menggunakan pendekatan token classification dengan skema BIO tagging. Dataset bersumber dari kompetisi **Kaggle — PII Detection Removal from Educational Data**.

Deteksi PII pada data teks pendidikan menghadirkan tantangan unik. Tidak seperti dokumen legal atau finansial yang terstruktur, esai siswa bersifat sangat tidak terstruktur (*unstructured*), sering mengandung variasi penulisan (*typos*), serta memiliki konteks kalimat yang membingungkan. Pendekatan *Rule-Based* atau *Reguler Expression* (Regex) tradisional seringkali gagal karena sistem ini tidak mampu memahami konteks (misalnya: gagal membedakan nama presiden yang dibahas di dalam esai dengan nama murid penulis esai). Oleh karena itu, pendekatan Machine Learning dan Deep Learning kontekstual diuji dalam penelitian ini.

Entitas PII yang dideteksi meliputi 7 kategori: `NAME_STUDENT`, `EMAIL`, `USERNAME`, `ID_NUM`, `PHONE_NUM`, `URL_PERSONAL`, dan `STREET_ADDRESS`.

---

## 2. Analisis Data

### 2.1 Karakteristik Dataset

| Karakteristik | Nilai |
|:---|:---|
| Total Dokumen | 6.807 esai |
| Rata-rata Panjang Dokumen | ~733 token |
| Panjang Maksimum | 3.298 token |
| Panjang Minimum | 69 token |

![Distribusi Panjang Dokumen](/home/ata/school/PII-Data-Detection/assets/doc_length_dist.png)
*Sebagian besar esai berpusat pada panjang 500–1000 token, dengan rata-rata 733 token.*

### 2.2 Distribusi Label

Dataset memiliki **ketidakseimbangan label yang ekstrem** — tipikal untuk task PII detection.

| Label | Jumlah | Persentase |
|:---|---:|---:|
| `O` (Non-PII) | 4.989.794 | 99,94% |
| `B-NAME_STUDENT` | 1.365 | 0,027% |
| `I-NAME_STUDENT` | 1.096 | 0,022% |
| `B-URL_PERSONAL` | 110 | 0,002% |
| `B-ID_NUM` | 78 | 0,002% |
| `B-EMAIL` | 39 | <0,001% |
| `I-STREET_ADDRESS` | 20 | <0,001% |
| `I-PHONE_NUM` | 15 | <0,001% |
| `B-USERNAME` | 6 | <0,001% |
| `B-PHONE_NUM` | 6 | <0,001% |
| `B-STREET_ADDRESS` | 2 | <0,001% |

![Distribusi Label PII Skala Logaritmik](/home/ata/school/PII-Data-Detection/assets/label_dist_log.png)
*Visualisasi persebaran entitas PII pada training set (skala logaritmik).*

**Implikasi penting:**
1. **Sparsity**: Entitas PII menyusun kurang dari 0,1% total token.
2. **Akuntabilitas Metrik Evaluasi**: Metrik Akurasi tradisional sangat *misleading* (klasifikasi semua token sebagai 'O' akan menghasilkan akurasi 99,9%). Oleh karena itu, evaluasi utama kami menggunakan **Micro F1-Score** pada kelas PII (mengabaikan 'O'). Pembobotan *Micro* dipilih ketimbang *Macro* agar beberapa kelas yang tidak ada representasinya pada *test set* (akibat kelangkaan ekstrem) tidak menghancurkan intepretasi performa model secara keseluruhan.
3. **Rare classes**: `PHONE_NUM`, `STREET_ADDRESS`, dan `USERNAME` memiliki sampel sangat sedikit (<20), sehingga sangat sulit dideteksi tanpa bantuan data augmentasi.

---

## 3. Strategi Pembagian Data (Data Split)

### 3.1 Split Utama (Imbalance)

Dataset dibagi berdasarkan **document ID** secara *random* untuk mencegah kebocoran data (*data leakage*) di mana paragraf dari dokumen yang sama muncul di *Train* dan *Test*.

| Split | Dokumen | Rasio |
|:---|---:|---:|
| Train | 4.764 | 70% |
| Validation | 1.021 | 15% |
| Test Internal | 1.022 | 15% |

**Catatan Kritis Metode Splitting:** 
Pembagian data ini sayangnya dilakukan secara acak level dokumen tanpa algoritma stratifikasi (*Stratified Split*). Mengingat kelas seperti `STREET_ADDRESS` hanya muncul di 2 dokumen di seluruh data, metode *random split* berisiko tinggi membuang kedua sampel tersebut ke dalam *Test Set* (membuat model tak pernah mempelajarinya) atau membuangnya ke *Train Set* (membuatnya tak bisa dievaluasi). Hal inilah yang berkontribusi kuat terhadap F1-score 0,000 pada kelas-kelas langka.

### 3.2 Skenario Balance vs Imbalance

Eksperimen dilakukan pada dua skenario data:

| Skenario | Deskripsi Metodologi | Rasio PII Train |
|:---|:---|---:|
| **Imbalance** | **Distribusi asli**, dibiarkan murni apa adanya seperti sumber data kompetisi Kaggle. | 0,055% |
| **Balance** | **Data Augmentation (Sintesis LLM)**: Dataset tidak sekadar di-oversample, tetapi *diciptakan* dengan mengkombinasikan data asli Kaggle, *Public Information* (seperti dataset nama/alamat publik), dan generasi kalimat menggunakan **LLM (Large Language Model)**. Tujuannya adalah mensintesis konteks kalimat baru yang memuat kelas PII langka untuk memperkaya representasi. | 2,207% |

![Perbandingan Dataset Balance vs Imbalance](/home/ata/school/PII-Data-Detection/assets/dataset_balance_vs_imbalance.png)
*Rasio jumlah kelas PII yang dioversampling sangat drastis mengubah representasi data asli.*

Rasio PII pada data balance **~40x lebih tinggi** dibanding data imbalance. Tujuannya adalah menguji apakah penyeimbangan data membantu model belajar pola PII yang langka.

### 3.3 Pencegahan Overlap

Script `split_imbalance_no_overlap.py` memastikan tidak ada document ID yang muncul di lebih dari satu split, mencegah data leakage antar skenario balance dan imbalance.

---

## 4. Model yang Diuji

### 4.1 Ringkasan Arsitektur

| Kategori | Model | Tipe |
|:---|:---|:---|
| Baseline ML | Logistic Regression | Token-level classifier |
| Baseline ML | Linear SVM | Token-level classifier |
| Sequence Model | CRF | Sequence labeler |
| Boosting | XGBoost | Token-level classifier |
| Boosting | LightGBM | Token-level classifier |
| Deep Learning | DistilBERT | Transformer fine-tuned |
| Deep Learning | DeBERTa (v3-small) | Transformer fine-tuned |
| Hybrid | Regex + Best ML | Rule-based overlay |

---

## 5. Hyperparameter Tuning & Optimasi Algoritma

Eksplorasi parameter (*Hyperparameter Tuning*) dicari melalui pengujian empiris pada *Validation Set*. Untuk Baseline ML dan Boosting, kami menggunakan teknik pencarian komprehensif pada subset fitur, sedangkan untuk algoritma *Transformer* (Deep Learning), pencarian dilakukan secara *Heuristik / Manual Search* demi menghemat limitasi waktu dan *resource* komputasi (GPU).

### 5.1 Baseline ML

| Parameter | Logistic Regression | Linear SVM |
|:---|:---|:---|
| `class_weight` | `balanced` | `balanced` |
| `max_iter` | 150 | 1000 |
| `solver` | `saga` | — |
| `tol` | 1e-3 | 1e-3 |
| `random_state` | 42 | 42 |

**Feature engineering**: Token shape, capitalization flags, prefix/suffix, context window (prev/next), TF-IDF character n-gram (2-4).

### 5.2 CRF

| Parameter | Nilai |
|:---|:---|
| `algorithm` | `lbfgs` |
| `c1` (L1 regularization) | 0,01 |
| `c2` (L2 regularization) | 0,01 |
| `max_iterations` | 300 |
| `all_possible_transitions` | True |

Beberapa eksperimen tuning (Exp A-D) dilakukan, namun konfigurasi baseline menghasilkan performa terbaik pada validation set.

### 5.3 Boosting (LightGBM & XGBoost)

| Parameter | LightGBM | XGBoost |
|:---|:---|:---|
| `n_estimators` | 600 | 500 |
| `max_depth` | 8 | 7 |
| `learning_rate` | 0,06 | 0,08 |
| `subsample` | 0,8 | 0,8 |
| `colsample_bytree` | 0,8 | 0,8 |
| `min_child` | 20 (imb) / 50 (bal) | 5 (imb) / 3 (bal) |
| `reg_alpha` | 0,5 | 0,5 |
| `reg_lambda` | 0,5 | 1,0 |

**Threshold tuning** dilakukan pada validation set dengan nilai `[0.05, 0.10, ..., 0.50]`. Threshold terbaik dipilih berdasarkan Val F1 tertinggi.

**Feature engineering**: 106 hand-crafted features + 32.768 character n-gram features (HashingVectorizer).

### 5.4 Transformer (DistilBERT & DeBERTa)

| Parameter | DistilBERT | DeBERTa |
|:---|:---|:---|
| Base model | `distilbert-base-uncased` | `microsoft/deberta-v3-small` |
| Epochs | 3 | 3-5 |
| Batch size | 8 | 2 |
| Gradient accumulation | 2 | 8 |
| Learning rate | 5e-5 | 1e-5 |
| Weight decay | 0,01 | 0,01 |
| Warmup ratio | 0,1 | 0,1 |
| Max length | 256 | 256 |
| Chunk size | 128 | 128 |
| Loss mode | `sqrt_balanced` | `sqrt_balanced` / `default` |
| Optimizer | AdamW | AdamW |
| Scheduler | Linear | Linear |

**Loss reweighting** (`sqrt_balanced`): Bobot kelas dihitung dari rasio `count(O)/count(label)`, di-square-root, dan di-cap pada 10x, untuk menangani ketidakseimbangan label tanpa destabilisasi training.

---

## 6. Penanganan Data Imbalance

Setiap model menerapkan strategi berbeda untuk mengatasi ketidakseimbangan label:

| Model | Strategi Imbalance Handling |
|:---|:---|
| Logistic Regression | `class_weight="balanced"` |
| Linear SVM | `class_weight="balanced"` |
| CRF | Tidak ada strategi eksplisit (CRF belajar transition) |
| LightGBM | `sample_weight` (inverse freq, power 0.5/0.3) + threshold tuning |
| XGBoost | `sample_weight` (inverse freq, power 0.5/0.3) + threshold tuning |
| DistilBERT | `sqrt_balanced` loss reweighting |
| DeBERTa | `sqrt_balanced` / `default` loss reweighting |

### Hasil: Balance vs Imbalance

| Model | F1 (Balance) | F1 (Imbalance) | Keterangan |
|:---|---:|---:|:---|
| Logistic Regression | 0,0000 | 0,3265 | Imbalance lebih baik |
| Linear SVM | 0,3464 | **0,7077** | Imbalance jauh lebih baik |
| CRF | 0,0000 | 0,7328 | Imbalance jauh lebih baik |
| LightGBM | 0,3575 | **0,7779** | Imbalance 2x lebih baik |
| XGBoost | 0,3682 | 0,5896 | Imbalance lebih baik |
| DistilBERT | 0,5819 | **0,8074** | Imbalance lebih baik |
| DeBERTa | 0,6190 | 0,7017 | Imbalance lebih baik |

**Kesimpulan dan Analisis Mendalam**: 
Pada semua model, **skenario imbalance** yang dipadukan dengan teknik *cost-sensitive learning* (seperti `sample_weight` atau loss reweighting) **secara konsisten menghasilkan F1-score yang jauh lebih tinggi** dibandingkan data yang di-balance secara offline. 

Terdapat tiga alasan teknis mengapa penggunaan dataset *balance* (sintesis LLM) justru merusak performa pada data uji internal:
1. **Distribution Shift (Pergeseran Distribusi Teks)**: Data *Balance* banyak dicampur dengan teks hasil *generate* LLM dan dokumen publik. LLM memiliki kecenderungan menulis kalimat yang sangat baku, struktural, dan berpola kaku. Sebaliknya, *Test Set* kita murni berisi esai siswa yang penuh dengan tata bahasa acak, *typo*, dan *slang*. Model akhirnya "overfit" pada gaya bahasa LLM dan kebingungan saat membaca tulisan manusia sungguhan. **Bukti Kuantitatif**: Kegagalan generalisasi ini terbukti dari meroketnya *False Negative* (gagal deteksi PII) pada model DistilBERT Balance, yang melonjak 4x lipat dari **71 kasus (imbalance)** menjadi **285 kasus (balance)**. Model "kaget" dan melewatkan tulisan siswa asli karena tidak seterbaca teks buatan LLM.
2. **Konteks Semantik yang Dipaksakan**: Saat menyisipkan PII publik ke dalam paragraf sintesis, strukturnya seringkali tidak natural secara linguistik. Model mempelajari pola *surrounding words* (kata sekitar) yang salah dan menjadi sangat *parno* (agresif) di dunia nyata, memicu ledakan jumlah *False Positive*. **Bukti Kuantitatif**: Kerusakan pemahaman konteks ini terbukti dari ledakan *False Positive*. Pada Linear SVM, kesalahan tebak meroket **676%** dari 140 FP (imbalance) menjadi **1.087 FP** (balance). Pada LightGBM, kesalahan tebak naik **468%** dari 170 FP menjadi **966 FP**.
3. **Keunggulan Loss Reweighting**: Mempertahankan data *imbalance* murni dari Kaggle dipadukan dengan modifikasi penalti (*loss reweighting* seperti `sqrt_balanced`) memungkinkan model untuk belajar *100% dari distribusi asli esai murid*, tanpa ada *noise* dari data sintetik, namun tetap memberikan bobot gradien ekstra saat memprediksi PII. **Bukti Kuantitatif**: Pendekatan data murni + penalti algoritmik ini terbukti superior secara matematis, dengan F1-Score DistilBERT bertahan kokoh di **0,8074**, sedangkan injeksi data sintesis justru mencekik performanya turun 28% ke angka **0,5819**.

---

## 7. Hasil Eksperimen Utama

### 7.1 Ranking Keseluruhan (Token-Level F1)

| Rank | Model | Precision | Recall | F1-Score |
|---:|:---|---:|---:|---:|
| 1 | **DistilBERT (imbalance)** | 0,7374 | 0,8921 | **0,8074** |
| 2 | Hybrid (Regex + ML) | 0,7197 | 0,8936 | 0,7973 |
| 3 | LightGBM (imbalance) | 0,7561 | 0,8009 | 0,7779 |
| 4 | CRF (imbalance) | 0,8574 | 0,6398 | 0,7328 |
| 5 | Linear SVM (imbalance) | 0,7574 | 0,6641 | 0,7077 |
| 6 | DeBERTa (imbalance) | 0,7027 | 0,7006 | 0,7017 |
| 7 | DeBERTa (balance) | 0,5014 | 0,8085 | 0,6190 |
| 8 | XGBoost (imbalance) | 0,4385 | 0,8997 | 0,5896 |
| 9 | DistilBERT (balance) | 0,5978 | 0,5669 | 0,5819 |
| 10 | Logistic Regression (imbalance) | 0,4227 | 0.2660 | 0,3265 |

### 7.2 Entity-Level F1

| Model | Precision | Recall | F1-Score |
|:---|---:|---:|---:|
| **DistilBERT (imbalance)** | 0,6378 | 0,8364 | **0,7237** |
| Hybrid (Regex + ML) | 0,6146 | 0,8417 | 0,7105 |
| CRF (imbalance) | 0,8431 | 0,6095 | 0,7075 |
| LightGBM (imbalance) | 0,6031 | 0,7177 | 0,6554 |
| DeBERTa (imbalance) | 0,5385 | 0,5541 | 0,5462 |
| Linear SVM (imbalance) | 0,4460 | 0,5119 | 0,4767 |
| Logistic Regression (imbalance) | 0,1153 | 0,1135 | 0,1144 |

### 7.3 Per-Class F1 (Top 5 Model)

| Label | DistilBERT (imb) | Hybrid | LightGBM (imb) | CRF (imb) | DeBERTa (imb) |
|:---|---:|---:|---:|---:|---:|
| B-EMAIL | 0,8750 | **0,9412** | 1,0000 | 1,0000 | 0,0000 |
| B-ID_NUM | 0,7667 | 0,6667 | **0,8400** | 0,8095 | 0,0000 |
| B-NAME_STUDENT | 0,7824 | **0,7824** | 0,7630 | 0,7029 | 0,7009 |
| B-URL_PERSONAL | 0,6024 | 0,5263 | **0,8519** | 0,7619 | 0,0000 |
| I-NAME_STUDENT | **0,8712** | 0,8727 | 0,7878 | 0,7515 | 0,7776 |
| B-PHONE_NUM | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| B-STREET_ADDRESS | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| B-USERNAME | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |

**Catatan**: `PHONE_NUM`, `STREET_ADDRESS`, dan `USERNAME` memiliki F1 = 0 pada semua model karena jumlah sampel di test set terlalu sedikit (<6 sampel).

### 7.4 Analisis False Positive dan False Negative (Token-Level)

Untuk melihat seberapa agresif atau konservatif sebuah model, kita dapat melihat distribusi salah tebakannya. Model yang *balance* umumnya menghasilkan **False Positive (FP) yang jauh lebih tinggi**.

| Rank | Model | True Positive (TP) | False Positive (FP) | False Negative (FN) | F1-Score |
|---:|:---|---:|---:|---:|---:|
| 1 | DistilBERT (imbalance) | 587 | 209 | 71 | 0,8074 |
| 2 | Hybrid (Regex + ML) | 588 | 229 | 70 | 0,7973 |
| 3 | LightGBM (imbalance) | 527 | 170 | 131 | 0,7779 |
| 4 | CRF (imbalance) | 421 | **70** (Paling Konservatif) | 237 | 0,7328 |
| 5 | Linear SVM (imbalance) | 437 | 140 | 221 | 0,7077 |
| 6 | DeBERTa (imbalance) | 461 | 195 | 197 | 0,7017 |
| 7 | DeBERTa (balance) | 532 | 529 | 126 | 0,6190 |
| 8 | XGBoost (imbalance) | **592** (Recall Tertinggi) | 758 | **66** | 0,5896 |
| 9 | DistilBERT (balance) | 373 | 251 | 285 | 0,5819 |
| 10 | Logistic Regression (imbalance)| 175 | 239 | 483 | 0,3265 |

**Perbandingan Ekstrem Data Balance**:
Sebagai perbandingan, model yang dilatih pada **data balance murni** tanpa pembobotan loss justru "meledak" angka False Positive-nya:
- **LightGBM (balance)**: 328 TP, **966 FP**, 213 FN
- **Linear SVM (balance)**: 341 TP, **1.087 FP**, 200 FN
- **XGBoost (balance)**: 326 TP, **904 FP**, 215 FN

![Distribusi False Positive per Label](/home/ata/school/PII-Data-Detection/assets/fp_per_label.png)
*Tingkat False Positive (kesalahan deteksi) pada model DistilBERT Balance melonjak drastis, terutama pada entitas langka dan STREET_ADDRESS.*

![Distribusi False Negative per Label](/home/ata/school/PII-Data-Detection/assets/fn_per_label.png)
*Tingkat False Negative (gagal deteksi) justru tidak membaik secara signifikan pada model Balance, bahkan memburuk pada kasus I-NAME_STUDENT.*

Ini memperkuat bukti bahwa kelas minoritas yang di-oversampling secara brutal (offline balance) membuat model "menebak PII di mana-mana" sehingga *precision* jatuh bebas ke angka 0,2—0,3.

### 7.5 Analisis Error secara Kualitatif

Selain analisis kuantitatif di atas, tinjauan kualitatif terhadap salah klasifikasi (Error Analysis) pada DistilBERT dan model ML mengungkapkan dua tantangan utama dalam domain esai teks:

1. **False Positive Kontekstual (Model Terlalu Parno)**:
   - *Teks Asli Contoh*: "...when President George Washington crossed the river..."
   - *Prediksi Sistem*: Kata `George Washington` diprediksi sebagai `NAME_STUDENT`.
   - *Analisis*: Esai pelajaran sejarah sering memuat nama tokoh publik. Model keliru mengenali mereka sebagai *siswa penulis esai* karena ketiadaan batasan konteks dunia nyata (*world knowledge*).
2. **False Negative akibat Obfuscation / Typo (Model Gagal Deteksi)**:
   - *Teks Asli Contoh*: "my personal email is john.doe at gmail.com"
   - *Prediksi Sistem*: Teks `john.doe at gmail.com` terlewat dan diprediksi sebagai `O`.
   - *Analisis*: Siswa dapat menyamarkan PII mereka (mengganti `@` dengan `at`). Pendekatan Regex statis langsung gugur di kasus seperti ini, sementara Transformer berpotensi menebak jika pernah dilatih dengan ragam variasi penulisan bahasa *slang*.

---

## 8. Skenario Uji

### 8.1 Dengan Regex vs Tanpa Regex (Hybrid)

Model hybrid mengkombinasikan **prediksi ML terbaik** (DistilBERT imbalance) dengan **regex rules** untuk deteksi pola EMAIL, URL, PHONE, dan ID_NUM. Jika regex match, prediksi regex digunakan; jika tidak, fallback ke prediksi ML.

| Metrik | DistilBERT (Tanpa Regex) | Hybrid (Dengan Regex) | Selisih |
|:---|---:|---:|:---|
| Token Precision | **0,7374** | 0,7197 | -0,0177 |
| Token Recall | 0,8921 | **0,8936** | +0,0015 |
| Token F1 | **0,8074** | 0,7973 | -0,0101 |
| Entity Precision | **0,6378** | 0,6146 | -0,0232 |
| Entity Recall | 0,8364 | **0,8417** | +0,0053 |
| Entity F1 | **0,7237** | 0,7105 | -0,0132 |

**Analisis**: Penambahan regex **sedikit meningkatkan recall** (+0,15%) namun **menurunkan precision** (-1,8%) dan **F1 keseluruhan** (-1,0%). Hal ini terjadi karena:

1. **Regex terlalu agresif**: Pattern seperti `\d{5,}` untuk `ID_NUM` menghasilkan banyak false positive (23 FP pada hybrid vs 12 FP pada DistilBERT murni).
2. **URL regex over-trigger**: Pattern URL mendeteksi 45 FP (vs 33 FP pada DistilBERT).
3. **Konteks diabaikan**: Regex tidak mempertimbangkan konteks kalimat, sehingga token yang secara struktural mirip PII tetapi bukan PII tetap ditandai.

**Kesimpulan**: Untuk dataset ini, **model ML murni (DistilBERT) lebih unggul** dibanding hybrid. Regex rules lebih cocok sebagai safety net pada deployment, bukan sebagai override.

### 8.2 Best Model Single vs Best Model Ensemble (Hybrid)

| Aspek | Single (DistilBERT) | Ensemble/Hybrid |
|:---|---:|---:|
| Token F1 | **0,8074** | 0,7973 |
| Entity F1 | **0,7237** | 0,7105 |
| Total TP | 587 | 588 |
| Total FP | **209** | 229 |
| Total FN | 71 | **70** |

Model single (DistilBERT imbalance) menghasilkan F1 lebih tinggi karena memiliki **false positive lebih rendah**. Ensemble/hybrid menambah 1 TP tetapi juga menambah 20 FP, sehingga net effect-nya negatif.

### 8.3 Best ML vs Best DL

| Metrik | Best ML: LightGBM (imb) | Best DL: DistilBERT (imb) | Keterangan |
|:---|---:|---:|:---|
| Token Precision | **0,7561** | 0,7374 | ML +0,019 |
| Token Recall | 0,8009 | **0,8921** | DL +0,091 |
| Token F1 | 0,7779 | **0,8074** | DL +0,030 |
| Entity Precision | 0,6031 | **0,6378** | DL +0,035 |
| Entity Recall | 0,7177 | **0,8364** | DL +0,119 |
| Entity F1 | 0,6554 | **0,7237** | DL +0,068 |

**Analisis per kelas (F1):**

| Label | LightGBM | DistilBERT | Unggul |
|:---|---:|---:|:---|
| B-EMAIL | **1,0000** | 0,8750 | ML |
| B-ID_NUM | **0,8400** | 0,7667 | ML |
| B-NAME_STUDENT | 0,7630 | **0,7824** | DL |
| B-URL_PERSONAL | **0,8519** | 0,6024 | ML |
| I-NAME_STUDENT | 0,7878 | **0,8712** | DL |

**Kesimpulan**:
- **DistilBERT (DL)** unggul secara keseluruhan berkat kemampuan kontekstual yang lebih kuat, terutama pada entitas NAME yang membutuhkan pemahaman semantik.
- **LightGBM (ML)** unggul pada entitas dengan pola struktural jelas (EMAIL, URL, ID_NUM) berkat feature engineering dan regex-aware features.
- DL menunjukkan **recall jauh lebih tinggi** (+9,1%), menangkap lebih banyak entitas PII yang terlewat oleh ML.

---

## 9. Analisis Mendalam

### 9.1 Mengapa DistilBERT > DeBERTa?

Secara teori, DeBERTa v3 memiliki arsitektur yang lebih canggih (disentangled attention). Namun pada eksperimen ini, DistilBERT mengungguli DeBERTa:

| Model | Token F1 | Entity F1 |
|:---|---:|---:|
| DistilBERT (imbalance) | **0,8074** | **0,7237** |
| DeBERTa (imbalance) | 0,7017 | 0,5462 |

Kemungkinan penyebab:
1. **DeBERTa gagal mendeteksi EMAIL, URL, ID_NUM** (F1 = 0 pada ketiga kelas), menunjukkan model tidak cukup belajar pola entitas struktural.
2. **Keterbatasan compute**: DeBERTa memerlukan GPU besar dan lebih banyak epoch. Training dengan batch size kecil (2) dan gradient accumulation mungkin tidak optimal.
3. **DistilBERT lebih ringan** dan konvergen lebih cepat pada dataset berukuran sedang ini.

### 9.2 Kekuatan CRF

CRF mencapai **precision tertinggi** (0,8574) di antara semua model, karena CRF secara eksplisit memodelkan **transisi antar label** (B->I, I->I, dll.), sehingga menghasilkan prediksi yang lebih valid secara sekuensial. Trade-off-nya adalah recall yang lebih rendah (0,6398).

### 9.3 Kelas yang Tidak Terdeteksi

Tidak ada model yang berhasil mendeteksi `PHONE_NUM`, `STREET_ADDRESS`, dan `USERNAME` pada test set. Ini disebabkan:
- `PHONE_NUM`: Hanya 6 sampel di seluruh dataset.
- `STREET_ADDRESS`: Hanya 2 sampel di seluruh dataset.
- `USERNAME`: Hanya 6 sampel di seluruh dataset.

Untuk mengatasi ini, diperlukan **data augmentation** atau **pre-trained knowledge** dari model yang sudah dilatih pada dataset NER yang lebih besar.

---

## 10. Analisis Trade-off Komputasi (Performance vs Efficiency)

Untuk mendeploy model ke lingkungan produksi (terutama pada aplikasi web Streamlit tanpa GPU besar), kita harus mempertimbangkan trade-off antara **Akurasi (F1)** vs **Kebutuhan Resource**.

| Kategori Model | Model Terbaik (Imbalance) | Token F1 | Estimasi Ukuran File | Kebutuhan Hardware | Kecepatan Inference | Rekomendasi Penggunaan |
|:---|:---|---:|---:|:---|:---|:---|
| Baseline ML | Linear SVM | 0,7077 | ~18 MB | CPU Standar | Sangat Cepat | Eksekusi di server RAM rendah |
| Sequence Model | CRF | 0,7328 | < 2 MB | CPU Standar | Cepat | Jika *precision* sangat diutamakan |
| Boosting | LightGBM | 0,7779 | < 5 MB | CPU Standar | Sangat Cepat | **Best Value** (Cepat & F1 Tinggi) |
| Deep Learning | DistilBERT | **0,8074** | ~260 MB | RAM Besar / GPU | Sedang | **Akurasi maksimal** (High Budget) |

**Insight Deployment:**
Meskipun **DistilBERT** adalah juara bertahan dalam hal F1-Score, model ini membutuhkan memori >250 MB dan komputasi transformer yang memakan waktu pada CPU biasa. Sebaliknya, **LightGBM** menawarkan F1-Score `0,7779` (hanya terpaut ~3% dari DistilBERT) dengan ukuran file yang **50x lebih kecil** dan eksekusi secepat kilat. LightGBM sangat ideal untuk deployment *low-budget* atau *real-time text checking*.

---

## 11. Ringkasan Hasil

### Best Model per Kategori

| Kategori | Best Model | Token F1 | Entity F1 |
|:---|:---|---:|---:|
| **Overall Best** | DistilBERT (imbalance) | **0,8074** | **0,7237** |
| Best ML | LightGBM (imbalance) | 0,7779 | 0,6554 |
| Best DL | DistilBERT (imbalance) | 0,8074 | 0,7237 |
| Best Precision | CRF (imbalance) | P=0,8574 | P=0,8431 |
| Best Recall | XGBoost (imbalance) | R=0,8997 | R=0,8549 |
| Best Hybrid | Hybrid (Regex + DistilBERT) | 0,7973 | 0,7105 |
| Best Baseline | Linear SVM (imbalance) | 0,7077 | 0,4767 |

---

## 12. Kesimpulan

1. **DistilBERT pada data imbalance** adalah model terbaik dengan Token F1 = 0,8074 dan Entity F1 = 0,7237. Keunggulannya terletak pada kemampuan kontekstual bidirectional yang menangkap nuansa semantik entitas PII.

2. **Data imbalance + loss reweighting lebih efektif** dibanding data yang di-balance secara offline. Semua model menunjukkan performa lebih baik pada skenario imbalance.

3. **Hybrid (Regex + ML) tidak selalu lebih baik**. Pada eksperimen ini, regex rules menurunkan precision tanpa gain recall yang signifikan. Regex lebih cocok sebagai lapisan keamanan tambahan, bukan pengganti prediksi model.

4. **Model ML tradisional tetap kompetitif**: LightGBM mencapai Token F1 = 0,7779, hanya selisih 0,03 dari DistilBERT, dengan waktu training yang jauh lebih cepat.

5. **Keterbatasan utama** adalah pada kelas dengan sampel sangat sedikit (`PHONE_NUM`, `STREET_ADDRESS`, `USERNAME`). Diperlukan augmentasi data atau transfer learning untuk meningkatkan deteksi kelas-kelas tersebut.

---

## 13. Rekomendasi

- **Deployment**: Gunakan DistilBERT (imbalance) sebagai model utama pada aplikasi Streamlit, dengan regex sebagai safety net opsional.
- **Peningkatan**: Tambahkan data training untuk kelas langka melalui augmentasi atau synthetic data generation.
- **Monitoring**: Implementasikan feedback loop pada aplikasi untuk mengumpulkan koreksi user dan meningkatkan model secara iteratif.

---

*Laporan ini disusun berdasarkan eksperimen yang dilakukan pada dataset Kaggle PII Detection Removal from Educational Data, dengan evaluasi pada test set internal (1.022 dokumen).*
