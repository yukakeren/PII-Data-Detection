# PII Shield

Deteksi dan redaksi Personally Identifiable Information (PII) pada teks pendidikan (khususnya esai siswa) menggunakan pendekatan token classification / NER.

## Tujuan Project

- Mendeteksi entitas PII pada level token (BIO tagging)
- Menghasilkan redacted text untuk demo privasi data
- Membandingkan beberapa keluarga model (baseline ML, sequence model, boosting, transformer, hybrid)
- Menilai performa dengan metrik yang relevan untuk data imbalanced: precision, recall, F1

## Dataset

Sumber utama: **Kaggle – PII Detection Removal from Educational Data**.

Data berisi dokumen teks dengan:
- `document` (id dokumen)
- `full_text`
- `tokens`
- `labels` (BIO)
- `trailing_whitespace`

### Kenapa tetap buat split internal?

Walaupun kompetisi menyediakan train/test, test publik umumnya tanpa ground truth. Karena itu evaluasi internal memakai split dari train berlabel:
- Train: 70%
- Validation: 15%
- Test internal: 15%

Split dilakukan **berdasarkan document id** (bukan token random) untuk mencegah leakage antarsplit.

## Label dan Task

Task utama: **Token Classification / NER**.

Entity PII:
- `NAME_STUDENT`
- `EMAIL`
- `USERNAME`
- `ID_NUM`
- `PHONE_NUM`
- `URL_PERSONAL`
- `STREET_ADDRESS`

Format label: BIO (`B-*`, `I-*`) + `O`.

## Struktur Repository

```text
.
├── app/
│   └── streamlit_app.py
├── configs/
│   └── label_schema.json
├── data/
│   ├── raw/
│   └── processed/
│       ├── train.json
│       ├── val.json
│       └── test_internal.json
├── models/
│   ├── baseline/
│   ├── boosting/
│   ├── crf/
│   └── transformer/
├── notebooks/
├── results/
│   ├── metrics/
│   ├── metrics_template.json
│   └── predictions/
├── src/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── integrate_predictions.py
│   ├── preprocessing.py
│   ├── split_data.py
│   ├── utils.py
│   └── verify_split.py
├── README.md
└── readme-chore.md
```

## Penjelasan Komponen Inti

- `src/data_loader.py`: load/save data mentah dan processed
- `src/split_data.py`: generate split internal 70/15/15 by document id
- `src/preprocessing.py`: preprocessing ringan dan util opsional per model
- `src/evaluate.py`: evaluasi token-level dan entity-level
- `src/integrate_predictions.py`: agregasi hasil prediksi antar model + ringkasan komparasi
- `app/streamlit_app.py`: prototype antarmuka demo deteksi dan redaksi PII

## Alasan Desain Preprocessing

Preprocessing untuk PII detection harus konservatif.

### Kenapa tidak agresif secara default?

Operasi seperti ini berisiko menghilangkan sinyal PII:
- lowercasing global
- punctuation removal
- stopword removal

Contoh alasan:
- Email butuh karakter `@` dan `.`
- URL butuh `/`, `:`, `.`
- Nama orang sering terbantu kapitalisasi
- Alamat/ID sering bergantung angka dan simbol

Karena itu di repo ini:
- default preprocessing = ringan (normalisasi whitespace + entity dasar)
- preprocessing agresif = **opsional per model**, bukan aturan global

## Alasan Metrik Evaluasi

Label `O` sangat dominan, jadi akurasi tidak representatif sebagai metrik utama.

Metrik utama:
- Precision
- Recall
- F1-score

Tambahan:
- Token-level metrics
- Entity-level metrics
- Per-class F1

## Format Output Wajib Antar Anggota

Setiap model wajib mengeluarkan CSV prediksi format:

```csv
document_id,token,true_label,pred_label
```

Agar semua model bisa dievaluasi dengan script yang sama (`src/evaluate.py`).

## Quick Start

1. Install dependency

```bash
pip install -r requirements.txt
```

2. Verifikasi split

```bash
python3 src/verify_split.py
```

3. Jalankan evaluasi (contoh)

```bash
python3 src/evaluate.py
```

4. Jalankan prototype Streamlit

```bash
streamlit run app/streamlit_app.py
```

## Progress dan Chore

Detail progres tugas, timeline, backlog, dan Definition of Done ada di:
- [readme-chore.md](readme-chore.md)

## Catatan Tim

- Semua anggota wajib pakai data dari `data/processed/`
- Jangan membuat split sendiri-sendiri
- Gunakan schema label dari `configs/label_schema.json`
- Simpan prediksi di `results/predictions/` dan metrik di `results/metrics/`
