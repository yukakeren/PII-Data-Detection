# Chore Board — PII Shield

Dokumen ini berisi progres kerja tim: apa yang sudah selesai, timeline, backlog, dan Definition of Done.

## Status Tugas Saat Ini

- [ ] Implementasi model baseline (Logistic Regression, Linear SVM)
- [ ] Implementasi CRF
- [ ] Implementasi boosting (XGBoost / LightGBM)
- [ ] Implementasi transformer (DistilBERT / DeBERTa)
- [ ] Integrasi hasil eksperimen antar anggota
- [ ] Finalisasi format output prediksi tiap anggota (dipastikan konsisten)
- [ ] Buat tabel komparasi final model
- [ ] Lengkapi Streamlit dengan model inference aktual
- [ ] Tulis laporan akhir dan analisis error per kelas

## Timeline (Rencana 2 Minggu)

## Minggu 1

### Hari 1–2: Setup & Data
- Setup repo, folder, schema label
- Validasi data dan split internal
- Sinkronisasi aturan format untuk seluruh anggota

### Hari 3–4: Baseline & Sequence
- Baseline ML (LogReg, SVM)
- CRF awal
- Evaluasi awal dengan script yang sama

### Hari 5–7: Boosting & Transformer Start
- XGBoost/LightGBM eksperimen awal
- DistilBERT/DeBERTa setup + fine-tuning awal

## Minggu 2

### Hari 8–10: Eksperimen Lanjutan
- Tuning model per anggota
- Konsolidasi output prediksi + metrics

### Hari 11–12: Integrasi & Komparasi
- Jalankan evaluasi terpadu lintas model
- Buat tabel komparasi final
- Pilih model terbaik untuk demo

### Hari 13–14: Finalisasi
- Integrasi ke Streamlit
- Screenshot & demo flow
- Final report + kesimpulan

## Backlog 

- [ ] Pakai data hanya dari `data/processed/`
- [ ] Jangan split data sendiri
- [ ] Output CSV wajib format:
  `document_id,token,true_label,pred_label`
- [ ] Simpan metrics JSON sesuai template
- [ ] Catat kendala eksperimen (error analysis singkat)

## Format Nama File yang Disepakati

### Prediksi
- `results/predictions/logistic_regression_predictions.csv`
- `results/predictions/linear_svm_predictions.csv`
- `results/predictions/crf_predictions.csv`
- `results/predictions/xgboost_predictions.csv`
- `results/predictions/distilbert_predictions.csv`

### Metrics
- `results/metrics/logistic_regression_metrics.json`
- `results/metrics/linear_svm_metrics.json`
- `results/metrics/crf_metrics.json`
- `results/metrics/xgboost_metrics.json`
- `results/metrics/distilbert_metrics.json`

## Risiko Utama dan Mitigasi

- Risiko: data leakage antar split
  - Mitigasi: split by document id, bukan by token
- Risiko: metrik bias karena label `O` dominan
  - Mitigasi: fokus precision/recall/F1, bukan accuracy
- Risiko: preprocessing merusak sinyal PII
  - Mitigasi: preprocessing konservatif default; agresif hanya opsional per model
- Risiko: format output antar anggota beda
  - Mitigasi: gunakan template CSV/JSON tunggal sejak awal


## Definition of Done

## DoD Model

Setiap model dianggap selesai jika:
- [ ] Code model bisa dijalankan end-to-end
- [ ] Menghasilkan file prediksi CSV sesuai format wajib
- [ ] Menghasilkan metrics JSON
- [ ] Menyertakan ringkasan pendek: metode + kendala

## DoD Final Project

Project dianggap selesai jika:
- [ ] Semua model utama sudah dijalankan
- [ ] Semua model dievaluasi dengan script yang sama
- [ ] Tabel komparasi final tersedia
- [ ] Analisis hasil (precision/recall/F1) tersedia
- [ ] Prototype Streamlit bisa demo deteksi + redaksi
- [ ] Laporan akhir lengkap dan siap submit

---

Last update: 30 Mei 2026