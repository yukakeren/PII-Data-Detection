# CRF Model for PII Detection

This model uses **Conditional Random Fields (CRF)** for token-level Personally Identifiable Information (PII) detection.

## Installation

Install required dependencies:

```bash
pip install -r models/crf/requirements.txt
```

## Run Training

Train the CRF model and generate predictions + metrics:

```bash
python models/crf/train_crf.py
```

## Dataset

This model uses processed datasets from:

```text
data/processed/
├── train.json
├── val.json
└── test_internal.json
```

## Outputs

### Predictions

Generated prediction file:

```text
results/predictions/crf_predictions.csv
```

CSV format:

```text
document_id,token,true_label,pred_label
```

### Metrics

Generated evaluation metrics:

```text
results/metrics/crf_metrics.json
```

### Trained Model

Saved trained model:

```text
models/crf/crf_model.pkl
```
