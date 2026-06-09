# Transformer Models for PII Detection

## Project Role

- `No. 1` prepares the data pipeline, preprocessing, split, and evaluation.
- `No. 2` provides baseline comparison models:
  - Logistic Regression: token F1 `0.0253`, entity F1 `0.0146`
  - Linear SVM: token F1 `0.6841`, entity F1 `0.4449`
- `No. 3` provides a classical sequence model:
  - CRF: token F1 `0.7940`, entity F1 `0.7692`
- `No. 4` provides context-aware deep learning models:
  - `DistilBERT` as the lighter transformer baseline
  - `DeBERTa` as the stronger context-focused transformer

The transformer outputs are compared against the baseline and CRF results, then reused by the demo application for detection and anonymization.

## Folder Contents

- `train_transformer.py`: shared transformer fine-tuning script
- `launch_training.py`: shared launcher for long-running training jobs
- `train_deberta.py`: stable DeBERTa-focused training script
- `launch_deberta.py`: DeBERTa-specific launcher
- `pipeline.py`: inference wrapper for saved local models
- `predict.py`: CLI for free-text prediction
- `evaluate_saved_transformer.py`: re-evaluate a saved model without retraining
- `requirements.txt`: extra dependencies for transformer experiments

## Dataset Layout

Training expects:

```text
data/processed/
├── train.json
├── val.json
└── test_internal.json
```

## Environment

Install dependencies:

```bash
pip install -r models/transformer/requirements.txt
```

Notes:

- `DeBERTa` requires `sentencepiece` for tokenization.
- If you are using the repository virtual environment:

```bash
source .venv/bin/activate
```

## Manual Training

### DistilBERT

```bash
python models/transformer/train_transformer.py \
  --model-key distilbert \
  --epochs 3 \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --max-length 256 \
  --chunk-size 128 \
  --loss-mode sqrt_balanced
```

### DeBERTa

```bash
python models/transformer/train_deberta.py \
  --artifact-prefix deberta_base \
  --output-dir models/transformer/deberta-pii \
  --epochs 5 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 \
  --max-length 256 \
  --chunk-size 128 \
  --loss-mode sqrt_balanced
```

## Background Launchers

### Shared Launcher

```bash
python models/transformer/launch_training.py --preset distilbert_full --detach
```

### DeBERTa Launcher

```bash
python models/transformer/launch_deberta.py --detach
```

If you need a dry run first:

```bash
python models/transformer/launch_deberta.py --dry-run
```

Launchers write:

- log: `results/logs/transformer/<run_name>.log`
- metadata: `results/logs/transformer/<run_name>.json`
- background process PID

To monitor progress:

```bash
tail -f results/logs/transformer/<run_name>.log
```

## Training Features

- automatically reads the latest processed dataset split
- automatically chunks long documents
- supports `DistilBERT` and `DeBERTa`
- supports `resume_from_checkpoint`
- supports class reweighting with `sqrt_balanced`
- uses GPU automatically when available unless `--force-cpu` is set
- saves a `run_summary.json` file inside the model directory

## Outputs

Training saves:

- local model directory under `models/transformer/`
- prediction CSV under `results/predictions/<artifact_prefix>_predictions.csv`
- metrics JSON under `results/metrics/<artifact_prefix>_metrics.json`

## Inference

### CLI

```bash
python models/transformer/predict.py \
  "My name is Michael Jordan and my email is mjordan@nba.com" \
  --model-key distilbert
```

### Simple Wrapper

```bash
python predict_transformer.py -1 "My name is Michael Jordan"
python predict_transformer.py -2 "My name is Michael Jordan"
```

## Technical Notes

- `DistilBERT` is lighter and faster, which makes it a good deep learning baseline.
- `DeBERTa` is stronger at contextual understanding, but heavier.
- The inference pipeline also uses chunking, so long inputs are not forced into a single forward pass.
