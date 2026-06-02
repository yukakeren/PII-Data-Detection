# CRF Model for PII Detection

This model uses **Conditional Random Fields (CRF)** for **token-level Personally Identifiable Information (PII) detection** using a **Named Entity Recognition (NER)** approach with BIO tagging.

## Features

The CRF model uses handcrafted sequence features including:

* Token lowercase representation
* Capitalization features
* Prefix and suffix (1–3 characters)
* Word shape patterns
* Regex-based features:

  * Email pattern
  * URL pattern
  * Phone pattern
* Character-level indicators:

  * Contains digit
  * Contains `@`
  * Contains dot (`.`)
  * Contains dash (`-`)
* Context window:

  * Previous 2 tokens
  * Next 2 tokens

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

## CRF Configuration

Best configuration (baseline):

```python
algorithm="lbfgs"
c1=0.01
c2=0.01
max_iterations=300
all_possible_transitions=True
```

Several tuning experiments were conducted (`Exp A–D`), but the baseline configuration produced the best validation performance.

## Dataset

This model uses processed datasets from:

```text
data/processed/
├── train.json
├── val.json
└── test_internal.json
```

Training pipeline:

* Train → `train.json`
* Hyperparameter tuning → `val.json`
* Final prediction → `test_internal.json`

## Evaluation

Run evaluation:

```bash
python -c "
from src.evaluate import evaluate_from_csv, print_metrics

metrics = evaluate_from_csv(
'results/predictions/crf_predictions.csv',
'crf'
)

print_metrics(metrics)
"
```

### Best Validation Result

| Metric           |  Score |
| ---------------- | -----: |
| Token Precision  | 0.8872 |
| Token Recall     | 0.7185 |
| Token F1         | 0.7940 |
| Entity Precision | 0.8871 |
| Entity Recall    | 0.6790 |
| Entity F1        | 0.7692 |

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

## Known Limitations

Current limitations of the CRF model:

* Limited performance on `USERNAME`
* No predictions for rare entities such as `PHONE_NUM` and `STREET_ADDRESS`
* Performance may be affected by label imbalance in the dataset
