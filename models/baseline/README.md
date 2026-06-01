# Baseline Models for PII Detection

This folder contains baseline machine learning models for token-level Personally Identifiable Information (PII) detection.

The models used are:

- Logistic Regression
- Linear SVM

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Run Training

Train the baseline models and generate predictions + metrics:

```bash
python models/baseline/train_baseline.py
```

## Dataset

These models use processed datasets from:

```text
data/processed/
├── train.json
├── val.json
└── test_internal.json
```

## Features

Each token is converted into simple token-level features, including:

- lowercase token
- token shape
- token length
- capitalization pattern
- digit pattern
- email pattern
- URL pattern
- prefix and suffix
- previous token
- next token

## Outputs

### Predictions

Generated prediction files:

```text
results/predictions/logistic_regression_predictions.csv
results/predictions/linear_svm_predictions.csv
```

CSV format:

```text
document_id,token,true_label,pred_label
```

### Metrics

Generated evaluation metrics:

```text
results/metrics/logistic_regression_metrics.json
results/metrics/linear_svm_metrics.json
```

### Trained Models

Saved trained models:

```text
models/baseline/logistic_regression_model.joblib
models/baseline/linear_svm_model.joblib
```

## Results

### Logistic Regression

```text
Token-level Precision : 0.0114
Token-level Recall    : 0.9737
Token-level F1-score  : 0.0226

Entity-level Precision: 0.0066
Entity-level Recall   : 0.8354
Entity-level F1-score : 0.0132
```

Logistic Regression achieved very high recall but very low precision. This means the model detected many PII tokens, but also produced many false positives.

### Linear SVM

```text
Token-level Precision : 0.7536
Token-level Recall    : 0.6220
Token-level F1-score  : 0.6815

Entity-level Precision: 0.4545
Entity-level Recall   : 0.4733
Entity-level F1-score : 0.4637
```

Linear SVM achieved better and more balanced performance compared to Logistic Regression.

## Conclusion

Linear SVM is the stronger baseline model in this experiment. It provides a better balance between precision and recall and can be used as the main baseline comparison for CRF, boosting models, ensemble models, and deep learning models.