# Baseline Models for PII Detection

This folder contains baseline machine learning models for token-level Personally Identifiable Information (PII) detection.

The models used are:

- Logistic Regression
- Linear SVM

## Method

Baseline models are implemented as token-level classifiers. Each token from the dataset is converted into numerical features and classified into either a PII label or non-PII label.

The models are trained using processed datasets from the project split:

```text
data/processed/
├── train.json
├── val.json
└── test_internal.json
```

The final evaluation is performed on `test_internal.json`.

## Feature Engineering

Each token is converted into handcrafted token-level features and TF-IDF representation.

The handcrafted features used include:

- token length
- character features
- capitalization pattern
- digit pattern
- email pattern
- URL pattern
- prefix and suffix
- previous token
- next token

TF-IDF representation is used to convert token text into numerical features based on character patterns. In this experiment, TF-IDF uses character n-grams so the model can capture useful patterns inside tokens, such as email format, URL pattern, name pattern, and ID number pattern.

Context tokens are included because Logistic Regression and Linear SVM do not directly model token sequences like CRF.

## Hyperparameter

Hyperparameters are model settings defined before training. These values control how the model learns from the training data.

### Logistic Regression

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=150,
    solver="saga",
    tol=1e-3,
    verbose=0
)
```

Explanation:

- `class_weight="balanced"` is used to handle class imbalance because non-PII tokens are much more frequent than PII tokens.
- `max_iter=150` sets the maximum number of optimization iterations.
- `solver="saga"` is used because it supports large-scale sparse features.
- `tol=1e-3` controls the stopping tolerance. A larger tolerance helps training finish faster.
- `verbose=0` disables detailed training logs.

### Linear SVM

```python
LinearSVC(
    class_weight="balanced",
    max_iter=1000,
    tol=1e-3
)
```

Explanation:

- `class_weight="balanced"` is used to reduce the effect of class imbalance.
- `max_iter=1000` sets the maximum number of training iterations.
- `tol=1e-3` controls the stopping tolerance during optimization.

## Run Training

Train the baseline models and generate predictions, metrics, and trained model files:

```bash
python models/baseline/train_baseline.py
```

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
Token-level Precision : 0.0128
Token-level Recall    : 0.9785
Token-level F1-score  : 0.0253

Entity-level Precision: 0.0073
Entity-level Recall   : 0.7942
Entity-level F1-score : 0.0146
```

Logistic Regression achieved very high recall but very low precision. This means the model detected many PII tokens, but also produced many false positives.

### Linear SVM

```text
Token-level Precision : 0.7565
Token-level Recall    : 0.6244
Token-level F1-score  : 0.6841

Entity-level Precision: 0.4336
Entity-level Recall   : 0.4568
Entity-level F1-score : 0.4449
```

Linear SVM achieved better and more balanced performance compared to Logistic Regression.

## Short Analysis

Logistic Regression tends to over-detect PII tokens. It has a very high recall score, but the precision is very low because many non-PII tokens are incorrectly predicted as PII.

Linear SVM performs better as a baseline model. It produces a more balanced result between precision and recall, with a token-level F1-score of 0.6841 and an entity-level F1-score of 0.4449.

## Conclusion

Linear SVM is the stronger baseline model in this experiment. It provides a better balance between precision and recall and can be used as the main baseline comparison for CRF, boosting models, ensemble models, and deep learning models.