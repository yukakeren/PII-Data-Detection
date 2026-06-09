# Hybrid Rule + ML for PII Detection

This model is a practical extension of the CRF work: a rule-based detector for structured PII combined with an ML component for more contextual entities, especially `NAME_STUDENT`.

## What It Does

- Regex rules for `EMAIL`, `URL_PERSONAL`, `PHONE_NUM`, and `ID_NUM`
- Lightweight context rules for `USERNAME`, `NAME_STUDENT`, and `STREET_ADDRESS`
- Optional integration with the CRF model when `models/crf/crf_model.pkl` is available
- Redaction / anonymization into placeholders such as `[EMAIL]` and `[NAME_STUDENT]`

## Run from the CLI

```bash
python3 models/hybrid/predict.py "My name is Michael Jordan and my email is mjordan@nba.com"
```

Available modes:

```bash
python3 models/hybrid/predict.py "my username is student_123" --mode rules_only
python3 models/hybrid/predict.py "My name is Farras" --mode crf_only
```
