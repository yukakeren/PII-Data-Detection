# Hybrid Model (Regex + ML)

This folder contains the implementation of the Hybrid Model for PII Shield.

## Approach

The hybrid model combines Rule-based (Regex) predictions with the best available Machine Learning predictions:
1. **Best ML Model Identification**: The script reads `results/metrics/` to dynamically select the ML model with the highest token-level F1 score.
2. **Rule-based Layer**: It applies regular expressions to detect patterns such as Email, URL, Phone Number, and ID Number on each token.
3. **Combination Strategy**:
   - If a regex rule is triggered for a token, the rule's prediction is used.
   - Otherwise, the ML model's prediction is used as the fallback.
4. **Evaluation**: Finally, the hybrid predictions are evaluated using the standard `evaluate_from_csv` method to compare against the baseline ML.

## Files

- `hybrid_model.py`: The main script that applies the hybrid logic.
- `README.md`: This file.

## Running

You can run the model by executing the notebook located at `notebooks/hybrid_ml_regex.ipynb` or by running the script directly from the root directory:

```bash
python3 models/hybrid/hybrid_model.py
```

## Results

The predictions and metrics will be saved in:
- **Predictions**: `results/predictions/hybrid_predictions.csv`
- **Metrics**: `results/metrics/hybrid_metrics.json`
