# Model Comparison

## Token-Level Metrics

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| hybrid | 0.8645 | 0.9696 | 0.9140 |
| lightgbm_balance | 0.1343 | 0.7082 | 0.2257 |
| lightgbm_imbalance | 0.2323 | 0.6018 | 0.3352 |
| xgboost_balance | 0.1220 | 0.7082 | 0.2082 |
| xgboost_imbalance | 0.2087 | 0.6672 | 0.3180 |

## Entity-Level Metrics

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| hybrid | 0.7987 | 0.9525 | 0.8688 |
| lightgbm_balance | 0.0920 | 0.6412 | 0.1609 |
| lightgbm_imbalance | 0.1592 | 0.5172 | 0.2435 |
| xgboost_balance | 0.0851 | 0.6438 | 0.1503 |
| xgboost_imbalance | 0.1510 | 0.6042 | 0.2416 |

## Per-Class F1 Scores

| Label | hybrid | lightgbm_balance | lightgbm_imbalance | xgboost_balance | xgboost_imbalance |
|-------|--------|------------------|--------------------|-----------------|-------------------|
| B-EMAIL | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| B-ID_NUM | 0.7667 | 0.1042 | 0.2857 | 0.1081 | 0.2975 |
| B-NAME_STUDENT | 0.9376 | 0.3347 | 0.3908 | 0.3352 | 0.3935 |
| B-PHONE_NUM | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B-STREET_ADDRESS | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B-URL_PERSONAL | 0.5319 | 0.5882 | 0.5797 | 0.5952 | 0.5882 |
| B-USERNAME | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| I-ID_NUM | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| I-NAME_STUDENT | 0.9665 | 0.3381 | 0.3894 | 0.3252 | 0.3857 |
| I-PHONE_NUM | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| I-STREET_ADDRESS | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |