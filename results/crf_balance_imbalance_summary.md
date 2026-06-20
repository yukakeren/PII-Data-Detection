# Ringkasan CRF: Balance vs Imbalance

## Hasil Dua Eksperimen

| experiment    | scenario   |   token_precision |   token_recall |   token_f1 |   entity_precision |   entity_recall |   entity_f1 |
|:--------------|:-----------|------------------:|---------------:|-----------:|-------------------:|----------------:|------------:|
| crf_imbalance | imbalance  |          0.801498 |       0.525799 |   0.635015 |           0.780822 |        0.457831 |    0.577215 |
| crf_balance   | balance    |          0        |       0        |   0        |           0        |        0        |    0        |

## Skenario Terbaik

Skenario terbaik untuk model CRF adalah **imbalance**, dengan token F1 `0.6350` dan entity F1 `0.5772`.