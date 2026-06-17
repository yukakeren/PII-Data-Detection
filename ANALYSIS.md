# PII Shield: Data Analysis and Model Selection

This document provides an overview of the dataset characteristics, label distribution, and the rationale behind the machine learning models selected for the Personally Identifiable Information (PII) detection task.

## 1. Dataset Characteristics

Based on the analysis of the `train.json` dataset, here are the core characteristics of the educational essays we are processing:

- **Total Documents**: 6,807 essays
- **Average Document Length**: ~733 tokens
- **Maximum Document Length**: 3,298 tokens
- **Minimum Document Length**: 69 tokens

The document length indicates that context is relatively long, which is a crucial factor when selecting models (e.g., models with small context windows might struggle with long documents).

### Label Distribution

The dataset is **extremely imbalanced**, which is typical for PII detection tasks. The vast majority of the text is not PII. 

| Label | Count |
| :--- | :--- |
| `O` (Outside / No PII) | 4,989,794 |
| `B-NAME_STUDENT` | 1,365 |
| `I-NAME_STUDENT` | 1,096 |
| `B-URL_PERSONAL` | 110 |
| `B-ID_NUM` | 78 |
| `B-EMAIL` | 39 |
| `I-STREET_ADDRESS`| 20 |
| `I-PHONE_NUM` | 15 |
| `B-USERNAME` | 6 |
| `B-PHONE_NUM` | 6 |
| `B-STREET_ADDRESS`| 2 |
| `I-URL_PERSONAL` | 1 |
| `I-ID_NUM` | 1 |

**Key Takeaways from the Data:**
1. **Sparsity**: PII entities make up less than 0.1% of the total tokens. 
2. **Class Imbalance**: Evaluating model performance using Accuracy is misleading. We must use **Precision, Recall, and F1-Score** (specifically on the PII classes) to properly measure success.
3. **Rare Classes**: Certain classes like `STREET_ADDRESS` and `PHONE_NUM` have fewer than 20 occurrences in the entire training set, making them exceptionally difficult to detect without pre-trained knowledge or heavy data augmentation.

---

## 2. Model Selection Rationale

Given the data characteristics—specifically the sequential nature of text and the extreme class imbalance—we have structured our modeling approach from simple baselines to advanced contextual models.

### Currently Implemented Models

#### A. Logistic Regression & Linear SVM (Baseline ML)
- **Why we chose them**: These serve as our foundational baselines. They are incredibly fast to train and easy to implement. By using simple token-level features (e.g., TF-IDF, word shape, capitalization, length), we can establish the lower bound of performance. 
- **Pros**: Fast inference, easy to interpret, good for debugging the data pipeline.
- **Cons**: They treat each token independently and fail to capture sequential context (e.g., they don't know that `I-NAME_STUDENT` must follow `B-NAME_STUDENT`).

#### B. Conditional Random Fields (CRF)
- **Why we chose it**: CRF is a classic, powerful statistical modeling method specifically designed for sequence labeling tasks like Named Entity Recognition (NER).
- **Pros**: Unlike Logistic Regression, CRF models the transition probabilities between labels. It explicitly learns that an `I-` tag must be preceded by a `B-` or `I-` tag of the same entity type. This significantly reduces invalid tag sequences.
- **Cons**: Feature engineering can be tedious (requires manually defining features like "is_upper", "prev_word", "next_word"), and it struggles to capture long-range dependencies.

---

### Next Steps: Advanced Models

#### C. XGBoost / LightGBM
- **Why we chose them**: Tree-based boosting algorithms are highly robust to class imbalance. We can utilize techniques like `scale_pos_weight` or custom focal loss to penalize the model heavily for missing rare PII classes.
- **Pros**: Excellent at handling non-linear interactions between engineered token features. Often outperforms linear baselines without requiring the massive compute of neural networks.
- **Cons**: Still requires manual feature engineering and sliding window approaches to capture context effectively.

#### D. Deep Learning (Transformers - e.g., DistilBERT, DeBERTa)
- **Why we chose them**: State-of-the-art for NLP and NER tasks. PII detection heavily relies on context—for example, differentiating between "John" (a historical figure mentioned in an essay, label `O`) and "John" (the student author, label `B-NAME_STUDENT`). Transformers excel at this.
- **Pros**: 
  - **Contextual Understanding**: Bidirectional attention captures full-sentence context, eliminating the need for manual feature engineering.
  - **Transfer Learning**: Pre-trained on massive corpora, meaning they already understand what an address or an email looks like, which is critical for our extremely rare classes (`STREET_ADDRESS`, `PHONE_NUM`).
- **Cons**: Computationally expensive to train and deploy. To mitigate this, we plan to experiment with lighter models like **DistilBERT** for faster inference in the Streamlit app, while using **DeBERTa** for maximum performance.
