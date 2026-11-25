# 🛡️ Fraud Detection System

An end-to-end machine learning pipeline that detects fraudulent card transactions on a **highly imbalanced dataset (1.5% fraud)**, balancing false positives against false negatives using Precision, Recall and F1-score.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![Status](https://img.shields.io/badge/status-complete-brightgreen)

## 🎯 Problem Statement
Fraud is rare but expensive. A model that predicts "legit" every time is 98.5% accurate yet useless. The goal is to **catch as much fraud as possible without annoying genuine customers** with false alarms.

## 📂 Dataset
60,000 transactions with 11 features (amount, hour, merchant category, payment channel, distance from home, transactions in last 24h, card age, foreign flag, new-device flag, customer age).
Generated with `generate_data.py` (includes missing values and deliberately "hard" fraud cases that mimic normal behaviour).
> 💡 To use a real dataset (e.g. Kaggle *Credit Card Fraud Detection*), replace `transactions.csv` and update the column lists in `fraud_detection.py`.

## 🔬 Methodology
| Step | What was done |
|---|---|
| **1. EDA** | Fraud patterns by hour, merchant category, channel, amount and distance |
| **2. Preprocessing** | Median / mode imputation, standard scaling, one-hot encoding of categoricals |
| **3. Feature engineering** | `is_night`, `log_amount`, `amount_per_txn_24h`, `new_card_flag` |
| **4. Imbalance handling** | Stratified split, then minority **oversampling on training data only** (no test-set leakage) |
| **5. Modelling** | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting |
| **6. Evaluation** | Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix |
| **7. Threshold tuning** | Sweep decision thresholds to pick the best precision/recall trade-off |
| **8. Interpretability** | Feature importance to explain fraud drivers |

## 📊 Results (Random Forest – best model)
| Metric | Score |
|---|---|
| Precision | **0.89** |
| Recall | **0.85** |
| F1-score | **0.87** |
| ROC-AUC | 0.946 |
| PR-AUC | 0.810 |

Out of 229 fraud cases in the test set, the model caught **195 (85%)** while flagging only **0.16%** of legitimate transactions.

**Top fraud indicators:** card age, distance from home, night-time transactions, transaction velocity (txns in last 24h), hour of day.

## 💡 Key Insights
- Fraud peaks between **11 PM and 5 AM** (13–17% fraud rate vs <1% in the daytime).
- **Jewelry, electronics and online services** are the riskiest categories; **online** is the riskiest channel.
- Fraudulent transactions occur far from the customer's home and on **newer cards / new devices**.
- Accuracy (99.6%) is misleading; **PR-AUC and F1** are the right metrics for imbalanced data.

## 🚀 How to Run
```bash
pip install -r requirements.txt
python generate_data.py        # creates transactions.csv
python fraud_detection.py      # runs the full pipeline, saves charts to outputs/
```

## 📁 Structure
```
fraud_detection_system/
├── generate_data.py
├── fraud_detection.py
├── transactions.csv
├── requirements.txt
└── outputs/   (EDA, model comparison, threshold tuning, confusion matrix, feature importance)
```

## 🔭 Future Work
Hyperparameter tuning (GridSearchCV), SMOTE / cost-sensitive learning, XGBoost, SHAP explainability, Streamlit app for real-time scoring.

## 👩‍💻 Author
**Bhagyashree Ghosh** · [LinkedIn](https://www.linkedin.com/in/bhagyashreeg4/) · [GitHub](https://github.com/bhagyashreeghosh004-hub)
