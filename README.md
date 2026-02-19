# 🔐 TrustLens — Customer Transaction Risk & Behavior Analysis Dashboard

A production-quality Data Science project built with Python, Streamlit, and Machine Learning.

## 🚀 Quick Start (Replit)

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

3. Open the Replit preview URL — the dashboard loads automatically.

---

## 📁 Project Structure

```
├── app.py              # Main Streamlit dashboard
├── data_generator.py   # Synthetic transaction dataset (5,000 rows)
├── data_pipeline.py    # Data cleaning, SQLite DB, SQL KPIs
├── eda.py              # Exploratory Data Analysis charts
├── ml_models.py        # Model training, evaluation, live prediction
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🎯 Features

| Feature | Details |
|---|---|
| **EDA** | Risk distribution, transaction trends, correlation heatmap, category analysis |
| **SQL Explorer** | 4 preset business queries + custom SQL editor backed by SQLite |
| **ML Models** | Logistic Regression & Random Forest with full evaluation metrics |
| **Live Prediction** | Real-time risk scoring with dual-model comparison |
| **Dashboard Filters** | Age, amount, category, high-risk toggle |

---

## 🤖 Machine Learning

- **Logistic Regression** — interpretable baseline with L2 regularization
- **Random Forest** — 200 estimators, captures non-linear feature interactions
- Evaluated on: Accuracy, Precision, Recall, F1, ROC-AUC
- Class-balanced weighting handles the imbalanced dataset

---

## 💡 Business Insights

- **Credit score** is the strongest inverse predictor of risk
- **Travel** and **Electronics** categories carry elevated risk rates
- **High-amount transactions** (>$5k) are significantly more likely to be flagged
- **Young customers** (18–24) show higher risk but lower total spend
- **Account tenure** negatively correlates with risk — loyalty signals trust

---

## 📊 Dataset

Synthetic dataset of 5,000 customer transactions generated with realistic distributions:
- 500+ unique customers with age, credit score, and account tenure attributes
- 8 merchant categories with category-specific amount distributions
- Logistic-function-based risk labeling incorporating multiple behavioral signals
- ~2% missing values and ~1% outlier rows for realistic cleaning exercises

- ## project Deployment
- The project is live at : https://customer-transaction-risk-behaviour.onrender.com/
