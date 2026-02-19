"""
Customer Transaction Risk & Behavior Analysis Dashboard
========================================================
A production-quality Streamlit app for transaction risk analysis,
EDA, SQL-based KPIs, and ML-driven customer classification.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import sqlite3
import warnings
from io import StringIO

warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrustLens · Risk Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #0d1117;
    color: #e6edf3;
  }
  .block-container { padding: 1.5rem 2.5rem; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
  }

  /* KPI Cards */
  .kpi-card {
    background: linear-gradient(135deg, #161b22, #1c2128);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
  }
  .kpi-card:hover { border-color: #58a6ff; }
  .kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #58a6ff;
    margin: 0;
  }
  .kpi-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8b949e;
    margin-top: 0.3rem;
  }
  .kpi-delta { font-size: 0.85rem; margin-top: 0.2rem; }
  .up { color: #3fb950; }
  .down { color: #f85149; }

  /* Section headers */
  .section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #8b949e;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem;
  }

  /* Risk badge */
  .risk-high {
    background: #3d1a1a; color: #f85149;
    padding: 2px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;
  }
  .risk-low {
    background: #0e2a1a; color: #3fb950;
    padding: 2px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;
  }

  /* DataFrame */
  [data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }

  /* Tabs */
  .stTabs [data-baseweb="tab"] { color: #8b949e; font-size: 0.85rem; }
  .stTabs [aria-selected="true"] { color: #58a6ff; border-bottom: 2px solid #58a6ff; }

  /* Prediction box */
  .pred-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
  }
  .pred-high { border-color: #f85149; background: #1a0d0d; }
  .pred-low  { border-color: #3fb950; background: #0a1a0d; }
  .pred-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
  }
</style>
""", unsafe_allow_html=True)

# ── Import project modules ────────────────────────────────────────────────────
from data_generator import generate_data
from data_pipeline   import clean_data, build_db, run_sql_kpis
from eda             import plot_risk_dist, plot_transaction_trend, plot_correlation, plot_top_categories
from ml_models       import train_models, predict_single

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_connection():
    return sqlite3.connect(":memory:", check_same_thread=False)

@st.cache_data(show_spinner=False)
def load_pipeline():
    raw = generate_data(n=5000, seed=42)
    df  = clean_data(raw)
    return df


@st.cache_resource(show_spinner=False)
def load_models(df):
    return train_models(df)

with st.spinner("🔐 Loading TrustLens engine…"):
    df = load_pipeline()

    conn = get_connection()
    build_db(df, conn)          # 👈 slight change (see below)
    kpis = run_sql_kpis(conn)

    model_bundle = load_models(df)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔐 TrustLens")
    st.caption("Customer Risk Intelligence Platform")
    st.divider()

    st.markdown("**Filter Data**")
    age_range = st.slider("Age range", 18, 80, (18, 80))
    amt_range = st.slider("Transaction amount ($)", 0, 10000, (0, 10000))
    selected_cats = st.multiselect(
        "Category",
        options=sorted(df["category"].unique()),
        default=sorted(df["category"].unique()),
    )
    show_high_risk = st.checkbox("High-risk only", False)
    st.divider()
    st.caption(f"Dataset: **{len(df):,}** transactions · SQLite DB active")

# Apply filters
mask = (
    df["age"].between(*age_range)
    & df["amount"].between(*amt_range)
    & df["category"].isin(selected_cats)
)
if show_high_risk:
    mask &= df["is_high_risk"] == 1
fdf = df[mask]

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("# Customer Transaction Risk & Behavior Analysis")
st.markdown(
    f"<p style='color:#8b949e;margin-top:-0.5rem;'>Showing <b style='color:#e6edf3'>{len(fdf):,}</b> "
    f"of {len(df):,} transactions after filters · "
    f"High-risk rate: <b style='color:#f85149'>{fdf['is_high_risk'].mean():.1%}</b></p>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="section-title">Key Performance Indicators (SQL)</p>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
kpi_items = [
    (c1, "Total Volume",    f"${kpis['total_volume']:,.0f}",   None),
    (c2, "Avg Transaction", f"${kpis['avg_amount']:.2f}",       None),
    (c3, "High-Risk Count", f"{kpis['high_risk_count']:,}",     "down"),
    (c4, "Risk Rate",       f"{kpis['risk_rate']:.1%}",         "down"),
    (c5, "Unique Customers",f"{kpis['unique_customers']:,}",     None),
]
for col, label, val, direction in kpi_items:
    icon = "▲" if direction == "up" else ("▼" if direction == "down" else "")
    cls  = direction or ""
    with col:
        st.markdown(
            f'<div class="kpi-card">'
            f'  <p class="kpi-value">{val}</p>'
            f'  <p class="kpi-label">{label}</p>'
            f'  {"<p class=kpi-delta><span class=" + cls + ">" + icon + "</span></p>" if icon else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Exploratory Analysis",
    "🗃️ SQL Query Explorer",
    "🤖 ML Model Performance",
    "🎯 Live Prediction",
    "📄 Documentation",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · EDA
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-title">Exploratory Data Analysis</p>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Risk Distribution**")
        fig = plot_risk_dist(fdf)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_r:
        st.markdown("**Transaction Amount by Category**")
        fig = plot_top_categories(fdf)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("**Transaction Trend Over Time**")
    fig = plot_transaction_trend(fdf)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("**Feature Correlation Heatmap**")
    fig = plot_correlation(fdf)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("**Raw Data Sample**")
    st.dataframe(
        fdf.sample(min(200, len(fdf)), random_state=1).reset_index(drop=True),
        use_container_width=True, height=300,
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · SQL Explorer
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-title">SQL Query Explorer</p>', unsafe_allow_html=True)

    preset_queries = {
        "Top 10 highest-risk categories": """
SELECT category,
       COUNT(*) AS total_transactions,
       SUM(is_high_risk) AS high_risk_count,
       ROUND(AVG(is_high_risk)*100, 2) AS risk_rate_pct,
       ROUND(AVG(amount), 2) AS avg_amount
FROM transactions
GROUP BY category
ORDER BY risk_rate_pct DESC
LIMIT 10;""",
        "Monthly transaction volume": """
SELECT strftime('%Y-%m', transaction_date) AS month,
       COUNT(*) AS num_transactions,
       ROUND(SUM(amount), 2) AS total_volume,
       SUM(is_high_risk) AS flagged
FROM transactions
GROUP BY month
ORDER BY month;""",
        "Risk rate by age bucket": """
SELECT
  CASE
    WHEN age < 25 THEN '18-24'
    WHEN age < 35 THEN '25-34'
    WHEN age < 45 THEN '35-44'
    WHEN age < 55 THEN '45-54'
    ELSE '55+'
  END AS age_group,
  COUNT(*) AS total,
  SUM(is_high_risk) AS high_risk,
  ROUND(AVG(is_high_risk)*100, 2) AS risk_pct
FROM transactions
GROUP BY age_group
ORDER BY age_group;""",
        "High-risk customers above $5k": """
SELECT customer_id, COUNT(*) AS txn_count,
       ROUND(SUM(amount), 2) AS total_spent,
       SUM(is_high_risk) AS flagged_txns
FROM transactions
WHERE amount > 5000
GROUP BY customer_id
HAVING flagged_txns > 0
ORDER BY flagged_txns DESC
LIMIT 20;""",
        "Custom query": "SELECT * FROM transactions LIMIT 50;",
    }

    chosen = st.selectbox("Preset queries", list(preset_queries.keys()))
    query  = st.text_area("SQL", value=preset_queries[chosen].strip(), height=160)

    if st.button("▶ Run Query", type="primary"):
        try:
            result = pd.read_sql_query(query, conn)
            st.success(f"{len(result):,} rows returned")
            st.dataframe(result, use_container_width=True)
        except Exception as e:
            st.error(f"SQL error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · ML Models
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-title">Machine Learning Model Evaluation</p>', unsafe_allow_html=True)

    metrics_df = model_bundle["metrics"]
    st.dataframe(
        metrics_df.style.highlight_max(subset=["Accuracy","Precision","Recall","F1","ROC-AUC"],
                                       color="#0e2a1a"),
        use_container_width=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**ROC Curves**")
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        colors = {"Logistic Regression": "#58a6ff", "Random Forest": "#3fb950"}
        for name, fpr, tpr, auc_val in model_bundle["roc_data"]:
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", color=colors[name], lw=2)
        ax.plot([0,1],[0,1],"--", color="#8b949e", lw=1)
        ax.set_xlabel("False Positive Rate", color="#8b949e")
        ax.set_ylabel("True Positive Rate", color="#8b949e")
        ax.tick_params(colors="#8b949e")
        ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=9)
        for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_b:
        st.markdown("**Feature Importance (Random Forest)**")
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        fi = model_bundle["feature_importance"].head(12)
        bars = ax.barh(fi["feature"], fi["importance"], color="#58a6ff", height=0.6)
        ax.set_xlabel("Importance", color="#8b949e")
        ax.tick_params(colors="#8b949e")
        ax.invert_yaxis()
        for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("**Confusion Matrices**")
    cm_cols = st.columns(2)
    for i, (name, cm) in enumerate(model_bundle["confusion_matrices"]):
        with cm_cols[i]:
            st.markdown(f"*{name}*")
            fig, ax = plt.subplots(figsize=(3.5, 3), facecolor="#0d1117")
            ax.set_facecolor("#0d1117")
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        linecolor="#30363d", linewidths=0.5,
                        xticklabels=["Low","High"], yticklabels=["Low","High"])
            ax.set_xlabel("Predicted", color="#8b949e")
            ax.set_ylabel("Actual", color="#8b949e")
            ax.tick_params(colors="#8b949e")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · Live Prediction
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-title">Real-Time Risk Prediction</p>', unsafe_allow_html=True)
    st.caption("Enter customer & transaction details to get an instant risk score from both models.")

    with st.form("prediction_form"):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            p_age    = st.number_input("Age",              18, 80,   35)
            p_amount = st.number_input("Amount ($)",       0.0, 50000.0, 500.0, step=50.0)
        with pc2:
            p_cat    = st.selectbox("Category", sorted(df["category"].unique()))
            p_freq   = st.number_input("Monthly txn frequency", 1, 200, 10)
        with pc3:
            p_credit = st.number_input("Credit score",    300, 850,  650)
            p_tenure = st.number_input("Account tenure (months)", 0, 240, 36)

        submitted = st.form_submit_button("⚡ Predict Risk", type="primary", use_container_width=True)

    if submitted:
        input_row = {
            "age": p_age, "amount": p_amount, "category": p_cat,
            "transaction_frequency": p_freq, "credit_score": p_credit,
            "account_tenure_months": p_tenure,
        }
        results = predict_single(input_row, model_bundle, df)

        res_cols = st.columns(len(results))
        for col, (model_name, label, prob) in zip(res_cols, results):
            css = "pred-high" if label == 1 else "pred-low"
            risk_text = "HIGH RISK" if label == 1 else "LOW RISK"
            color = "#f85149" if label == 1 else "#3fb950"
            with col:
                st.markdown(
                    f'<div class="pred-box {css}">'
                    f'  <p style="color:#8b949e;font-size:0.8rem;margin-bottom:0.5rem">{model_name}</p>'
                    f'  <p class="pred-score" style="color:{color}">{prob:.0%}</p>'
                    f'  <p style="color:{color};font-weight:700;font-size:0.9rem">{risk_text}</p>'
                    f'  <p style="color:#8b949e;font-size:0.75rem">risk probability</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 · Documentation / README
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("""
## 📄 TrustLens — Documentation

### Problem Statement
Financial institutions process millions of transactions daily. Identifying high-risk customers
early reduces fraud losses, improves compliance posture, and enables targeted interventions.
This platform automates risk classification using behavioral and transactional signals.

---

### Data Pipeline
| Stage | Tool | Description |
|---|---|---|
| Generation | NumPy / Pandas | Synthetic 5,000-row dataset with realistic distributions |
| Cleaning | Pandas | Outlier clipping, missing-value imputation, one-hot encoding |
| Storage | SQLite | Loaded into `transactions` table for SQL analytics |
| Feature engineering | Scikit-learn | Scaling, categorical encoding, train/test split |

---

### Machine Learning Approach
Two classifiers are trained and compared on an 80/20 stratified split:

**Logistic Regression** — fast, interpretable baseline with L2 regularization.

**Random Forest** — ensemble of 200 decision trees; captures non-linear interactions
and yields feature importances for explainability.

Both models are evaluated on Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

---

### Business Insights
- **Category risk variance**: Some merchant categories carry 3–4× baseline risk.
- **Amount threshold**: Transactions above $5,000 show significantly elevated flagging rates.
- **Age dynamics**: Younger customers (18–24) show higher risk but lower total volume.
- **Credit score**: Strong inverse predictor of risk — the single most informative feature.
- **Account tenure**: Longer-tenured accounts exhibit lower risk on average.

---

### How to Run on Replit
```
pip install -r requirements.txt
streamlit run app.py
```

---

### Project Structure
```
├── app.py              # Streamlit dashboard (this file)
├── data_generator.py   # Synthetic dataset generation
├── data_pipeline.py    # Cleaning, SQLite loading, SQL KPIs
├── eda.py              # EDA chart functions
├── ml_models.py        # Model training, evaluation, prediction
└── requirements.txt    # Dependencies
```
""")
