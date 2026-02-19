"""
data_pipeline.py
Handles data cleaning, SQLite persistence, and SQL-based KPI extraction.
"""

import sqlite3
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw transaction DataFrame:
    - Impute missing values
    - Clip outliers (IQR-based for amount)
    - Encode categoricals
    - Return analysis-ready DataFrame
    """
    df = raw.copy()

    # ── Missing value imputation ─────────────────────────────────────────────
    df["amount"] = df["amount"].fillna(df["amount"].median())
    df["transaction_frequency"] = df["transaction_frequency"].fillna(df["transaction_frequency"].median())

    # ── Outlier removal: clip amount at 99th percentile ──────────────────────
    upper = df["amount"].quantile(0.99)
    df["amount"] = df["amount"].clip(upper=upper)

    # ── Ensure correct dtypes ────────────────────────────────────────────────
    df["age"]                   = df["age"].astype(int)
    df["credit_score"]          = df["credit_score"].astype(int)
    df["account_tenure_months"] = df["account_tenure_months"].astype(int)
    df["transaction_frequency"] = df["transaction_frequency"].round().astype(int)
    df["is_high_risk"]          = df["is_high_risk"].astype(int)
    df["transaction_date"]      = pd.to_datetime(df["transaction_date"])

    # ── Derived features ─────────────────────────────────────────────────────
    df["year_month"]   = df["transaction_date"].dt.to_period("M").astype(str)
    df["day_of_week"]  = df["transaction_date"].dt.dayofweek          # 0=Mon

    # Amount buckets (useful for segmentation)
    df["amount_bucket"] = pd.cut(
        df["amount"],
        bins=[0, 100, 500, 1000, 5000, np.inf],
        labels=["<100", "100–500", "500–1k", "1k–5k", "5k+"],
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. SQLite Database
# ─────────────────────────────────────────────────────────────────────────────

def build_db(df: pd.DataFrame, conn: sqlite3.Connection) -> None:
    """
    Load the cleaned DataFrame into an EXISTING SQLite connection.
    Does NOT return the connection (Streamlit-safe).
    """
    db_df = df.drop(columns=["amount_bucket"], errors="ignore")
    db_df["transaction_date"] = db_df["transaction_date"].astype(str)
    db_df["year_month"]       = db_df["year_month"].astype(str)

    db_df.to_sql("transactions", conn, if_exists="replace", index=False)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_risk ON transactions(is_high_risk)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cat  ON transactions(category)")
    conn.commit()

    return conn


# ─────────────────────────────────────────────────────────────────────────────
# 3. SQL KPIs
# ─────────────────────────────────────────────────────────────────────────────

def run_sql_kpis(conn: sqlite3.Connection) -> dict:
    """
    Execute SQL queries to compute business KPIs.
    Returns a dictionary of scalar metrics.
    """
    kpis = {}

    kpis["total_volume"] = pd.read_sql_query(
        "SELECT SUM(amount) AS v FROM transactions", conn
    )["v"].iloc[0]

    kpis["avg_amount"] = pd.read_sql_query(
        "SELECT AVG(amount) AS v FROM transactions", conn
    )["v"].iloc[0]

    kpis["high_risk_count"] = pd.read_sql_query(
        "SELECT SUM(is_high_risk) AS v FROM transactions", conn
    )["v"].iloc[0]

    total = pd.read_sql_query(
        "SELECT COUNT(*) AS v FROM transactions", conn
    )["v"].iloc[0]
    kpis["risk_rate"] = kpis["high_risk_count"] / total

    kpis["unique_customers"] = pd.read_sql_query(
        "SELECT COUNT(DISTINCT customer_id) AS v FROM transactions", conn
    )["v"].iloc[0]

    return kpis
