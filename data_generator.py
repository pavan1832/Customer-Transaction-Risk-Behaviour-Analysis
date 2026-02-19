"""
data_generator.py
Generates a realistic synthetic customer transaction dataset.
"""

import numpy as np
import pandas as pd


def generate_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate n synthetic transaction records.
    Returns a raw DataFrame before any cleaning.
    """
    rng = np.random.default_rng(seed)

    # ── Customer attributes ──────────────────────────────────────────────────
    n_customers = max(500, n // 8)
    customer_ids     = np.arange(1, n_customers + 1)
    customer_ages    = rng.integers(18, 81, size=n_customers)
    customer_credits = rng.normal(650, 90, size=n_customers).clip(300, 850).astype(int)
    customer_tenure  = rng.integers(1, 241, size=n_customers)          # months

    # Each transaction belongs to a random customer
    cust_idx = rng.integers(0, n_customers, size=n)
    ages     = customer_ages[cust_idx]
    credits  = customer_credits[cust_idx]
    tenures  = customer_tenure[cust_idx]

    # ── Transaction attributes ───────────────────────────────────────────────
    categories = [
        "Retail", "Grocery", "Travel", "Dining",
        "Electronics", "Healthcare", "Entertainment", "Online Services",
    ]
    cat_array = rng.choice(categories, size=n)

    # Amount: log-normal, category-adjusted
    cat_amount_scale = {
        "Retail": 1.3, "Grocery": 0.8, "Travel": 2.5, "Dining": 0.9,
        "Electronics": 1.8, "Healthcare": 1.6, "Entertainment": 1.0, "Online Services": 1.1,
    }
    base_amounts = rng.lognormal(mean=5.2, sigma=1.1, size=n)
    amounts = np.array([
        amt * cat_amount_scale[cat]
        for amt, cat in zip(base_amounts, cat_array)
    ])

    # Transaction frequency per customer per month
    freq = rng.integers(1, 150, size=n).astype(float)

    # Dates: past 2 years
    start_ts = pd.Timestamp("2023-01-01").value // 10**9
    end_ts   = pd.Timestamp("2024-12-31").value // 10**9
    ts_vals  = rng.integers(start_ts, end_ts, size=n)
    dates    = pd.to_datetime(ts_vals, unit="s").normalize()

    # ── Risk label: high-risk driven by multiple factors ────────────────────
    # Logistic-style probability
    log_odds = (
        - 3.5
        + 0.015  * (amounts / 100)
        - 0.004  * (credits - 500)
        + 0.008  * freq
        - 0.003  * tenures
        + np.where(cat_array == "Travel",    0.6, 0)
        + np.where(cat_array == "Electronics", 0.4, 0)
        + np.where(cat_array == "Online Services", 0.3, 0)
        + np.where(ages < 25, 0.4, 0)
    )
    risk_prob = 1 / (1 + np.exp(-log_odds))
    is_high_risk = (rng.random(n) < risk_prob).astype(int)

    # ── Introduce missingness & noise for realism ───────────────────────────
    miss_idx = rng.choice(n, size=int(n * 0.02), replace=False)
    amounts_noisy = amounts.copy().astype(float)
    amounts_noisy[miss_idx[:len(miss_idx)//2]] = np.nan

    freq_noisy = freq.copy()
    freq_noisy[miss_idx[len(miss_idx)//2:]] = np.nan

    # Outlier rows (~1%)
    outlier_idx = rng.choice(n, size=int(n * 0.01), replace=False)
    amounts_noisy[outlier_idx] *= rng.uniform(10, 30, size=len(outlier_idx))

    df = pd.DataFrame({
        "customer_id":            customer_ids[cust_idx],
        "age":                    ages,
        "credit_score":           credits,
        "account_tenure_months":  tenures,
        "category":               cat_array,
        "amount":                 amounts_noisy,
        "transaction_frequency":  freq_noisy,
        "transaction_date":       dates,
        "is_high_risk":           is_high_risk,
    })

    return df
