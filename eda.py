"""
eda.py
Exploratory Data Analysis chart functions — all dark-themed, consistent style.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np


# ── Shared style helpers ──────────────────────────────────────────────────────
BG       = "#0d1117"
SURFACE  = "#161b22"
BORDER   = "#30363d"
ACCENT   = "#58a6ff"
GREEN    = "#3fb950"
RED      = "#f85149"
MUTED    = "#8b949e"
TEXT     = "#e6edf3"


def _apply_dark(ax, fig):
    """Apply consistent dark styling to any axes."""
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)


# ─────────────────────────────────────────────────────────────────────────────
# Risk Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_risk_dist(df: pd.DataFrame) -> plt.Figure:
    """Donut chart + bar chart side-by-side for risk distribution."""
    counts = df["is_high_risk"].value_counts().sort_index()
    labels = ["Low Risk", "High Risk"]
    colors = [GREEN, RED]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), facecolor=BG)

    # Donut
    wedges, texts, autotexts = ax1.pie(
        counts, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90,
        wedgeprops={"width": 0.5, "edgecolor": BG, "linewidth": 2},
    )
    for t in texts:    t.set_color(MUTED); t.set_fontsize(8)
    for at in autotexts: at.set_color(TEXT); at.set_fontweight("bold"); at.set_fontsize(9)
    ax1.set_facecolor(BG)
    ax1.set_title("Risk Split", color=TEXT, fontsize=10, pad=8)

    # Bar
    bars = ax2.bar(labels, counts.values, color=colors, width=0.45, edgecolor=BG)
    for bar, val in zip(bars, counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f"{val:,}", ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")
    _apply_dark(ax2, fig)
    ax2.set_ylabel("Count")
    ax2.set_title("Transaction Count", color=TEXT, fontsize=10, pad=8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.tight_layout(pad=1.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Transaction Trend Over Time
# ─────────────────────────────────────────────────────────────────────────────

def plot_transaction_trend(df: pd.DataFrame) -> plt.Figure:
    """Monthly transaction volume stacked by risk level."""
    trend = (
        df.groupby(["year_month", "is_high_risk"])["amount"]
        .sum()
        .unstack(fill_value=0)
        .rename(columns={0: "Low Risk", 1: "High Risk"})
    )

    fig, ax = plt.subplots(figsize=(12, 3.5), facecolor=BG)
    trend["Low Risk"].plot(ax=ax, color=GREEN, lw=2, label="Low Risk")
    trend["High Risk"].plot(ax=ax, color=RED, lw=2, linestyle="--", label="High Risk")
    ax.fill_between(range(len(trend)), trend["Low Risk"].values, alpha=0.1, color=GREEN)
    ax.fill_between(range(len(trend)), trend["High Risk"].values, alpha=0.1, color=RED)
    ax.set_xticks(range(len(trend)))
    ax.set_xticklabels(trend.index.tolist(), rotation=45, ha="right", fontsize=7)
    _apply_dark(ax, fig)
    ax.set_title("Monthly Transaction Volume by Risk Level", color=TEXT, fontsize=11, pad=10)
    ax.set_ylabel("Total Amount ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    ax.legend(facecolor=SURFACE, labelcolor=TEXT, fontsize=9, framealpha=0.8)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation(df: pd.DataFrame) -> plt.Figure:
    """Pearson correlation heatmap of numeric features."""
    num_cols = [
        "age", "amount", "credit_score",
        "account_tenure_months", "transaction_frequency", "is_high_risk",
    ]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
    ax.set_facecolor(BG)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 8, "color": TEXT},
        linecolor=BG, linewidths=0.5,
        cbar_kws={"shrink": 0.75},
    )
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title("Feature Correlation Matrix", color=TEXT, fontsize=11, pad=10)
    ax.figure.axes[-1].yaxis.label.set_color(MUTED)  # colorbar
    ax.figure.axes[-1].tick_params(colors=MUTED)
    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Top Categories
# ─────────────────────────────────────────────────────────────────────────────

def plot_top_categories(df: pd.DataFrame) -> plt.Figure:
    """Grouped bar: average transaction amount + risk rate per category."""
    grp = (
        df.groupby("category")
        .agg(avg_amount=("amount", "mean"), risk_rate=("is_high_risk", "mean"))
        .sort_values("avg_amount", ascending=False)
    )

    fig, ax1 = plt.subplots(figsize=(7, 3.8), facecolor=BG)
    ax2 = ax1.twinx()

    x = range(len(grp))
    ax1.bar(x, grp["avg_amount"], color=ACCENT, alpha=0.75, width=0.55, label="Avg Amount")
    ax2.plot(x, grp["risk_rate"], color=RED, marker="o", ms=5, lw=2, label="Risk Rate")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(grp.index.tolist(), rotation=30, ha="right", fontsize=8)
    _apply_dark(ax1, fig)
    ax2.set_facecolor(SURFACE)
    ax2.tick_params(colors=MUTED, labelsize=8)
    ax2.yaxis.label.set_color(RED)
    ax2.spines["right"].set_edgecolor(BORDER)

    ax1.set_ylabel("Avg Amount ($)", color=ACCENT)
    ax2.set_ylabel("Risk Rate", color=RED)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax1.set_title("Category: Avg Amount vs Risk Rate", color=TEXT, fontsize=11, pad=10)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               facecolor=SURFACE, labelcolor=TEXT, fontsize=8, loc="upper right")

    fig.tight_layout()
    return fig
