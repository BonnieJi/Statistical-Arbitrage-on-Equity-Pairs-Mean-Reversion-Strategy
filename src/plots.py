"""Plotting utilities for case studies and cross-pair comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, out_dir: Path, filename: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_case_study(
    daily: pd.DataFrame,
    pair_name: str,
    out_dir: Path,
    filename_prefix: str,
) -> list[str]:
    """Create spread, z-score, entry/exit, and equity plots for one pair."""
    paths: list[str] = []
    if daily.empty:
        return paths

    # Spread plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily["spread"], label="Spread", color="steelblue")
    ax.set_title(f"{pair_name} - Spread")
    ax.set_ylabel("Spread")
    ax.grid(alpha=0.25)
    ax.legend()
    paths.append(_save(fig, out_dir, f"{filename_prefix}_spread.png"))

    # Z-score plot with thresholds
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily["z_score"], label="Z-score", color="darkorange")
    ax.axhline(1.5, ls="--", lw=1.0, color="red", alpha=0.8, label="Entry +/-1.5")
    ax.axhline(-1.5, ls="--", lw=1.0, color="red", alpha=0.8)
    ax.axhline(0.5, ls=":", lw=1.0, color="gray", alpha=0.8, label="Exit +/-0.5")
    ax.axhline(-0.5, ls=":", lw=1.0, color="gray", alpha=0.8)
    ax.axhline(0.0, lw=0.8, color="black", alpha=0.6)
    ax.set_title(f"{pair_name} - Z-score")
    ax.set_ylabel("Z")
    ax.grid(alpha=0.25)
    ax.legend()
    paths.append(_save(fig, out_dir, f"{filename_prefix}_zscore.png"))

    # Entry/exit visualization on spread
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily["spread"], color="royalblue", label="Spread")
    entry = daily["entry_event"].fillna(False)
    exit_ = daily["exit_event"].fillna(False)
    ax.scatter(daily.index[entry], daily.loc[entry, "spread"], marker="^", color="green", s=35, label="Entry")
    ax.scatter(daily.index[exit_], daily.loc[exit_, "spread"], marker="v", color="crimson", s=35, label="Exit")
    ax.set_title(f"{pair_name} - Trade Entries/Exits")
    ax.set_ylabel("Spread")
    ax.grid(alpha=0.25)
    ax.legend()
    paths.append(_save(fig, out_dir, f"{filename_prefix}_entries_exits.png"))

    # Equity + drawdown curve
    drawdown = daily["cum_net"] - daily["cum_net"].cummax()
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(daily.index, daily["cum_net"], color="seagreen", label="Cumulative net")
    axes[0].plot(daily.index, daily["cum_gross"], color="slategray", alpha=0.8, label="Cumulative gross")
    axes[0].set_title(f"{pair_name} - Equity Curve")
    axes[0].set_ylabel("PnL (cumulative)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].fill_between(daily.index, drawdown, 0.0, color="firebrick", alpha=0.35)
    axes[1].plot(daily.index, drawdown, color="firebrick", lw=1.0, label="Drawdown")
    axes[1].set_title("Drawdown Curve")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    paths.append(_save(fig, out_dir, f"{filename_prefix}_equity_drawdown.png"))

    # Rolling Sharpe curve.
    if "rolling_sharpe_60d" in daily.columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(daily.index, daily["rolling_sharpe_60d"], color="purple")
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.6)
        ax.set_title(f"{pair_name} - Rolling Sharpe (60d)")
        ax.set_ylabel("Sharpe")
        ax.grid(alpha=0.25)
        paths.append(_save(fig, out_dir, f"{filename_prefix}_rolling_sharpe_60d.png"))

    return paths


def plot_pair_comparisons(
    pair_daily: dict[str, pd.DataFrame],
    out_dir: Path,
) -> list[str]:
    """Create cumulative net by pair and gross-vs-net by pair."""
    paths: list[str] = []
    non_empty = {k: v for k, v in pair_daily.items() if not v.empty}
    if not non_empty:
        return paths

    # Cumulative net return by pair.
    fig, ax = plt.subplots(figsize=(12, 5))
    for pair_name, daily in non_empty.items():
        ax.plot(daily.index, daily["cum_net"], label=pair_name)
    ax.set_title("Cumulative Net Return by Pair")
    ax.set_ylabel("Cumulative net PnL")
    ax.grid(alpha=0.25)
    ax.legend()
    paths.append(_save(fig, out_dir, "comparison_cumulative_net_by_pair.png"))

    # Gross vs net by pair (bar chart using OOS totals from each daily frame).
    pair_names = list(non_empty.keys())
    gross_vals = [float(df["pnl_gross"].sum()) for df in non_empty.values()]
    net_vals = [float(df["pnl_net"].sum()) for df in non_empty.values()]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(pair_names))
    ax.bar([i - 0.18 for i in x], gross_vals, width=0.36, label="Gross")
    ax.bar([i + 0.18 for i in x], net_vals, width=0.36, label="Net")
    ax.set_xticks(list(x))
    ax.set_xticklabels(pair_names)
    ax.set_title("Gross vs Net Return by Pair")
    ax.set_ylabel("Total return")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    paths.append(_save(fig, out_dir, "comparison_gross_vs_net_by_pair.png"))

    return paths


def plot_trade_distribution(trades: pd.DataFrame, pair_name: str, out_dir: Path, filename_prefix: str) -> list[str]:
    paths: list[str] = []
    if trades.empty:
        return paths
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(trades["net_pnl"], bins=min(20, max(8, len(trades))), color="teal", alpha=0.8)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_title(f"{pair_name} - Trade PnL Distribution (Net)")
    ax.set_xlabel("Net PnL per trade")
    ax.grid(alpha=0.25, axis="y")
    paths.append(_save(fig, out_dir, f"{filename_prefix}_trade_pnl_hist.png"))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(trades["holding_days"], bins=np.arange(1, trades["holding_days"].max() + 2) - 0.5, color="darkcyan", alpha=0.8)
    ax.set_title(f"{pair_name} - Holding Period Distribution")
    ax.set_xlabel("Holding days")
    ax.grid(alpha=0.25, axis="y")
    paths.append(_save(fig, out_dir, f"{filename_prefix}_holding_period_hist.png"))
    return paths


def plot_sensitivity_curves(sensitivity: pd.DataFrame, out_dir: Path, filename_prefix: str) -> list[str]:
    paths: list[str] = []
    if sensitivity.empty:
        return paths
    for metric in ("sharpe", "net_return"):
        fig, ax = plt.subplots(figsize=(11, 4))
        for p_name, grp in sensitivity.groupby("parameter"):
            g = grp.sort_values("value")
            ax.plot(g["value"], g[metric], marker="o", label=p_name)
        ax.set_title(f"Sensitivity Analysis - {metric.replace('_', ' ').title()}")
        ax.set_xlabel("Parameter value")
        ax.grid(alpha=0.25)
        ax.legend()
        paths.append(_save(fig, out_dir, f"{filename_prefix}_sensitivity_{metric}.png"))
    return paths


def plot_portfolio_equity(portfolio: pd.DataFrame, out_dir: Path, filename_prefix: str) -> list[str]:
    paths: list[str] = []
    if portfolio.empty:
        return paths
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(portfolio.index, portfolio["cum_net"], color="navy", label="Portfolio cumulative net")
    ax.set_title("Portfolio Equity Curve")
    ax.set_ylabel("Cumulative net PnL")
    ax.grid(alpha=0.25)
    ax.legend()
    paths.append(_save(fig, out_dir, f"{filename_prefix}_portfolio_equity_curve.png"))
    return paths
