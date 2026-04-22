"""Plotting utilities for case studies and cross-pair comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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

    # Equity curve
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily["cum_net"], color="seagreen", label="Cumulative net")
    ax.plot(daily.index, daily["cum_gross"], color="slategray", alpha=0.8, label="Cumulative gross")
    ax.set_title(f"{pair_name} - Equity Curve")
    ax.set_ylabel("PnL (cumulative)")
    ax.grid(alpha=0.25)
    ax.legend()
    paths.append(_save(fig, out_dir, f"{filename_prefix}_equity_curve.png"))

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
