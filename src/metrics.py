"""Risk and trade-distribution metrics for backtest diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_drawdown(cum_pnl: pd.Series) -> pd.Series:
    peak = cum_pnl.cummax()
    return cum_pnl - peak


def max_drawdown(cum_pnl: pd.Series) -> float:
    if cum_pnl.empty:
        return 0.0
    return float(compute_drawdown(cum_pnl).min())


def rolling_sharpe(pnl: pd.Series, window: int = 60, annualization: int = 252) -> pd.Series:
    mu = pnl.rolling(window, min_periods=max(10, window // 3)).mean()
    sd = pnl.rolling(window, min_periods=max(10, window // 3)).std()
    return (mu / (sd + 1e-12)) * np.sqrt(annualization)


def summarize_trade_distribution(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "n_trades": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_holding_days": 0.0,
        }
    pnl = trades["net_pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    return {
        "n_trades": float(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "avg_win": float(wins.mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses.mean()) if not losses.empty else 0.0,
        "avg_holding_days": float(trades["holding_days"].mean()),
    }
