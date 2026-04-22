"""Spread construction, rolling z-scores, and Ornstein–Uhlenbeck parameter estimates."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def log_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Natural log of strictly positive adjusted closes."""
    out = np.log(prices.astype(float))
    return out.replace([np.inf, -np.inf], np.nan)


def rolling_hedge_ratio_ols(
    log_p1: pd.Series,
    log_p2: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling OLS slope: log(P1) ~ alpha + beta * log(P2).

    beta_t is the hedge ratio at time t using the trailing ``window`` observations.
    """
    if min_periods is None:
        min_periods = max(10, window // 2)
    x = log_p2
    y = log_p1
    cov_xy = x.rolling(window=window, min_periods=min_periods).cov(y)
    var_x = x.rolling(window=window, min_periods=min_periods).var()
    beta = cov_xy / var_x.replace(0.0, np.nan)
    return beta


def spread_from_hedge(
    log_p1: pd.Series,
    log_p2: pd.Series,
    beta: pd.Series,
) -> pd.Series:
    """spread_t = log(P1_t) - beta_t * log(P2_t)."""
    return log_p1 - beta * log_p2


def rolling_zscore(
    spread: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling (spread - mean) / std of spread."""
    if min_periods is None:
        min_periods = max(10, window // 2)
    m = spread.rolling(window=window, min_periods=min_periods).mean()
    s = spread.rolling(window=window, min_periods=min_periods).std()
    return (spread - m) / s.replace(0.0, np.nan)


def estimate_ou_parameters(
    x: pd.Series | np.ndarray,
    dt: float = 1.0,
) -> dict[str, float]:
    """Estimate OU parameters from a discrete sample using exact AR(1) mapping.

    Model (uniform step ``dt``): dX_t = kappa * (mu - X_t) * dt + sigma * dW_t
    Exact discretization: X_{t+1} = mu * (1 - b) + b * X_t + eta, with b = exp(-kappa * dt).

    OLS: X_{t+1} = a + b * X_t + eps  =>  mu = a / (1 - b), kappa = -log(b) / dt,
    sigma from innovation variance: sigma^2 = Var(eps) * 2 * kappa / (1 - b^2).
    """
    arr = np.asarray(x, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    if len(arr) < 30:
        return {
            "kappa": float("nan"),
            "mu": float("nan"),
            "sigma": float("nan"),
            "half_life": float("nan"),
            "ar1_b": float("nan"),
        }

    x0 = arr[:-1]
    x1 = arr[1:]
    # OLS: x1 = a + b * x0
    b, a = np.polyfit(x0, x1, 1)
    if not np.isfinite(b) or b <= 0.0 or b >= 1.0:
        return {
            "kappa": float("nan"),
            "mu": float("nan"),
            "sigma": float("nan"),
            "half_life": float("nan"),
            "ar1_b": float(b),
        }

    kappa = -float(np.log(b)) / dt
    mu = float(a / (1.0 - b))
    resid = x1 - (a + b * x0)
    var_eps = float(np.var(resid, ddof=1))
    sigma_sq = var_eps * 2.0 * kappa / (1.0 - b * b)
    sigma = float(np.sqrt(max(sigma_sq, 0.0)))
    half_life = float(np.log(2.0) / kappa) if kappa > 0 else float("nan")

    return {
        "kappa": kappa,
        "mu": mu,
        "sigma": sigma,
        "half_life": half_life,
        "ar1_b": float(b),
    }


def model_pair_spread(
    prices: pd.DataFrame,
    symbol_1: str,
    symbol_2: str,
    hedge_window: int,
    zscore_window: int | None = None,
    min_periods: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build log prices, rolling hedge ratio, spread, z-score, and OU parameters for one pair.

    Returns the time-series DataFrame (aligned to ``prices`` index) and OU scalars estimated
    on the spread after dropping NaNs.
    """
    if zscore_window is None:
        zscore_window = hedge_window
    pair = prices[[symbol_1, symbol_2]].dropna()
    lp = log_prices(pair)
    log_p1 = lp[symbol_1]
    log_p2 = lp[symbol_2]

    beta = rolling_hedge_ratio_ols(log_p1, log_p2, window=hedge_window, min_periods=min_periods)
    spread = spread_from_hedge(log_p1, log_p2, beta)
    z_spread = rolling_zscore(spread, window=zscore_window, min_periods=min_periods)

    ou = estimate_ou_parameters(spread.dropna(), dt=1.0)

    ts = pd.DataFrame(
        {
            f"log_{symbol_1}": log_p1,
            f"log_{symbol_2}": log_p2,
            "hedge_ratio_beta": beta,
            "spread": spread,
            "spread_zscore": z_spread,
        },
        index=pair.index,
    )
    return ts, ou


def model_all_pairs(
    prices: pd.DataFrame,
    pairs: Iterable[tuple[str, str]],
    hedge_window: int,
    zscore_window: int | None = None,
    min_periods: int | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Run :func:`model_pair_spread` for each pair; return dict of series and summary table."""
    series_by_pair: dict[str, pd.DataFrame] = {}
    ou_rows: list[dict[str, float | str]] = []

    for s1, s2 in pairs:
        key = f"{s1}/{s2}"
        ts, ou = model_pair_spread(
            prices=prices,
            symbol_1=s1,
            symbol_2=s2,
            hedge_window=hedge_window,
            zscore_window=zscore_window,
            min_periods=min_periods,
        )
        series_by_pair[key] = ts
        ou_rows.append({"pair": key, **ou})

    summary = pd.DataFrame(ou_rows)
    return series_by_pair, summary
