"""OLS hedge ratio on log prices: log(P1) = alpha + beta * log(P2) + epsilon."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ols_hedge_ratio(log_p1: pd.Series, log_p2: pd.Series) -> float:
    """Full-sample OLS slope beta in log_p1 ~ log_p2 (with intercept)."""
    df = pd.concat([log_p1.rename("y"), log_p2.rename("x")], axis=1).dropna()
    if len(df) < 10:
        return float("nan")
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    xc = x - x.mean()
    yc = y - y.mean()
    den = float(np.dot(xc, xc))
    if den < 1e-14:
        return float("nan")
    return float(np.dot(xc, yc) / den)


def rolling_hedge_ratio(
    log_p1: pd.Series,
    log_p2: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling OLS beta using only past data: beta[t] from log prices on [t-window, t-1]."""
    if window < 5:
        raise ValueError("window must be at least 5 for rolling OLS.")
    idx = log_p1.index
    a1 = log_p1.to_numpy(dtype=float)
    a2 = log_p2.to_numpy(dtype=float)
    n = len(a1)
    out = np.full(n, np.nan)
    for i in range(window, n):
        sl1 = a1[i - window : i]
        sl2 = a2[i - window : i]
        m1, m2 = np.nanmean(sl1), np.nanmean(sl2)
        xc = sl2 - m2
        yc = sl1 - m1
        den = float(np.dot(xc, xc))
        if den < 1e-14:
            out[i] = np.nan
        else:
            out[i] = float(np.dot(xc, yc) / den)
    return pd.Series(out, index=idx, name="hedge_ratio_roll")
