"""Ornstein–Uhlenbeck-style parameters from spread via AR(1) discretization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ou_params_from_spread(spread: pd.Series) -> dict[str, float]:
    """Fit S_{t+1} = a + b S_t + eps; map to rough OU kappa, mu, sigma (daily dt=1).

    Half-life (days) = ln(2) / kappa when 0 < b < 1 and kappa = -ln(b).
    """
    s = spread.dropna().astype(float)
    if len(s) < 10:
        return {
            "kappa": float("nan"),
            "mu": float("nan"),
            "sigma": float("nan"),
            "half_life_days": float("nan"),
            "ar1_b": float("nan"),
        }
    s0 = s.iloc[:-1].to_numpy()
    s1 = s.iloc[1:].to_numpy()
    m0, m1 = s0.mean(), s1.mean()
    x = s0 - m0
    y = s1 - m1
    den = float(np.dot(x, x))
    if den < 1e-14:
        b = float("nan")
    else:
        b = float(np.dot(x, y) / den)
    a = float(m1 - b * m0)
    resid = s1 - (a + b * s0)
    sigma = float(resid.std(ddof=1)) if len(resid) > 1 else float("nan")
    if not np.isfinite(b) or b <= 0 or b >= 1:
        kappa = float("nan")
        half_life = float("nan")
    else:
        kappa = float(-np.log(b))
        half_life = float(np.log(2) / kappa) if kappa > 1e-8 else float("nan")
    mu = float(a / (1 - b)) if np.isfinite(b) and abs(1 - b) > 1e-8 else float("nan")
    return {
        "kappa": kappa,
        "mu": mu,
        "sigma": sigma,
        "half_life_days": half_life,
        "ar1_b": b,
    }


def rolling_ou_half_life(
    spread: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling half-life (days) from AR(1) on spread in each past-only window."""
    idx = spread.index
    vals = spread.to_numpy(dtype=float)
    n = len(vals)
    out = np.full(n, np.nan)
    for i in range(window, n):
        seg = vals[i - window : i]
        if np.any(~np.isfinite(seg)):
            continue
        s = pd.Series(seg)
        out[i] = ou_params_from_spread(s)["half_life_days"]
    return pd.Series(out, index=idx, name="half_life_roll")


def rolling_ou_params(spread: pd.Series, window: int) -> pd.DataFrame:
    """Rolling AR(1) / OU-style parameters on past spread windows [t-window, t-1]."""
    idx = spread.index
    vals = spread.to_numpy(dtype=float)
    n = len(vals)
    kappa = np.full(n, np.nan)
    mu = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    half_life = np.full(n, np.nan)
    for i in range(window, n):
        seg = vals[i - window : i]
        if np.any(~np.isfinite(seg)):
            continue
        p = ou_params_from_spread(pd.Series(seg))
        kappa[i] = p["kappa"]
        mu[i] = p["mu"]
        sigma[i] = p["sigma"]
        half_life[i] = p["half_life_days"]
    return pd.DataFrame(
        {"kappa_roll": kappa, "mu_roll": mu, "sigma_roll": sigma, "half_life_roll": half_life},
        index=idx,
    )
