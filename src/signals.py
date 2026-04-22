"""Step 3: spread features, filters, and z-score signals with continuous sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

import config as cfg
from src.hedge_ratio import ols_hedge_ratio, rolling_hedge_ratio
from src.ou_model import rolling_ou_params


def rolling_engle_granger_pvalue(
    log_p1: pd.Series,
    log_p2: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling Engle–Granger p-value on log prices using past-only window [t-window, t-1]."""
    idx = log_p1.index
    a1 = log_p1.to_numpy(dtype=float)
    a2 = log_p2.to_numpy(dtype=float)
    n = len(a1)
    out = np.full(n, np.nan)
    for i in range(window, n):
        s1 = a1[i - window : i]
        s2 = a2[i - window : i]
        if np.any(~np.isfinite(s1)) or np.any(~np.isfinite(s2)):
            out[i] = np.nan
            continue
        try:
            _, pval, _ = coint(s1, s2)
            out[i] = float(pval)
        except Exception:  # noqa: BLE001
            out[i] = 1.0
    return pd.Series(out, index=idx, name="rolling_eg_pvalue")


def rolling_slope_tstat(series: pd.Series, window: int) -> pd.Series:
    """|t-stat| on OLS slope of spread vs time index over past ``window`` days (trend strength)."""
    yv = series.to_numpy(dtype=float)
    n = len(yv)
    out = np.full(n, np.nan)
    t_grid = np.arange(window, dtype=float)
    t_c = t_grid - t_grid.mean()
    den_t = float(np.dot(t_c, t_c))
    for i in range(window, n):
        seg = yv[i - window : i]
        if not np.all(np.isfinite(seg)) or den_t < 1e-14:
            continue
        seg_c = seg - float(np.nanmean(seg))
        slope = float(np.dot(t_c, seg_c) / den_t)
        resid = seg_c - slope * t_c
        if window > 2:
            mse = float(np.dot(resid, resid)) / (window - 2)
            se = float(np.sqrt(mse / den_t)) if mse >= 0 else np.nan
            out[i] = slope / se if se and se > 1e-12 else np.nan
    return pd.Series(out, index=series.index, name="spread_slope_tstat")


def build_pair_feature_panel(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    symbol_1: str,
    symbol_2: str,
) -> pd.DataFrame:
    """Per-day features: logs, rolling hedge, spread, z, OU, filters (no positions)."""
    px = prices[[symbol_1, symbol_2]].sort_index().astype(float)
    vol = volumes.reindex(px.index)[[symbol_1, symbol_2]].astype(float)
    df = pd.DataFrame(index=px.index)
    df["log_p1"] = np.log(px[symbol_1])
    df["log_p2"] = np.log(px[symbol_2])

    df["hedge_ratio_ols_full_sample"] = ols_hedge_ratio(df["log_p1"], df["log_p2"])
    df["hedge_ratio_roll"] = rolling_hedge_ratio(
        df["log_p1"], df["log_p2"], cfg.DEFAULT_HEDGE_WINDOW
    )
    df["spread"] = df["log_p1"] - df["hedge_ratio_roll"] * df["log_p2"]

    mu_z = df["spread"].shift(1).rolling(cfg.DEFAULT_ZSCORE_WINDOW, min_periods=10).mean()
    sd_z = df["spread"].shift(1).rolling(cfg.DEFAULT_ZSCORE_WINDOW, min_periods=10).std()
    df["z_score"] = (df["spread"] - mu_z) / (sd_z + 1e-12)

    df["spread_vol_roll"] = (
        df["spread"].diff().rolling(cfg.DEFAULT_SPREAD_VOL_WINDOW, min_periods=10).std()
    )

    ou = rolling_ou_params(df["spread"], cfg.DEFAULT_HALF_LIFE_WINDOW)
    df = df.join(ou)

    df["rolling_eg_pvalue"] = rolling_engle_granger_pvalue(
        df["log_p1"], df["log_p2"], cfg.DEFAULT_COINT_ROLLING_WINDOW
    )

    r1 = df["log_p1"].diff()
    r2 = df["log_p2"].diff()
    df["leg1_realized_vol"] = r1.rolling(cfg.DEFAULT_SPREAD_VOL_WINDOW, min_periods=10).std()
    df["leg2_realized_vol"] = r2.rolling(cfg.DEFAULT_SPREAD_VOL_WINDOW, min_periods=10).std()

    dv1 = (px[symbol_1] * vol[symbol_1]).rolling(cfg.DEFAULT_LIQUIDITY_WINDOW, min_periods=5).mean()
    dv2 = (px[symbol_2] * vol[symbol_2]).rolling(cfg.DEFAULT_LIQUIDITY_WINDOW, min_periods=5).mean()
    df["dollar_volume_leg1"] = dv1
    df["dollar_volume_leg2"] = dv2
    df["dollar_volume_min"] = pd.concat([dv1, dv2], axis=1).min(axis=1)

    roll_corr = r1.rolling(cfg.CORR_ROLLING_WINDOW, min_periods=5).corr(r2)
    df["corr_instability"] = roll_corr.rolling(
        cfg.CORR_INSTABILITY_ROLLING, min_periods=10
    ).std(ddof=1)

    df["spread_slope_tstat"] = rolling_slope_tstat(df["spread"], cfg.SPREAD_SLOPE_WINDOW)

    # --- Gates used at entry (subset of features; rest available for research) ---
    df["gate_coint_stable"] = df["rolling_eg_pvalue"] < cfg.DEFAULT_COINT_P_MAX
    hl = df["half_life_roll"]
    df["gate_half_life"] = (hl >= cfg.DEFAULT_HALF_LIFE_MIN_DAYS) & (
        hl <= cfg.DEFAULT_HALF_LIFE_MAX_DAYS
    )
    df["gate_liquidity"] = df["dollar_volume_min"] >= cfg.DEFAULT_MIN_DOLLAR_VOLUME_USD
    ci = df["corr_instability"]
    df["gate_corr_stable"] = ci <= cfg.MAX_CORR_INSTABILITY_STD
    st = df["spread_slope_tstat"]
    df["gate_spread_not_trending"] = st.abs() <= cfg.SPREAD_SLOPE_MAX_ABS_TSTAT

    df["entry_gate_all"] = (
        df["gate_coint_stable"].fillna(False)
        & df["gate_half_life"].fillna(False)
        & df["gate_liquidity"].fillna(False)
        & df["gate_corr_stable"].fillna(False)
        & df["gate_spread_not_trending"].fillna(False)
    )

    df.attrs["symbol_1"] = symbol_1
    df.attrs["symbol_2"] = symbol_2
    return df


def simulate_positions(df: pd.DataFrame) -> pd.DataFrame:
    """State machine: entry |z|>2 with gates, exit |z|<0.5 or stop |z|>3.5 or max hold.

    Continuous sizing: min(|z|/Z_SIZE_REF, 1) * TARGET_SPREAD_DAILY_VOL / spread_vol,
    capped by max capital per pair and max gross leverage (single-pair path).
    """
    out = df.copy()
    z = out["z_score"].to_numpy(dtype=float)
    entry_ok = out["entry_gate_all"].to_numpy()
    vol_spread = out["spread_vol_roll"].to_numpy(dtype=float)
    n = len(out)
    pos = np.zeros(n, dtype=float)
    flat = True
    direction = 0  # +1 long spread, -1 short spread
    entry_idx = -1

    for i in range(n):
        z_i = z[i]
        if flat:
            if entry_ok[i] and np.isfinite(z_i):
                if z_i > cfg.Z_ENTRY:
                    flat = False
                    direction = -1
                    entry_idx = i
                elif z_i < -cfg.Z_ENTRY:
                    flat = False
                    direction = 1
                    entry_idx = i
        else:
            exit_now = False
            if not np.isfinite(z_i):
                exit_now = True
            elif abs(z_i) < cfg.Z_EXIT:
                exit_now = True
            elif abs(z_i) > cfg.Z_STOP:
                exit_now = True
            elif entry_idx >= 0 and (i - entry_idx + 1) >= cfg.MAX_HOLDING_DAYS:
                exit_now = True
            if exit_now:
                flat = True
                direction = 0
                entry_idx = -1

        if flat:
            pos[i] = 0.0
        else:
            mag = min(abs(z_i) / cfg.Z_SIZE_REF, 1.0)
            v = vol_spread[i] if np.isfinite(vol_spread[i]) else np.nan
            if np.isfinite(v) and v > 0:
                vol_scale = min(cfg.TARGET_SPREAD_DAILY_VOL / max(v, 1e-8), 50.0)
            else:
                vol_scale = 0.0
            raw = float(direction) * mag * vol_scale
            cap = min(cfg.DEFAULT_MAX_CAPITAL_PER_PAIR, cfg.DEFAULT_MAX_GROSS_LEVERAGE)
            raw = float(np.clip(raw, -cap, cap))
            pos[i] = raw

    out["position"] = pos
    out["position_gross"] = np.abs(pos)
    return out


def generate_signals_for_pair(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    symbol_1: str,
    symbol_2: str,
) -> pd.DataFrame:
    """Full Step 3 pipeline for one pair: features + positions."""
    feats = build_pair_feature_panel(prices, volumes, symbol_1, symbol_2)
    return simulate_positions(feats)
