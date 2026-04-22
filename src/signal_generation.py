"""Step 3: z-score signals, continuous sizing, filters, and risk-controlled pair simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from src.spread_modeling import estimate_ou_parameters, model_pair_spread


def rolling_engle_granger_p(
    price_x: pd.Series,
    price_y: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling Engle–Granger p-value on **levels** (matches Step 1)."""
    if min_periods is None:
        min_periods = max(30, window // 2)
    df = pd.concat([price_x, price_y], axis=1).dropna()
    df.columns = ["x", "y"]
    pvals = pd.Series(np.nan, index=df.index, dtype=float)
    for i in range(window - 1, len(df)):
        w = df.iloc[i - window + 1 : i + 1]
        if len(w) < min_periods:
            continue
        try:
            _, p, _ = coint(w["x"], w["y"])
            pvals.iloc[i] = float(p)
        except Exception:
            pvals.iloc[i] = np.nan
    return pvals.reindex(price_x.index)


def rolling_half_life(
    spread: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling OU half-life (days) from trailing spread window."""
    if min_periods is None:
        min_periods = max(20, window // 2)
    s = spread.dropna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    for i in range(window - 1, len(s)):
        w = s.iloc[i - window + 1 : i + 1]
        if len(w.dropna()) < min_periods:
            continue
        hl = estimate_ou_parameters(w, dt=1.0)["half_life"]
        out.iloc[i] = hl
    return out.reindex(spread.index)


def rolling_spread_volatility(
    spread: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling std of spread level (same units as spread)."""
    if min_periods is None:
        min_periods = max(10, window // 2)
    return spread.rolling(window=window, min_periods=min_periods).std()


def rolling_mean_dollar_volume(
    prices: pd.Series,
    volume: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling mean of price * volume (approximate dollar volume)."""
    if min_periods is None:
        min_periods = max(5, window // 2)
    dv = (prices.astype(float) * volume.astype(float)).replace([np.inf, -np.inf], np.nan)
    return dv.rolling(window=window, min_periods=min_periods).mean()


@dataclass(frozen=True)
class SignalParams:
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 3.5
    max_holding_days: int = 20
    z_size_ref: float = 3.0
    target_spread_daily_vol: float = 0.01
    coint_p_max: float = 0.05
    half_life_min: float = 3.0
    half_life_max: float = 90.0
    min_dollar_volume_usd: float = 5_000_000.0
    max_capital_per_pair: float = 0.25
    max_gross_leverage: float = 2.0


def build_pair_signal_features(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    symbol_1: str,
    symbol_2: str,
    hedge_window: int,
    zscore_window: int,
    coint_window: int,
    spread_vol_window: int,
    half_life_window: int,
    liquidity_window: int,
    min_periods: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Spread features + rolling coint p, half-life, spread vol, liquidity for one pair."""
    ts, ou_full = model_pair_spread(
        prices=prices,
        symbol_1=symbol_1,
        symbol_2=symbol_2,
        hedge_window=hedge_window,
        zscore_window=zscore_window,
        min_periods=min_periods,
    )
    ix = ts.index
    spread = ts["spread"]
    rolling_vol = rolling_spread_volatility(spread, window=spread_vol_window, min_periods=min_periods)
    rolling_coint_p = rolling_engle_granger_p(
        prices[symbol_1].reindex(ix),
        prices[symbol_2].reindex(ix),
        window=coint_window,
        min_periods=min_periods,
    ).reindex(ix)
    half_life_roll = rolling_half_life(spread, window=half_life_window, min_periods=min_periods).reindex(ix)

    v1 = volumes[symbol_1].reindex(ix)
    v2 = volumes[symbol_2].reindex(ix)
    dv1 = rolling_mean_dollar_volume(
        prices[symbol_1].reindex(ix), v1, liquidity_window, min_periods
    )
    dv2 = rolling_mean_dollar_volume(
        prices[symbol_2].reindex(ix), v2, liquidity_window, min_periods
    )
    min_dv = pd.concat([dv1, dv2], axis=1).min(axis=1)

    out = ts.copy()
    out[f"price_{symbol_1}"] = prices[symbol_1].reindex(ix)
    out[f"price_{symbol_2}"] = prices[symbol_2].reindex(ix)
    out["rolling_spread_vol"] = rolling_vol
    out["rolling_coint_p"] = rolling_coint_p
    out["rolling_half_life"] = half_life_roll
    out["rolling_min_dollar_volume"] = min_dv
    return out, ou_full


def _filters_ok(
    row: pd.Series,
    p: SignalParams,
) -> bool:
    if not np.isfinite(row.get("rolling_coint_p", np.nan)) or row["rolling_coint_p"] > p.coint_p_max:
        return False
    hl = row.get("rolling_half_life", np.nan)
    if not np.isfinite(hl) or hl < p.half_life_min or hl > p.half_life_max:
        return False
    sig = row.get("rolling_spread_vol", np.nan)
    if not np.isfinite(sig) or sig <= 0.0:
        return False
    dv = row.get("rolling_min_dollar_volume", np.nan)
    if not np.isfinite(dv) or dv < p.min_dollar_volume_usd:
        return False
    return True


def _position_scale(
    z: float,
    sigma_spread: float,
    p: SignalParams,
) -> float:
    """Continuous scale: magnitude from z, volatility targeting on spread."""
    zc = float(np.clip(abs(z) / p.z_size_ref, 0.0, 1.0))
    vol = max(float(sigma_spread), 1e-12)
    vol_scale = p.target_spread_daily_vol / vol
    return float(zc * vol_scale)


def _dollar_weights_short_spread(beta: float) -> tuple[float, float]:
    """Short spread: short leg1, long leg2; gross normalized to 1 before leverage cap."""
    den = 1.0 + float(beta)
    return -1.0 / den, float(beta) / den


def _dollar_weights_long_spread(beta: float) -> tuple[float, float]:
    """Long spread: long leg1, short leg2."""
    w1, w2 = _dollar_weights_short_spread(beta)
    return -w1, -w2


def simulate_pair_signals(
    features: pd.DataFrame,
    symbol_1: str,
    symbol_2: str,
    params: SignalParams | None = None,
) -> pd.DataFrame:
    """Daily positions with entry/exit, continuous sizing, and gross leverage cap.

    Columns include ``in_position``, ``spread_side`` (+1 long spread / -1 short spread),
    ``weight_leg1`` / ``weight_leg2`` (dollar weights as fraction of unit capital after caps),
    ``held_days``.
    """
    if params is None:
        params = SignalParams()
    p1 = f"price_{symbol_1}"
    p2 = f"price_{symbol_2}"
    cols = [
        "spread_zscore",
        "hedge_ratio_beta",
        "rolling_spread_vol",
        "rolling_coint_p",
        "rolling_half_life",
        "rolling_min_dollar_volume",
        p1,
        p2,
    ]
    df = features.dropna(subset=["spread_zscore", "hedge_ratio_beta", p1, p2]).copy()
    n = len(df)
    in_pos = np.zeros(n, dtype=bool)
    side = np.zeros(n, dtype=float)
    w1 = np.zeros(n, dtype=float)
    w2 = np.zeros(n, dtype=float)
    held = np.zeros(n, dtype=int)
    entry_idx = -1
    entry_side = 0.0

    z_arr = df["spread_zscore"].to_numpy()
    beta_arr = df["hedge_ratio_beta"].to_numpy()
    sig_arr = df["rolling_spread_vol"].to_numpy()

    for i in range(n):
        row = df.iloc[i]
        z = float(z_arr[i])
        beta = float(beta_arr[i])
        sig = float(sig_arr[i]) if np.isfinite(sig_arr[i]) else np.nan

        if entry_idx < 0:
            if (
                np.isfinite(z)
                and abs(z) > params.z_entry
                and _filters_ok(row, params)
            ):
                sc = _position_scale(z, sig if np.isfinite(sig) else 1.0, params)
                sc = min(sc, params.max_capital_per_pair)
                entry_side = -1.0 if z > params.z_entry else 1.0
                if entry_side < 0:
                    dw1, dw2 = _dollar_weights_short_spread(beta)
                else:
                    dw1, dw2 = _dollar_weights_long_spread(beta)
                gross0 = abs(dw1) + abs(dw2)
                lev = min(1.0, params.max_gross_leverage / max(gross0, 1e-12))
                w1[i] = dw1 * sc * lev
                w2[i] = dw2 * sc * lev
                in_pos[i] = True
                side[i] = entry_side
                held[i] = 1
                entry_idx = i
            continue

        hd = i - entry_idx + 1
        exit_now = (
            not np.isfinite(z)
            or abs(z) < params.z_exit
            or abs(z) > params.z_stop
            or hd >= params.max_holding_days
        )
        if exit_now:
            entry_idx = -1
            entry_side = 0.0
            continue

        sc = _position_scale(z, sig if np.isfinite(sig) else 1.0, params)
        sc = min(sc, params.max_capital_per_pair)
        if entry_side < 0:
            dw1, dw2 = _dollar_weights_short_spread(beta)
        else:
            dw1, dw2 = _dollar_weights_long_spread(beta)
        gross0 = abs(dw1) + abs(dw2)
        lev = min(1.0, params.max_gross_leverage / max(gross0, 1e-12))
        w1[i] = dw1 * sc * lev
        w2[i] = dw2 * sc * lev
        in_pos[i] = True
        side[i] = entry_side
        held[i] = hd

    out = df[cols].copy()
    out["in_position"] = in_pos
    out["spread_side"] = side
    out["weight_leg1"] = w1
    out["weight_leg2"] = w2
    out["held_days"] = held
    out["gross_exposure"] = np.abs(w1) + np.abs(w2)
    return out


def run_step3_for_pairs(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    pairs: Iterable[tuple[str, str]],
    hedge_window: int,
    zscore_window: int,
    coint_window: int,
    spread_vol_window: int,
    half_life_window: int,
    liquidity_window: int,
    params: SignalParams | None = None,
    min_periods: int | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Build features and simulated weights for each pair."""
    if params is None:
        params = SignalParams()
    features_map: dict[str, pd.DataFrame] = {}
    sim_map: dict[str, pd.DataFrame] = {}
    for s1, s2 in pairs:
        key = f"{s1}/{s2}"
        feat, _ = build_pair_signal_features(
            prices=prices,
            volumes=volumes,
            symbol_1=s1,
            symbol_2=s2,
            hedge_window=hedge_window,
            zscore_window=zscore_window,
            coint_window=coint_window,
            spread_vol_window=spread_vol_window,
            half_life_window=half_life_window,
            liquidity_window=liquidity_window,
            min_periods=min_periods,
        )
        features_map[key] = feat
        sim_map[key] = simulate_pair_signals(feat, s1, s2, params=params)
    return features_map, sim_map
