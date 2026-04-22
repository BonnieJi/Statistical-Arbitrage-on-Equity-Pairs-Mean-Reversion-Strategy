"""Walk-forward pairs backtest with explicit train/validation/test split and costs."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from src.data_loader import fetch_adjusted_close_and_volume
from src.hedge_ratio import ols_hedge_ratio
from src.metrics import rolling_sharpe
from src.ou_model import ou_params_from_spread
from src.signals import generate_signals_for_pair


def _period_bounds(freq: str, start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Create walk-forward trading periods [period_start, period_end]."""
    normalized = "ME" if freq == "M" else ("QE" if freq == "Q" else freq)
    starts = pd.date_range(start=start, end=end, freq=normalized)
    if len(starts) == 0:
        starts = pd.DatetimeIndex([start])
    bounds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for i, s in enumerate(starts):
        e = starts[i + 1] - pd.Timedelta(days=1) if i + 1 < len(starts) else end
        bounds.append((s, min(e, end)))
    return bounds


def _cost_from_turnover(leg1_change: pd.Series, leg2_change: pd.Series) -> pd.Series:
    """Simple daily cost model: bps-per-leg + slippage + turnover penalty."""
    per_leg_total_bps = cfg.TRANSACTION_COST_BPS_PER_LEG + cfg.SLIPPAGE_BPS_PER_LEG
    per_leg_total = per_leg_total_bps / 10_000.0
    turnover_penalty = cfg.TURNOVER_PENALTY_BPS / 10_000.0
    turnover = leg1_change.abs() + leg2_change.abs()
    # Two components: linear per-leg cost and additional turnover penalty.
    return per_leg_total * turnover + turnover_penalty * turnover


def run_walk_forward_backtest_for_pair(
    symbol_1: str,
    symbol_2: str,
    strategy_params: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run monthly/quarterly walk-forward using 2018-2021 train, 2022 validation, 2023-2025 OOS test.

    Every rebalance period:
    - re-estimate hedge ratio on data up to period start
    - re-estimate OU params on calibrated spread history
    - re-test cointegration (rolling gate is already inside step3 features)
    - trade the next period with step3 signals
    """
    prices, volumes = fetch_adjusted_close_and_volume(
        symbols=[symbol_1, symbol_2],
        start_date=cfg.TRAIN_START,
        end_date=None,
    )
    panel = generate_signals_for_pair(prices, volumes, symbol_1, symbol_2, strategy_params=strategy_params)
    panel = panel.copy()
    panel["ret_1"] = panel["log_p1"].diff().fillna(0.0)
    panel["ret_2"] = panel["log_p2"].diff().fillna(0.0)

    test_start = pd.Timestamp(cfg.TEST_START)
    test_end = pd.Timestamp(cfg.TEST_END)
    wf_bounds = _period_bounds(cfg.WALK_FORWARD_REBALANCE_FREQ, test_start, test_end)

    out_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for p_start, p_end in wf_bounds:
        calib_end = p_start - pd.Timedelta(days=1)
        calib = panel.loc[:calib_end].dropna(subset=["log_p1", "log_p2", "spread"])
        trade = panel.loc[p_start:p_end].copy()
        if calib.empty or trade.empty:
            continue

        beta = ols_hedge_ratio(calib["log_p1"], calib["log_p2"])
        spread_hist = calib["log_p1"] - beta * calib["log_p2"]
        ou = ou_params_from_spread(spread_hist)

        # Use step3 position directly; only disable if calibration is invalid.
        gate = np.isfinite(beta) and np.isfinite(ou["half_life_days"])
        if not gate:
            trade["position"] = 0.0

        pos = trade["position"].fillna(0.0)
        # Spread PnL proxy in log-return space using beta-adjusted second leg.
        leg1_exp = pos
        leg2_exp = -pos * float(beta if np.isfinite(beta) else 0.0)
        gross_prev = leg1_exp.shift(1).fillna(0.0), leg2_exp.shift(1).fillna(0.0)
        trade["gross_exposure"] = gross_prev[0].abs() + gross_prev[1].abs()
        trade["pnl_gross"] = gross_prev[0] * trade["ret_1"] + gross_prev[1] * trade["ret_2"]

        c = _cost_from_turnover(leg1_exp.diff().fillna(0.0), leg2_exp.diff().fillna(0.0))
        trade["cost"] = c
        trade["pnl_net"] = trade["pnl_gross"] - trade["cost"]

        trade["wf_beta"] = float(beta)
        trade["wf_kappa"] = float(ou["kappa"])
        trade["wf_mu"] = float(ou["mu"])
        trade["wf_sigma"] = float(ou["sigma"])
        trade["wf_half_life"] = float(ou["half_life_days"])
        trade["wf_period_start"] = p_start
        trade["wf_period_end"] = p_end
        out_rows.append(trade)

        summary_rows.append(
            {
                "period_start": p_start.date().isoformat(),
                "period_end": p_end.date().isoformat(),
                "n_days": int(len(trade)),
                "eligible_entry_days": int(trade["entry_gate_all"].fillna(False).sum()),
                "executed_trades": int(trade["entry_event"].fillna(False).sum()),
                "avg_abs_z": float(trade["z_score"].abs().mean()),
                "avg_rolling_eg_pvalue": float(trade["rolling_eg_pvalue"].mean()),
                "beta": float(beta),
                "half_life": float(ou["half_life_days"]),
                "gross_return": float(trade["pnl_gross"].sum()),
                "net_return": float(trade["pnl_net"].sum()),
                "cost_paid": float(trade["cost"].sum()),
                "avg_gross_exposure": float(trade["gross_exposure"].mean()),
            }
        )

    if not out_rows:
        return pd.DataFrame(), pd.DataFrame()

    daily = pd.concat(out_rows).sort_index()
    daily["cum_gross"] = daily["pnl_gross"].cumsum()
    daily["cum_net"] = daily["pnl_net"].cumsum()
    daily["rolling_sharpe_60d"] = rolling_sharpe(daily["pnl_net"], window=60)
    summary = pd.DataFrame(summary_rows)
    return daily, summary


def run_sensitivity_analysis(symbol_1: str, symbol_2: str) -> pd.DataFrame:
    """One-factor sweeps for core signal parameters."""
    rows: list[dict] = []
    sweep_defs = [
        ("z_entry", [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]),
        ("z_exit", [0.0, 0.25, 0.5, 0.75, 1.0]),
        ("coint_p_max", [0.05, 0.10, 0.15, 0.20]),
        ("half_life_max", [45.0, 60.0, 90.0, 120.0]),
    ]
    for param_name, values in sweep_defs:
        for value in values:
            p = {param_name: value}
            daily, _ = run_walk_forward_backtest_for_pair(symbol_1, symbol_2, strategy_params=p)
            if daily.empty:
                continue
            ret = daily["pnl_net"]
            sharpe = float((ret.mean() / (ret.std() + 1e-12)) * np.sqrt(252))
            rows.append(
                {
                    "pair": f"{symbol_1}/{symbol_2}",
                    "parameter": param_name,
                    "value": float(value),
                    "net_return": float(ret.sum()),
                    "sharpe": sharpe,
                }
            )
    return pd.DataFrame(rows)


def build_equal_risk_portfolio(pair_dailies: dict[str, pd.DataFrame], max_leverage: float = 1.5) -> pd.DataFrame:
    """Combine pairs with inverse-vol risk weights and leverage cap."""
    non_empty = {k: v for k, v in pair_dailies.items() if not v.empty}
    if not non_empty:
        return pd.DataFrame()
    net = pd.concat({k: v["pnl_net"] for k, v in non_empty.items()}, axis=1).fillna(0.0)
    vol = net.rolling(20, min_periods=10).std().replace(0.0, np.nan)
    inv_vol = 1.0 / vol
    w = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0.0)
    gross_w = w.abs().sum(axis=1)
    scale = np.minimum(1.0, max_leverage / (gross_w + 1e-12))
    w_scaled = w.mul(scale, axis=0)
    port = pd.DataFrame(index=net.index)
    port["pnl_net"] = (w_scaled * net).sum(axis=1)
    port["cum_net"] = port["pnl_net"].cumsum()
    port["rolling_sharpe_60d"] = rolling_sharpe(port["pnl_net"], window=60)
    port["gross_leverage"] = w_scaled.abs().sum(axis=1)
    return port


def summarize_trades(daily: pd.DataFrame) -> pd.DataFrame:
    """Trade-by-trade table for diagnostics (entry/exit, z, holding days, gross/net pnl)."""
    if daily.empty or "entry_event" not in daily.columns:
        return pd.DataFrame()
    records: list[dict] = []
    in_trade = False
    entry_date = None
    entry_z = np.nan
    trade_id = None
    for idx, row in daily.iterrows():
        if bool(row.get("entry_event", False)) and not in_trade:
            in_trade = True
            entry_date = idx
            entry_z = float(row.get("z_score", np.nan))
            trade_id = int(row.get("trade_id", 0))
        if in_trade and bool(row.get("exit_event", False)):
            seg = daily.loc[entry_date:idx]
            records.append(
                {
                    "trade_id": trade_id,
                    "entry_date": entry_date.date().isoformat(),
                    "exit_date": idx.date().isoformat(),
                    "entry_z": entry_z,
                    "exit_z": float(row.get("z_score", np.nan)),
                    "holding_days": int(len(seg)),
                    "gross_pnl": float(seg["pnl_gross"].sum()),
                    "net_pnl": float(seg["pnl_net"].sum()),
                    "cost_paid": float(seg["cost"].sum()),
                }
            )
            in_trade = False
            entry_date = None
            entry_z = np.nan
            trade_id = None
    return pd.DataFrame(records)


def split_period_performance(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate gross/net by train, validation, and out-of-sample test windows."""
    if daily.empty:
        return pd.DataFrame()
    idx = daily.index
    train = (idx >= pd.Timestamp(cfg.TRAIN_START)) & (idx <= pd.Timestamp(cfg.TRAIN_END))
    val = (idx >= pd.Timestamp(cfg.VALIDATION_START)) & (idx <= pd.Timestamp(cfg.VALIDATION_END))
    test = (idx >= pd.Timestamp(cfg.TEST_START)) & (idx <= pd.Timestamp(cfg.TEST_END))
    out = []
    for name, mask in (("train", train), ("validation", val), ("test_oos", test)):
        seg = daily.loc[mask]
        out.append(
            {
                "split": name,
                "n_days": int(len(seg)),
                "eligible_entry_days": int(seg["entry_gate_all"].fillna(False).sum()) if not seg.empty else 0,
                "executed_trades": int(seg["entry_event"].fillna(False).sum()) if not seg.empty else 0,
                "gross_return": float(seg["pnl_gross"].sum()) if not seg.empty else 0.0,
                "net_return": float(seg["pnl_net"].sum()) if not seg.empty else 0.0,
                "cost_drag": (
                    float(seg["pnl_gross"].sum() - seg["pnl_net"].sum()) if not seg.empty else 0.0
                ),
                "cost_paid": float(seg["cost"].sum()) if not seg.empty else 0.0,
                "avg_gross_exposure": float(seg["gross_exposure"].mean()) if not seg.empty else 0.0,
            }
        )
    return pd.DataFrame(out)
