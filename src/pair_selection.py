"""Pair selection via Engle-Granger and Johansen cointegration tests on log prices."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def test_pair_cointegration(
    prices: pd.DataFrame,
    symbol_x: str,
    symbol_y: str,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> dict:
    """Run Engle-Granger and Johansen tests on log prices for a single pair."""
    pair_df = prices[[symbol_x, symbol_y]].dropna()
    if len(pair_df) < 50:
        raise ValueError(f"Insufficient data for pair {symbol_x}/{symbol_y}.")

    log_x = np.log(pair_df[symbol_x].astype(float))
    log_y = np.log(pair_df[symbol_y].astype(float))
    log_df = pd.concat([log_x.rename(symbol_x), log_y.rename(symbol_y)], axis=1).dropna()

    score, p_value, critical_values = coint(log_df[symbol_x], log_df[symbol_y])
    johansen = coint_johansen(log_df[[symbol_x, symbol_y]], det_order, k_ar_diff)

    trace_sig = float(johansen.lr1[0]) > float(johansen.cvt[0, 1])
    eig_sig = float(johansen.lr2[0]) > float(johansen.cvm[0, 1])
    eg_sig = float(p_value) < 0.05

    return {
        "pair": f"{symbol_x}/{symbol_y}",
        "n_obs": int(len(log_df)),
        "engle_granger_t_stat": float(score),
        "engle_granger_p_value": float(p_value),
        "engle_granger_crit_1pct": float(critical_values[0]),
        "engle_granger_crit_5pct": float(critical_values[1]),
        "engle_granger_crit_10pct": float(critical_values[2]),
        "johansen_trace_stat_r0": float(johansen.lr1[0]),
        "johansen_trace_crit_95_r0": float(johansen.cvt[0, 1]),
        "johansen_eig_stat_r0": float(johansen.lr2[0]),
        "johansen_eig_crit_95_r0": float(johansen.cvm[0, 1]),
        "engle_granger_is_significant_5pct": eg_sig,
        "johansen_trace_significant_95pct": trace_sig,
        "johansen_eig_significant_95pct": eig_sig,
        # Strict combined screen: EG plus at least one Johansen rank-0 rejection.
        "combined_selection_pass": bool(eg_sig and (trace_sig or eig_sig)),
    }


def rank_candidate_pairs(
    prices: pd.DataFrame,
    pairs: Iterable[tuple[str, str]],
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> pd.DataFrame:
    """Evaluate and rank candidate pairs by Engle-Granger p-value.

    Rows that fail (e.g. insufficient data or stats errors) are kept with NaNs
    and ``rank_error`` set so screening does not abort the whole batch.
    """
    results: list[dict] = []
    for symbol_x, symbol_y in pairs:
        row: dict = {
            "pair": f"{symbol_x}/{symbol_y}",
            "symbol_x": symbol_x,
            "symbol_y": symbol_y,
            "rank_error": None,
        }
        try:
            row.update(
                test_pair_cointegration(
                    prices=prices,
                    symbol_x=symbol_x,
                    symbol_y=symbol_y,
                    det_order=det_order,
                    k_ar_diff=k_ar_diff,
                )
            )
        except Exception as exc:  # noqa: BLE001 — batch screening must continue
            row["rank_error"] = str(exc)
            row["n_obs"] = int(
                prices[[symbol_x, symbol_y]].dropna().shape[0]
                if symbol_x in prices.columns and symbol_y in prices.columns
                else 0
            )
            for key in (
                "engle_granger_t_stat",
                "engle_granger_p_value",
                "engle_granger_crit_1pct",
                "engle_granger_crit_5pct",
                "engle_granger_crit_10pct",
                "johansen_trace_stat_r0",
                "johansen_trace_crit_95_r0",
                "johansen_eig_stat_r0",
                "johansen_eig_crit_95_r0",
                "engle_granger_is_significant_5pct",
                "johansen_trace_significant_95pct",
                "johansen_eig_significant_95pct",
                "combined_selection_pass",
            ):
                row[key] = np.nan if "significant" not in key and "pass" not in key else False
        results.append(row)

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    if "engle_granger_p_value" in out.columns:
        out = out.sort_values(
            "engle_granger_p_value",
            ascending=True,
            na_position="last",
        ).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out
