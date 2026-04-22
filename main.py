"""CLI entry: Step 1 cointegration screen or Step 3 signal generation."""

from __future__ import annotations

import sys

import pandas as pd

from config import DEFAULT_END_DATE, DEFAULT_PAIRS, DEFAULT_START_DATE
from src.data_loader import fetch_adjusted_close, fetch_adjusted_close_and_volume
from src.pair_selection import rank_candidate_pairs
from src.signals import generate_signals_for_pair


def run_step_1() -> pd.DataFrame:
    symbols = sorted({symbol for pair in DEFAULT_PAIRS for symbol in pair})
    prices = fetch_adjusted_close(
        symbols=symbols,
        start_date=DEFAULT_START_DATE,
        end_date=DEFAULT_END_DATE,
    )
    return rank_candidate_pairs(prices=prices, pairs=DEFAULT_PAIRS)


def run_step_3(pair: tuple[str, str] | None = None) -> pd.DataFrame:
    """Build spread features and simulated positions for one pair (Yahoo + volume)."""
    s1, s2 = pair if pair is not None else DEFAULT_PAIRS[0]
    prices, volumes = fetch_adjusted_close_and_volume(
        symbols=[s1, s2],
        start_date=DEFAULT_START_DATE,
        end_date=DEFAULT_END_DATE,
    )
    return generate_signals_for_pair(prices, volumes, s1, s2)


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    mode = (sys.argv[1] if len(sys.argv) > 1 else "step1").lower()

    if mode in ("step3", "3", "signals"):
        p = (sys.argv[2], sys.argv[3]) if len(sys.argv) >= 4 else None
        sig = run_step_3(pair=p)
        cols = [
            "z_score",
            "hedge_ratio_roll",
            "half_life_roll",
            "spread_vol_roll",
            "rolling_eg_pvalue",
            "entry_gate_all",
            "position",
        ]
        present = [c for c in cols if c in sig.columns]
        print(f"\nStep 3 — last 12 rows ({sig.attrs.get('symbol_1')}/{sig.attrs.get('symbol_2')}):\n")
        print(sig[present].tail(12).to_string())
        print(f"\nNon-zero position days: {(sig['position'].abs() > 1e-12).sum()}")
    else:
        results = run_step_1()
        if results.empty:
            print("No results generated.")
        else:
            print("\nStep 1 — cointegration on log prices (sorted by Engle-Granger p-value):\n")
            print(results.to_string(index=False))
