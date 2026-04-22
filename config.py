"""Project configuration for statistical arbitrage research."""

from __future__ import annotations

DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = None  # Use latest available market close.

# Starter candidate pairs for mean-reversion cointegration screening.
DEFAULT_PAIRS = [
    ("GLD", "SLV"),
    ("XOM", "CVX"),
    ("KO", "PEP"),
    ("SPY", "QQQ"),
]

# Step 2 — spread / OU: rolling OLS hedge ratio and z-score windows (trading days).
DEFAULT_HEDGE_WINDOW = 60
DEFAULT_ZSCORE_WINDOW = 60

# Step 3 — signals, sizing, and risk (trading days unless noted).
DEFAULT_COINT_ROLLING_WINDOW = 60
DEFAULT_SPREAD_VOL_WINDOW = 60
DEFAULT_HALF_LIFE_WINDOW = 60
DEFAULT_LIQUIDITY_WINDOW = 20
# Minimum rolling average dollar volume per leg (USD); filter uses min(leg1, leg2).
DEFAULT_MIN_DOLLAR_VOLUME_USD = 5_000_000.0
# Engle–Granger rolling p-value must be below this to treat cointegration as stable.
DEFAULT_COINT_P_MAX = 0.15
# Half-life (days) must lie in this band for mean-reversion to be usable.
DEFAULT_HALF_LIFE_MIN_DAYS = 3.0
DEFAULT_HALF_LIFE_MAX_DAYS = 90.0

Z_ENTRY = 1.5
Z_EXIT = 0.5
Z_STOP = 3.0
MAX_HOLDING_DAYS = 20
# Continuous sizing: |z| is scaled by this before capping at 1 (min(|z|/Z_SIZE_REF, 1)).
Z_SIZE_REF = 2.5
# Target daily volatility of spread (fraction of spread level); scales position inversely with realized vol.
TARGET_SPREAD_DAILY_VOL = 0.015
# Portfolio / pair limits (fraction of unit capital / gross exposure cap).
DEFAULT_MAX_CAPITAL_PER_PAIR = 0.35
DEFAULT_MAX_GROSS_LEVERAGE = 2.0

# Supporting filters (computed in features; subset used for entry gate).
CORR_ROLLING_WINDOW = 20
CORR_INSTABILITY_ROLLING = 60
# Reject pair-day if std of rolling correlation exceeds this (unstable relationship).
MAX_CORR_INSTABILITY_STD = 0.25
SPREAD_SLOPE_WINDOW = 20
# Block entry if |t-stat| of OLS slope of spread on time exceeds this (strong trend).
SPREAD_SLOPE_MAX_ABS_TSTAT = 2.5

# Step 4 — backtest design and costs.
TRAIN_START = "2018-01-01"
TRAIN_END = "2021-12-31"
VALIDATION_START = "2022-01-01"
VALIDATION_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2025-12-31"
# Walk-forward refit frequency: "M" (monthly) or "Q" (quarterly).
WALK_FORWARD_REBALANCE_FREQ = "M"

# Transaction costs (applied on change in each leg exposure).
TRANSACTION_COST_BPS_PER_LEG = 7.0
SLIPPAGE_BPS_PER_LEG = 3.0
TURNOVER_PENALTY_BPS = 1.0

