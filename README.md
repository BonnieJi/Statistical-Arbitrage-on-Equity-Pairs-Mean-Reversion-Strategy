# Statistical Arbitrage on Equity Pairs

## Step 1 - Cointegration Testing

This project starts by pulling daily adjusted close prices from Yahoo Finance and
testing candidate pairs for cointegration using:

- Engle-Granger 2-step test
- Johansen trace and max-eigenvalue tests

**Correlation vs cointegration:** correlation is about co-movement of returns or
levels over a window; cointegration is about a *stable linear combination* of
price levels that is mean-reverting (I(0) spread) even when each price series is
non-stationary (I(1)). Pairs trading on spreads relies on cointegration (or a
close empirical analogue), not correlation alone.

Starter pairs:

- `GLD/SLV`
- `XOM/CVX`
- `KO/PEP`
- `SPY/QQQ`

### Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Defaults (dates, pairs) live in `config.py`. Engle-Granger and Johansen can
disagree depending on deterministic terms and VAR lag choices; see
`src/pair_selection.py` for the exact `statsmodels` calls.

Step 1 tests are run on **log prices** (not raw levels).

## Step 3 — Signal generation

Spread on logs: `spread_t = log(P1_t) - beta_t * log(P2_t)` with **rolling OLS**
`beta_t` on past data only. Signals use **z-score** of the spread, **rolling
Engle–Granger p-value** (cointegration stability), **rolling half-life**,
**rolling spread volatility** (for sizing), plus liquidity, correlation
stability, and spread slope (trend) gates. See `config.py` for thresholds.

**Entry:** `|z| > 2`, all entry gates true (including rolling cointegration
`p < DEFAULT_COINT_P_MAX`). **Short spread** when `z > 2`; **long spread** when
`z < -2`. **Exit:** `|z| < 0.5`, or `|z| > 3.5` (stop), or max holding period.
**Sizing:** continuous `min(|z|/Z_SIZE_REF, 1)` times volatility scaling toward
`TARGET_SPREAD_DAILY_VOL`, capped by `DEFAULT_MAX_CAPITAL_PER_PAIR` and
`DEFAULT_MAX_GROSS_LEVERAGE`. No new entry until the position is flat again.

```bash
python main.py step3              # first default pair
python main.py step3 GLD SLV      # explicit pair
```

## Step 4 — Backtest design

The backtest uses explicit time splits and walk-forward recalibration:

- Train / pair calibration: `2018-2021`
- Validation / tuning: `2022`
- Out-of-sample test: `2023-2025`

Walk-forward loop (monthly by default, configurable):

- Re-estimate hedge ratio
- Re-estimate OU parameters
- Retest cointegration stability (already embedded in step3 gating)
- Trade the next period only

Transaction costs are explicit and configurable in `config.py`:

- Per-leg trading cost in bps
- Per-leg slippage proxy in bps
- Turnover penalty in bps

Default assumption is `7 bps` trading + `3 bps` slippage + `1 bps` turnover
penalty, which is in the requested 5-10 bps per-leg range.

```bash
python main.py step4              # first default pair
python main.py step4 XOM CVX      # explicit pair
```

## Plot Reports

Generate all plots:

```bash
python main.py plots
```

Each run is archived in `reports/runs/<timestamp>/` and mirrored to
`reports/latest/` (latest run only).

### GLD/SLV Lead Case Study

![GLD/SLV Spread](reports/latest/gld_slv_lead_case_spread.png)
![GLD/SLV Z-Score](reports/latest/gld_slv_lead_case_zscore.png)
![GLD/SLV Entries and Exits](reports/latest/gld_slv_lead_case_entries_exits.png)
![GLD/SLV Equity Curve](reports/latest/gld_slv_lead_case_equity_curve.png)

### XOM/CVX Negative Example

![XOM/CVX Spread](reports/latest/xom_cvx_negative_case_spread.png)
![XOM/CVX Z-Score](reports/latest/xom_cvx_negative_case_zscore.png)
![XOM/CVX Entries and Exits](reports/latest/xom_cvx_negative_case_entries_exits.png)
![XOM/CVX Equity Curve](reports/latest/xom_cvx_negative_case_equity_curve.png)

### Pair Comparisons

![Cumulative Net Return by Pair](reports/latest/comparison_cumulative_net_by_pair.png)
![Gross vs Net Return by Pair](reports/latest/comparison_gross_vs_net_by_pair.png)

## Risk & Distribution Upgrades

- Drawdown-aware reporting: max drawdown, rolling Sharpe (60d), and equity+drawdown charts.
- Trade distribution diagnostics: trade PnL histogram, holding-period histogram, win-rate stats.
- Sensitivity sweeps (research robustness):
  - entry threshold (`1.0 -> 2.5`)
  - exit threshold (`0.0 -> 1.0`)
  - cointegration p-value threshold
  - half-life cutoff

Generated plots include:

- `reports/latest/gld_slv_lead_case_rolling_sharpe_60d.png`
- `reports/latest/gld_slv_lead_case_trade_pnl_hist.png`
- `reports/latest/gld_slv_lead_case_holding_period_hist.png`
- `reports/latest/gld_slv_sensitivity_sharpe.png`
- `reports/latest/gld_slv_sensitivity_net_return.png`
- `reports/latest/multi_pair_portfolio_equity_curve.png`

## Portfolio-Level Strategy

Pairs are now combined using inverse-volatility weights with leverage cap
(`max_leverage=1.5` in plotting mode) to produce a portfolio equity curve and
portfolio Sharpe.

## Link to Market Making

Conceptual bridge to execution/microstructure:

- When spread z-score is high (`z > entry`), inventory/quote skew can be biased
  toward selling pressure (downward quote skew).
- When spread z-score is low (`z < -entry`), skew can be biased toward buying
  pressure (upward quote skew).

This ties medium-horizon stat-arb alpha to short-horizon market-making quote
placement.