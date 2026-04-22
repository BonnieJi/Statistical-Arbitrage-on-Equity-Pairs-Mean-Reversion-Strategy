"""CLI entry: Step 1 cointegration screen or Step 3 signal generation."""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
import shutil

import pandas as pd

from config import DEFAULT_END_DATE, DEFAULT_PAIRS, DEFAULT_START_DATE
from src.backtest import (
    build_equal_risk_portfolio,
    run_sensitivity_analysis,
    run_walk_forward_backtest_for_pair,
    split_period_performance,
    summarize_trades,
)
from src.data_loader import fetch_adjusted_close, fetch_adjusted_close_and_volume
from src.metrics import max_drawdown, summarize_trade_distribution
from src.pair_selection import rank_candidate_pairs
from src.plots import (
    plot_case_study,
    plot_pair_comparisons,
    plot_portfolio_equity,
    plot_sensitivity_curves,
    plot_trade_distribution,
)
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


def run_step_4(
    pair: tuple[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Walk-forward backtest (monthly/quarterly refit + explicit transaction costs)."""
    s1, s2 = pair if pair is not None else DEFAULT_PAIRS[0]
    daily, wf_summary = run_walk_forward_backtest_for_pair(s1, s2)
    split_summary = split_period_performance(daily) if not daily.empty else pd.DataFrame()
    trades = summarize_trades(daily) if not daily.empty else pd.DataFrame()
    return daily, wf_summary, split_summary, trades


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    mode = (sys.argv[1] if len(sys.argv) > 1 else "step1").lower()

    if mode in ("step3", "3", "signals"):
        p = (sys.argv[2], sys.argv[3]) if len(sys.argv) >= 4 else None
        sig = run_step_3(pair=p)
        cols = [
            "z_score",
            "entry_gate_z",
            "entry_gate_cointegration",
            "entry_gate_half_life",
            "entry_gate_volatility",
            "hedge_ratio_roll",
            "half_life_roll",
            "spread_vol_roll",
            "rolling_eg_pvalue",
            "entry_gate_all",
            "entry_event",
            "exit_event",
            "position",
        ]
        present = [c for c in cols if c in sig.columns]
        print(f"\nStep 3 — last 20 rows ({sig.attrs.get('symbol_1')}/{sig.attrs.get('symbol_2')}):\n")
        print(sig[present].tail(20).to_string())
        gate_counts = sig.attrs.get("gate_counts", {})
        if gate_counts:
            print("\nStep 3 — gate/debug counts:\n")
            for k, v in gate_counts.items():
                print(f"{k}: {v}")
    elif mode in ("step4", "4", "backtest"):
        if len(sys.argv) >= 3 and sys.argv[2].lower() == "all":
            rows = []
            for s1, s2 in DEFAULT_PAIRS:
                daily_i, _, split_i, trades_i = run_step_4(pair=(s1, s2))
                if split_i.empty:
                    continue
                test_row = split_i.loc[split_i["split"] == "test_oos"].head(1)
                if test_row.empty:
                    continue
                r = test_row.iloc[0].to_dict()
                r["pair"] = f"{s1}/{s2}"
                r["trades"] = int(len(trades_i))
                rows.append(r)
            out = pd.DataFrame(rows)
            print("\nStep 4 — multi-pair test summary:\n")
            print(out.to_string(index=False) if not out.empty else "No results.")
            raise SystemExit(0)

        p = (sys.argv[2], sys.argv[3]) if len(sys.argv) >= 4 else None
        daily, wf_summary, split_summary, trades = run_step_4(pair=p)
        if daily.empty:
            print("\nStep 4 — no backtest output generated.\n")
        else:
            print("\nStep 4 — walk-forward period summary (OOS trading windows):\n")
            print(wf_summary.tail(12).to_string(index=False))
            print("\nStep 4 — gate/debug counts (whole daily panel):\n")
            print(
                f"days_z_abs_gt_entry: {(daily['entry_gate_z'].fillna(False)).sum()}\n"
                f"days_rolling_eg_p_lt_thresh: {(daily['entry_gate_cointegration'].fillna(False)).sum()}\n"
                f"days_half_life_in_range: {(daily['entry_gate_half_life'].fillna(False)).sum()}\n"
                f"days_entry_gate_all_true: {(daily['entry_gate_all'].fillna(False)).sum()}\n"
                f"non_zero_position_days: {(daily['position'].abs() > 1e-12).sum()}\n"
                f"completed_trades: {(daily['exit_event'].fillna(False)).sum()}"
            )
            print("\nStep 4 — split summary (train/validation/test):\n")
            print(split_summary.to_string(index=False))
            if not trades.empty:
                print("\nStep 4 — trade-by-trade diagnostics:\n")
                print(trades.to_string(index=False))
                print(
                    f"\nNo-cost total (gross): {trades['gross_pnl'].sum():.6f} | "
                    f"After-cost total (net): {trades['net_pnl'].sum():.6f} | "
                    f"Total cost drag: {(trades['gross_pnl'].sum() - trades['net_pnl'].sum()):.6f}"
                )
            print(
                f"\nFinal cumulative gross: {daily['cum_gross'].iloc[-1]:.6f} | "
                f"Final cumulative net: {daily['cum_net'].iloc[-1]:.6f}"
            )
    elif mode in ("plots", "plot", "viz"):
        pair_daily: dict[str, pd.DataFrame] = {}
        for s1, s2 in DEFAULT_PAIRS:
            daily_i, _, _, _ = run_step_4(pair=(s1, s2))
            pair_daily[f"{s1}/{s2}"] = daily_i

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("reports") / "runs" / run_id
        latest_dir = Path("reports") / "latest"
        written: list[str] = []
        # Lead case study (strongest OOS performer in current runs): GLD/SLV.
        if "GLD/SLV" in pair_daily:
            gld_trades = summarize_trades(pair_daily["GLD/SLV"])
            written.extend(
                plot_case_study(
                    pair_daily["GLD/SLV"],
                    pair_name="GLD/SLV (Lead Case)",
                    out_dir=out_dir,
                    filename_prefix="gld_slv_lead_case",
                )
            )
            written.extend(
                plot_trade_distribution(
                    gld_trades,
                    pair_name="GLD/SLV (Lead Case)",
                    out_dir=out_dir,
                    filename_prefix="gld_slv_lead_case",
                )
            )
            td = summarize_trade_distribution(gld_trades)
            print(
                "\nGLD/SLV trade distribution stats:\n"
                f"n_trades={int(td['n_trades'])}, win_rate={td['win_rate']:.2%}, "
                f"avg_win={td['avg_win']:.6f}, avg_loss={td['avg_loss']:.6f}, "
                f"avg_holding_days={td['avg_holding_days']:.2f}"
            )
            if not pair_daily["GLD/SLV"].empty:
                print(f"GLD/SLV max drawdown: {max_drawdown(pair_daily['GLD/SLV']['cum_net']):.6f}")
        # Negative example: XOM/CVX.
        if "XOM/CVX" in pair_daily:
            written.extend(
                plot_case_study(
                    pair_daily["XOM/CVX"],
                    pair_name="XOM/CVX (Negative Example)",
                    out_dir=out_dir,
                    filename_prefix="xom_cvx_negative_case",
                )
            )
        written.extend(plot_pair_comparisons(pair_daily, out_dir=out_dir))
        # Sensitivity analysis on lead pair.
        sens = run_sensitivity_analysis("GLD", "SLV")
        written.extend(plot_sensitivity_curves(sens, out_dir=out_dir, filename_prefix="gld_slv"))
        # Portfolio-level aggregation over all pairs.
        portfolio = build_equal_risk_portfolio(pair_daily, max_leverage=1.5)
        written.extend(plot_portfolio_equity(portfolio, out_dir=out_dir, filename_prefix="multi_pair"))
        if not portfolio.empty:
            port_sharpe = (portfolio["pnl_net"].mean() / (portfolio["pnl_net"].std() + 1e-12)) * (252**0.5)
            print(f"Portfolio Sharpe (daily, annualized): {port_sharpe:.3f}")
            print(f"Portfolio max drawdown: {max_drawdown(portfolio['cum_net']):.6f}")

        # Mirror most recent run into a stable folder for easy access.
        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        latest_dir.mkdir(parents=True, exist_ok=True)
        for path_str in written:
            src = Path(path_str)
            if src.exists():
                shutil.copy2(src, latest_dir / src.name)

        print("\nGenerated plots:\n")
        for p in written:
            print(p)
        print(f"\nLatest run mirror:\n{latest_dir}")
    else:
        results = run_step_1()
        if results.empty:
            print("No results generated.")
        else:
            print("\nStep 1 — cointegration on log prices (sorted by Engle-Granger p-value):\n")
            print(results.to_string(index=False))
