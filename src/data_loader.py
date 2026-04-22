"""Utilities for fetching and preparing market data."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import yfinance as yf


def fetch_adjusted_close(
    symbols: Iterable[str],
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download daily adjusted close prices for the provided symbols.

    Returns a DataFrame indexed by trading date with one column per symbol.
    """
    symbols = list(dict.fromkeys(symbols))  # Preserve order and remove duplicates.
    if not symbols:
        raise ValueError("At least one symbol is required to fetch prices.")

    raw = yf.download(
        tickers=symbols,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise ValueError("No data returned from Yahoo Finance.")

    if isinstance(raw.columns, pd.MultiIndex):
        price_df = raw["Adj Close"].copy()
    else:
        # Single ticker download can be flat columns.
        if "Adj Close" not in raw.columns:
            raise ValueError("Expected 'Adj Close' column is missing from download.")
        price_df = raw[["Adj Close"]].rename(columns={"Adj Close": symbols[0]})

    price_df = price_df.sort_index().dropna(how="all")
    missing_symbols = [symbol for symbol in symbols if symbol not in price_df.columns]
    if missing_symbols:
        raise ValueError(f"Missing downloaded symbols: {missing_symbols}")

    return price_df[symbols]


def fetch_adjusted_close_and_volume(
    symbols: Iterable[str],
    start_date: str,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download daily adjusted close and share volume (aligned columns per symbol)."""
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        raise ValueError("At least one symbol is required to fetch prices.")

    raw = yf.download(
        tickers=symbols,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise ValueError("No data returned from Yahoo Finance.")

    if isinstance(raw.columns, pd.MultiIndex):
        price_df = raw["Adj Close"].copy()
        vol_df = raw["Volume"].copy()
    else:
        if "Adj Close" not in raw.columns or "Volume" not in raw.columns:
            raise ValueError("Expected 'Adj Close' and 'Volume' columns.")
        price_df = raw[["Adj Close"]].rename(columns={"Adj Close": symbols[0]})
        vol_df = raw[["Volume"]].rename(columns={"Volume": symbols[0]})

    price_df = price_df.sort_index().dropna(how="all")
    vol_df = vol_df.sort_index().reindex(price_df.index)
    missing = [s for s in symbols if s not in price_df.columns]
    if missing:
        raise ValueError(f"Missing downloaded symbols: {missing}")

    return price_df[symbols], vol_df[symbols]

