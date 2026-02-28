"""Data loading utilities for financial time series."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional


def load_yahoo_finance(
    symbols: List[str],
    period: str = "5y",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Download OHLCV data from Yahoo Finance.

    Args:
        symbols: List of ticker symbols (e.g. ['SPY', 'AAPL', 'BTC-USD']).
        period: Data period ('1y', '5y', 'max', etc.).
        interval: Candle interval ('1d', '1h', '5m', etc.).

    Returns:
        Dict mapping symbol -> DataFrame with OHLCV columns.
    """
    import yfinance as yf

    data = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            print(f"Warning: No data returned for {symbol}")
            continue
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df = df.dropna()
        data[symbol] = df
    return data


def load_csv(
    path: str,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load OHLCV data from a CSV file.

    Args:
        path: Path to CSV file.
        columns: Column names to use. If None, expects
                 open/high/low/close/volume headers.

    Returns:
        DataFrame with OHLCV columns.
    """
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    if columns is not None:
        df.columns = columns
    expected = ["open", "high", "low", "close", "volume"]
    df.columns = [c.lower() for c in df.columns]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[expected].dropna()


def combine_datasets(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combine multiple symbol DataFrames into one, adding a symbol column."""
    frames = []
    for symbol, df in datasets.items():
        df = df.copy()
        df["symbol"] = symbol
        frames.append(df)
    return pd.concat(frames, axis=0).sort_index()
