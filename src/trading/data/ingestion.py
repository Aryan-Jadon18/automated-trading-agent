"""Market data ingestion — downloads and caches OHLCV data."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

_CACHE_DIR = Path("data/cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(symbol: str, interval: str, start: str, end: str) -> Path:
    return _CACHE_DIR / f"{symbol}_{interval}_{start}_{end}.parquet"


def fetch_ohlcv(
    symbol: str,
    interval: str = "1d",
    lookback_days: int = 1825,
    start: Optional[str] = None,
    end: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV data for *symbol* via yfinance.

    Returns a DataFrame with columns [open, high, low, close, volume]
    indexed by UTC date/datetime. Columns are lower-cased.
    """
    _end = end or datetime.utcnow().strftime("%Y-%m-%d")
    _start = start or (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    cache_file = _cache_path(symbol, interval, _start, _end)

    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=_start, end=_end, interval=interval, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for {symbol} [{_start} → {_end}]")

    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "datetime"

    if use_cache:
        df.to_parquet(cache_file)

    return df


def fetch_multi(
    symbols: list[str],
    interval: str = "1d",
    lookback_days: int = 1825,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple symbols. Returns {symbol: DataFrame}."""
    return {
        sym: fetch_ohlcv(sym, interval=interval, lookback_days=lookback_days, use_cache=use_cache)
        for sym in symbols
    }


def align_closes(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Align close prices across symbols to a common datetime index.
    Forward-fills gaps (e.g. holidays in some markets).
    """
    closes = {sym: df["close"].rename(sym) for sym, df in data.items()}
    combined = pd.concat(closes, axis=1).sort_index()
    return combined.ffill()
