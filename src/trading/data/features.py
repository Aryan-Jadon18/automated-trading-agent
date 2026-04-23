"""Feature engineering — technical indicators + return-based features."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Trend ────────────────────────────────────────────────────────────────────

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": macd_line - signal_line,
    })


# ── Momentum ─────────────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


# ── Volatility ────────────────────────────────────────────────────────────────

def bollinger_bands(
    close: pd.Series, window: int = 20, n_std: float = 2.0
) -> pd.DataFrame:
    mid = sma(close, window)
    std = close.rolling(window).std()
    return pd.DataFrame({
        "bb_upper": mid + n_std * std,
        "bb_mid": mid,
        "bb_lower": mid - n_std * std,
        "bb_width": (2 * n_std * std) / mid,
        "bb_pct": (close - (mid - n_std * std)) / (2 * n_std * std),
    })


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ── Volume ───────────────────────────────────────────────────────────────────

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical = (high + low + close) / 3
    return (typical * volume).cumsum() / volume.cumsum()


# ── Return features ───────────────────────────────────────────────────────────

def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def rolling_features(close: pd.Series, windows: list[int] = [5, 10, 20, 60]) -> pd.DataFrame:
    rets = log_returns(close)
    frames = {}
    for w in windows:
        frames[f"ret_{w}d"] = rets.rolling(w).sum()
        frames[f"vol_{w}d"] = rets.rolling(w).std() * np.sqrt(252)
        frames[f"mom_{w}d"] = close / close.shift(w) - 1
    return pd.DataFrame(frames)


# ── Master builder ────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a raw OHLCV DataFrame, return a feature matrix ready for ML.
    Drops rows with NaN (warm-up period).
    """
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    parts = [
        df[["open", "high", "low", "close", "volume"]],
        sma(c, 20).rename("sma_20"),
        sma(c, 50).rename("sma_50"),
        ema(c, 12).rename("ema_12"),
        ema(c, 26).rename("ema_26"),
        macd(c),
        rsi(c, 14).rename("rsi_14"),
        stochastic(h, l, c),
        bollinger_bands(c),
        atr(h, l, c).rename("atr_14"),
        obv(c, v).rename("obv"),
        rolling_features(c),
    ]

    out = pd.concat(parts, axis=1)
    out["close_vs_sma20"] = c / out["sma_20"] - 1
    out["close_vs_sma50"] = c / out["sma_50"] - 1
    out["volume_sma20"] = v / v.rolling(20).mean()

    return out.dropna()
