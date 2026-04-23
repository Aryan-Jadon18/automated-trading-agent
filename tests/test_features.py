"""Unit tests for feature engineering pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.trading.data.features import (
    atr,
    bollinger_bands,
    build_features,
    ema,
    log_returns,
    macd,
    obv,
    rsi,
    sma,
    stochastic,
)


@pytest.fixture
def price_series() -> pd.Series:
    np.random.seed(42)
    prices = 100 * np.exp(np.random.randn(200).cumsum() * 0.01)
    return pd.Series(prices, name="close")


@pytest.fixture
def ohlcv_df(price_series) -> pd.DataFrame:
    n = len(price_series)
    np.random.seed(0)
    close = price_series
    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": pd.Series(high, name="high"),
        "low": pd.Series(low, name="low"),
        "close": close,
        "volume": pd.Series(volume, name="volume"),
    })


def test_sma_length(price_series):
    result = sma(price_series, 20)
    assert len(result) == len(price_series)
    assert result.iloc[:19].isna().all()
    assert not result.iloc[19:].isna().any()


def test_ema_no_nan_after_warmup(price_series):
    result = ema(price_series, 12)
    assert not result.iloc[11:].isna().any()


def test_rsi_bounds(price_series):
    result = rsi(price_series, 14).dropna()
    assert (result >= 0).all() and (result <= 100).all()


def test_macd_columns(price_series):
    result = macd(price_series)
    assert set(result.columns) == {"macd", "macd_signal", "macd_hist"}


def test_bollinger_width_positive(price_series):
    bb = bollinger_bands(price_series).dropna()
    assert (bb["bb_upper"] >= bb["bb_lower"]).all()
    assert (bb["bb_width"] >= 0).all()


def test_log_returns_mean_near_zero(price_series):
    rets = log_returns(price_series).dropna()
    assert abs(rets.mean()) < 0.05


def test_build_features_no_nan(ohlcv_df):
    features = build_features(ohlcv_df)
    assert not features.isna().any().any()
    assert len(features) > 0


def test_build_features_columns(ohlcv_df):
    features = build_features(ohlcv_df)
    expected = {"rsi_14", "macd", "bb_upper", "atr_14", "obv", "sma_20", "sma_50"}
    assert expected.issubset(set(features.columns))
