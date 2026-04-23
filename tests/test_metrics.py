"""Unit tests for performance metrics."""
import numpy as np
import pandas as pd
import pytest

from src.trading.utils.metrics import (
    cagr,
    calmar,
    max_drawdown,
    profit_factor,
    sharpe,
    sortino,
    summary,
    total_return,
    win_rate,
)


@pytest.fixture
def flat_equity() -> pd.Series:
    return pd.Series(np.linspace(100_000, 150_000, 252))


@pytest.fixture
def random_returns() -> pd.Series:
    np.random.seed(42)
    return pd.Series(np.random.randn(252) * 0.01)


def test_total_return(flat_equity):
    assert abs(total_return(flat_equity) - 0.5) < 1e-6


def test_max_drawdown_non_positive(flat_equity):
    assert max_drawdown(flat_equity) <= 0


def test_sharpe_zero_variance():
    flat = pd.Series([0.001] * 252)
    assert sharpe(flat) > 0


def test_win_rate_bounds(random_returns):
    wr = win_rate(random_returns)
    assert 0.0 <= wr <= 1.0


def test_summary_keys(flat_equity, random_returns):
    result = summary(flat_equity, random_returns)
    expected_keys = {"total_return", "cagr", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate", "profit_factor"}
    assert expected_keys == set(result.keys())
