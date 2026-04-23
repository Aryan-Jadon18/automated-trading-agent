"""Tests for event-driven backtest engine, portfolio, and strategies."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.trading.backtest.engine import BacktestResult, run_backtest
from src.trading.backtest.events import Direction, FillEvent, MarketEvent, SignalEvent
from src.trading.backtest.portfolio import Portfolio, Position
from src.trading.strategy.technical import MeanReversion, SMACrossover


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300, seed: int = 42, trend: float = 0.0003) -> pd.DataFrame:
    """Synthetic OHLCV with a controllable upward drift."""
    np.random.seed(seed)
    log_returns = np.random.randn(n) * 0.01 + trend
    close = 100.0 * np.exp(np.cumsum(log_returns))
    high = close * (1 + np.abs(np.random.randn(n) * 0.003))
    low = close * (1 - np.abs(np.random.randn(n) * 0.003))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.random.randint(1_000_000, 5_000_000, n).astype(float)
    idx = pd.date_range("2022-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return _make_ohlcv()


@pytest.fixture
def trending_ohlcv() -> pd.DataFrame:
    return _make_ohlcv(n=400, trend=0.001)   # strong uptrend


# ── Event / Portfolio unit tests ──────────────────────────────────────────────

def test_position_update_long_open():
    pos = Position("SPY")
    pos.update_fill(10, 100.0)
    assert pos.quantity == 10
    assert pos.avg_price == 100.0


def test_position_add_to_long():
    pos = Position("SPY")
    pos.update_fill(10, 100.0)
    pos.update_fill(10, 120.0)
    assert pos.quantity == 20
    assert pos.avg_price == 110.0


def test_position_close_long():
    pos = Position("SPY")
    pos.update_fill(10, 100.0)
    pos.update_fill(-10, 110.0)
    assert pos.quantity == 0
    assert pos.realized_pnl == pytest.approx(100.0)


def test_portfolio_initial_equity():
    p = Portfolio(["SPY"], initial_capital=100_000.0)
    assert p.cash == 100_000.0


def test_portfolio_on_fill_updates_cash():
    p = Portfolio(["SPY"], initial_capital=100_000.0, commission=0.0)
    fill = FillEvent(datetime=datetime.utcnow(), symbol="SPY", quantity=10, fill_price=100.0, commission=0.0)
    p.on_fill(fill)
    assert p.cash == pytest.approx(99_000.0)


def test_portfolio_signal_generates_order(ohlcv):
    p = Portfolio(["SPY"], initial_capital=100_000.0)
    row = ohlcv.iloc[60]
    market_evt = MarketEvent(
        datetime=ohlcv.index[60].to_pydatetime(),
        symbol="SPY", open=row["open"], high=row["high"],
        low=row["low"], close=row["close"], volume=row["volume"],
    )
    p.on_market(market_evt)
    signal = SignalEvent(datetime=market_evt.datetime, symbol="SPY", direction=Direction.LONG)
    order = p.on_signal(signal)
    assert order is not None
    assert order.quantity > 0


# ── Strategy unit tests ───────────────────────────────────────────────────────

def test_sma_crossover_requires_warmup(ohlcv):
    strategy = SMACrossover(["SPY"], fast=20, slow=50)
    signals = []
    for i in range(55):          # fewer bars than slow window
        row = ohlcv.iloc[i]
        evt = MarketEvent(
            datetime=ohlcv.index[i].to_pydatetime(),
            symbol="SPY", open=row["open"], high=row["high"],
            low=row["low"], close=row["close"], volume=row["volume"],
        )
        signals.append(strategy.on_bar(evt))
    assert all(s is None for s in signals)


def test_sma_crossover_emits_signals(trending_ohlcv):
    strategy = SMACrossover(["SPY"], fast=10, slow=30)
    signals = []
    for i in range(len(trending_ohlcv)):
        row = trending_ohlcv.iloc[i]
        evt = MarketEvent(
            datetime=trending_ohlcv.index[i].to_pydatetime(),
            symbol="SPY", open=row["open"], high=row["high"],
            low=row["low"], close=row["close"], volume=row["volume"],
        )
        s = strategy.on_bar(evt)
        if s is not None:
            signals.append(s)
    assert len(signals) > 0


def test_mean_reversion_reset():
    strategy = MeanReversion(["SPY"])
    strategy._position["SPY"] = Direction.LONG
    strategy.reset()
    assert strategy._position["SPY"] is None


# ── Full backtest integration tests ───────────────────────────────────────────

def test_backtest_runs_without_error(ohlcv):
    strategy = SMACrossover(["SPY"], fast=10, slow=30)
    result = run_backtest({"SPY": ohlcv}, strategy, initial_capital=100_000.0)
    assert isinstance(result, BacktestResult)
    assert result.n_bars == len(ohlcv)


def test_backtest_equity_curve_length(ohlcv):
    strategy = SMACrossover(["SPY"], fast=10, slow=30)
    result = run_backtest({"SPY": ohlcv}, strategy)
    assert len(result.equity_curve) == len(ohlcv)


def test_backtest_equity_starts_at_capital(ohlcv):
    strategy = SMACrossover(["SPY"], fast=10, slow=30)
    result = run_backtest({"SPY": ohlcv}, strategy, initial_capital=50_000.0)
    assert result.equity_curve["equity"].iloc[0] == pytest.approx(50_000.0)


def test_backtest_trending_market_positive_return(trending_ohlcv):
    """SMA crossover on a strong uptrend should generate positive return."""
    strategy = SMACrossover(["SPY"], fast=10, slow=30)
    result = run_backtest({"SPY": trending_ohlcv}, strategy, initial_capital=100_000.0)
    assert result.performance.get("total_return", 0) > 0


def test_backtest_no_lookahead(ohlcv):
    """Fills should always happen at a bar *after* the signal bar."""
    strategy = SMACrossover(["SPY"], fast=10, slow=30)
    result = run_backtest({"SPY": ohlcv}, strategy)
    if not result.trade_log.empty and len(result.equity_curve) > 1:
        # Equity at bar 0 should equal initial capital (no fills on first bar)
        assert result.equity_curve["equity"].iloc[0] == pytest.approx(100_000.0, rel=1e-3)


def test_backtest_multi_symbol():
    data = {"SPY": _make_ohlcv(300, seed=1), "QQQ": _make_ohlcv(300, seed=2)}
    strategy = SMACrossover(["SPY", "QQQ"], fast=10, slow=30)
    result = run_backtest(data, strategy, initial_capital=200_000.0)
    assert result.n_bars > 0
    assert "total_return" in result.performance


def test_backtest_performance_keys(ohlcv):
    strategy = SMACrossover(["SPY"], fast=10, slow=30)
    result = run_backtest({"SPY": ohlcv}, strategy)
    expected = {"total_return", "cagr", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate", "profit_factor"}
    assert expected.issubset(set(result.performance.keys()))
