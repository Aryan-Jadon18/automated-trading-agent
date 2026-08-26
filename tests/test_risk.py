"""Tests for the risk manager: Kelly sizing, drawdown guard, exposure and correlation limits."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.trading.risk.manager import (
    CorrelationLimiter,
    DrawdownGuard,
    ExposureLimits,
    RiskDecision,
    RiskManager,
    kelly_fraction,
    kelly_from_returns,
)


# ── Kelly ─────────────────────────────────────────────────────────────────────

def test_kelly_positive_edge():
    # 60% win rate, 2:1 payoff → f* = (0.6*2 - 0.4)/2 = 0.4
    assert kelly_fraction(0.6, 2.0, 1.0) == pytest.approx(0.4)


def test_kelly_no_edge_returns_zero():
    # 50% win rate, 1:1 payoff → zero edge
    assert kelly_fraction(0.5, 1.0, 1.0) == pytest.approx(0.0)


def test_kelly_negative_edge_clipped_to_zero():
    assert kelly_fraction(0.3, 1.0, 1.0) == 0.0


def test_kelly_clipped_at_one():
    assert kelly_fraction(0.99, 10.0, 1.0) <= 1.0


def test_kelly_handles_zero_loss():
    assert kelly_fraction(0.6, 2.0, 0.0) == 0.0


def test_kelly_rejects_invalid_win_rate():
    with pytest.raises(ValueError):
        kelly_fraction(1.5, 2.0, 1.0)


def test_kelly_from_returns_positive_drift():
    np.random.seed(0)
    rets = pd.Series(np.random.randn(500) * 0.01 + 0.002)
    assert kelly_from_returns(rets) > 0


def test_kelly_from_returns_all_wins_returns_zero():
    assert kelly_from_returns(pd.Series([0.01, 0.02, 0.03])) == 0.0


def test_kelly_from_returns_empty():
    assert kelly_from_returns(pd.Series([], dtype=float)) == 0.0


# ── DrawdownGuard ─────────────────────────────────────────────────────────────

def test_drawdown_guard_tracks_peak():
    g = DrawdownGuard(max_drawdown_pct=0.15)
    g.update(100_000)
    g.update(120_000)
    g.update(110_000)
    assert g.peak == 120_000


def test_drawdown_guard_current_drawdown():
    g = DrawdownGuard(max_drawdown_pct=0.20)
    g.update(100_000)
    g.update(90_000)
    assert g.current_drawdown == pytest.approx(-0.10)


def test_drawdown_guard_halts_on_breach():
    g = DrawdownGuard(max_drawdown_pct=0.15)
    g.update(100_000)
    assert not g.update(90_000)      # -10%, still fine
    assert g.update(84_000)          # -16%, breach


def test_drawdown_guard_hysteresis_resume():
    g = DrawdownGuard(max_drawdown_pct=0.15, resume_drawdown_pct=0.05)
    g.update(100_000)
    g.update(80_000)                 # -20% → halted
    assert g.is_halted
    g.update(90_000)                 # -10% → still halted (above resume level)
    assert g.is_halted
    g.update(96_000)                 # -4% → resumes
    assert not g.is_halted


def test_drawdown_guard_halt_count():
    g = DrawdownGuard(max_drawdown_pct=0.10, resume_drawdown_pct=0.02)
    g.update(100_000)
    g.update(85_000)                 # halt 1
    g.update(99_000)                 # resume
    g.update(80_000)                 # halt 2
    assert g.halt_count == 2


def test_drawdown_guard_reset():
    g = DrawdownGuard()
    g.update(100_000)
    g.update(50_000)
    g.reset()
    assert g.peak is None and not g.is_halted


def test_drawdown_guard_rejects_bad_config():
    with pytest.raises(ValueError):
        DrawdownGuard(max_drawdown_pct=0.0)
    with pytest.raises(ValueError):
        DrawdownGuard(max_drawdown_pct=0.10, resume_drawdown_pct=0.20)


# ── ExposureLimits ────────────────────────────────────────────────────────────

def test_exposure_caps_position_size():
    lim = ExposureLimits(max_position_pct=0.20)
    qty, reasons = lim.cap_quantity(
        "SPY", proposed_qty=1000, price=100.0, equity=100_000,
        positions={}, prices={"SPY": 100.0},
    )
    assert qty == pytest.approx(200)   # 20% of 100k / $100
    assert reasons


def test_exposure_allows_within_limit():
    lim = ExposureLimits(max_position_pct=0.20)
    qty, reasons = lim.cap_quantity(
        "SPY", proposed_qty=100, price=100.0, equity=100_000,
        positions={}, prices={"SPY": 100.0},
    )
    assert qty == 100
    assert reasons == []


def test_exposure_allows_reducing_position():
    lim = ExposureLimits(max_position_pct=0.05)
    qty, reasons = lim.cap_quantity(
        "SPY", proposed_qty=-500, price=100.0, equity=100_000,
        positions={"SPY": 500}, prices={"SPY": 100.0},
    )
    assert qty == -500                 # closing is never blocked


def test_exposure_max_open_positions():
    lim = ExposureLimits(max_open_positions=2)
    qty, reasons = lim.cap_quantity(
        "NEW", proposed_qty=10, price=100.0, equity=1_000_000,
        positions={"A": 10, "B": 10}, prices={"A": 100.0, "B": 100.0, "NEW": 100.0},
    )
    assert qty == 0
    assert any("open positions" in r for r in reasons)


def test_exposure_gross_cap():
    lim = ExposureLimits(max_position_pct=1.0, max_gross_exposure=0.5)
    qty, reasons = lim.cap_quantity(
        "SPY", proposed_qty=1000, price=100.0, equity=100_000,
        positions={}, prices={"SPY": 100.0},
    )
    assert abs(qty * 100.0) <= 50_000 + 1e-6


def test_exposure_rejects_bad_price():
    lim = ExposureLimits()
    qty, reasons = lim.cap_quantity(
        "SPY", 100, price=0.0, equity=100_000, positions={}, prices={},
    )
    assert qty == 0


# ── CorrelationLimiter ────────────────────────────────────────────────────────

@pytest.fixture
def corr_matrix() -> pd.DataFrame:
    syms = ["SPY", "QQQ", "GLD"]
    data = np.array([
        [1.00, 0.90, 0.10],   # SPY & QQQ highly correlated
        [0.90, 1.00, 0.05],
        [0.10, 0.05, 1.00],   # GLD uncorrelated
    ])
    return pd.DataFrame(data, index=syms, columns=syms)


def test_cluster_detection(corr_matrix):
    lim = CorrelationLimiter(corr_matrix, corr_threshold=0.7)
    assert lim.cluster_of("SPY") == {"SPY", "QQQ"}
    assert lim.cluster_of("GLD") == {"GLD"}


def test_correlation_caps_cluster_exposure(corr_matrix):
    lim = CorrelationLimiter(corr_matrix, corr_threshold=0.7, max_cluster_pct=0.30)
    # Already holding $25k of QQQ; adding SPY should be capped to $5k
    qty, reasons = lim.cap_quantity(
        "SPY", proposed_qty=500, price=100.0, equity=100_000,
        positions={"QQQ": 250}, prices={"QQQ": 100.0, "SPY": 100.0},
    )
    assert abs(qty * 100.0) <= 5_000 + 1e-6
    assert reasons


def test_correlation_rejects_when_cluster_full(corr_matrix):
    lim = CorrelationLimiter(corr_matrix, corr_threshold=0.7, max_cluster_pct=0.20)
    qty, reasons = lim.cap_quantity(
        "SPY", proposed_qty=100, price=100.0, equity=100_000,
        positions={"QQQ": 250}, prices={"QQQ": 100.0, "SPY": 100.0},
    )
    assert qty == 0


def test_correlation_allows_uncorrelated(corr_matrix):
    lim = CorrelationLimiter(corr_matrix, corr_threshold=0.7, max_cluster_pct=0.30)
    qty, reasons = lim.cap_quantity(
        "GLD", proposed_qty=100, price=100.0, equity=100_000,
        positions={"QQQ": 250}, prices={"QQQ": 100.0, "GLD": 100.0},
    )
    assert qty == 100
    assert reasons == []


def test_correlation_noop_without_matrix():
    lim = CorrelationLimiter(corr=None)
    qty, reasons = lim.cap_quantity(
        "SPY", 500, 100.0, 100_000, {}, {"SPY": 100.0},
    )
    assert qty == 500 and reasons == []


# ── RiskManager integration ───────────────────────────────────────────────────

def test_risk_manager_approves_reasonable_order():
    rm = RiskManager(max_position_pct=0.20)
    rm.update_equity(100_000)
    d = rm.evaluate("SPY", 100, 100.0, 100_000, {}, {"SPY": 100.0})
    assert d.approved
    assert d.approved_quantity == 100


def test_risk_manager_reduces_oversized_order():
    rm = RiskManager(max_position_pct=0.10)
    rm.update_equity(100_000)
    d = rm.evaluate("SPY", 1000, 100.0, 100_000, {}, {"SPY": 100.0})
    assert d.approved
    assert d.approved_quantity == pytest.approx(100)
    assert d.was_reduced


def test_risk_manager_halts_new_orders_on_drawdown():
    rm = RiskManager(max_drawdown_pct=0.15)
    rm.update_equity(100_000)
    rm.update_equity(80_000)            # -20% → halted
    assert rm.is_halted
    d = rm.evaluate("SPY", 100, 100.0, 80_000, {}, {"SPY": 100.0})
    assert not d.approved
    assert "halted" in d.reasons[0]


def test_risk_manager_allows_closing_while_halted():
    rm = RiskManager(max_drawdown_pct=0.15)
    rm.update_equity(100_000)
    rm.update_equity(80_000)
    assert rm.is_halted
    # Selling an existing long reduces risk — must still be permitted
    d = rm.evaluate("SPY", -100, 100.0, 80_000, {"SPY": 100}, {"SPY": 100.0})
    assert d.approved
    assert d.approved_quantity == -100


def test_risk_manager_kelly_size_with_returns():
    rm = RiskManager(kelly_fraction_scale=0.25, max_position_pct=1.0)
    np.random.seed(1)
    rets = pd.Series(np.random.randn(500) * 0.01 + 0.003)
    shares = rm.kelly_size(equity=100_000, price=100.0, returns=rets)
    assert shares > 0


def test_risk_manager_kelly_size_scales_with_signal_strength():
    rm = RiskManager(kelly_fraction_scale=0.25, max_position_pct=1.0)
    full = rm.kelly_size(100_000, 100.0, win_rate=0.6, avg_win=2.0, avg_loss=1.0,
                         signal_strength=1.0)
    half = rm.kelly_size(100_000, 100.0, win_rate=0.6, avg_win=2.0, avg_loss=1.0,
                         signal_strength=0.5)
    assert half == pytest.approx(full / 2)


def test_risk_manager_kelly_size_respects_position_cap():
    rm = RiskManager(kelly_fraction_scale=1.0, max_position_pct=0.10)
    shares = rm.kelly_size(100_000, 100.0, win_rate=0.99, avg_win=10.0, avg_loss=1.0)
    assert shares * 100.0 <= 10_000 + 1e-6


def test_risk_manager_decision_log():
    rm = RiskManager()
    rm.update_equity(100_000)
    rm.evaluate("SPY", 100, 100.0, 100_000, {}, {"SPY": 100.0})
    rm.evaluate("QQQ", 50, 200.0, 100_000, {}, {"QQQ": 200.0})
    log = rm.decision_log()
    assert len(log) == 2
    assert set(log.columns) >= {"symbol", "approved", "approved_quantity"}


def test_risk_manager_summary():
    rm = RiskManager(max_position_pct=0.10)
    rm.update_equity(100_000)
    rm.evaluate("SPY", 100, 100.0, 100_000, {}, {"SPY": 100.0})
    s = rm.summary()
    assert s["n_decisions"] == 1
    assert 0.0 <= s["approval_rate"] <= 1.0
    assert "is_halted" in s


def test_risk_manager_reset():
    rm = RiskManager()
    rm.update_equity(100_000)
    rm.update_equity(50_000)
    rm.evaluate("SPY", 100, 100.0, 50_000, {}, {"SPY": 100.0})
    rm.reset()
    assert rm.decisions == []
    assert not rm.is_halted


def test_risk_manager_from_config():
    from config.settings import Settings
    s = Settings.from_yaml()
    rm = RiskManager.from_config(s.risk)
    assert rm.exposure_limits.max_position_pct == s.risk.max_position_pct
    assert rm.drawdown_guard.max_drawdown_pct == s.risk.max_drawdown_pct


def test_risk_decision_str():
    d = RiskDecision("SPY", True, 1000, 200, ["position cap"])
    text = str(d)
    assert "APPROVED" in text and "SPY" in text


# ── Backtest integration ──────────────────────────────────────────────────────

def _make_ohlcv(n=300, seed=42, trend=0.0003) -> pd.DataFrame:
    np.random.seed(seed)
    lr = np.random.randn(n) * 0.01 + trend
    close = 100.0 * np.exp(np.cumsum(lr))
    high = close * (1 + np.abs(np.random.randn(n) * 0.003))
    low = close * (1 - np.abs(np.random.randn(n) * 0.003))
    open_ = np.roll(close, 1); open_[0] = close[0]
    vol = np.random.randint(1_000_000, 5_000_000, n).astype(float)
    idx = pd.date_range("2022-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_backtest_with_risk_manager_runs():
    from src.trading.backtest.engine import run_backtest
    from src.trading.strategy.technical import SMACrossover

    data = {"SPY": _make_ohlcv()}
    rm = RiskManager(max_position_pct=0.10, max_drawdown_pct=0.15)
    result = run_backtest(
        data, SMACrossover(["SPY"], fast=10, slow=30),
        initial_capital=100_000.0, risk_manager=rm,
    )
    assert result.n_bars > 0
    assert rm.summary()["n_decisions"] >= 0


def test_risk_manager_constrains_position_size_in_backtest():
    """A tight position cap must produce smaller fills than a loose one."""
    from src.trading.backtest.engine import run_backtest
    from src.trading.strategy.technical import SMACrossover

    data = {"SPY": _make_ohlcv(trend=0.001)}

    tight = run_backtest(
        data, SMACrossover(["SPY"], fast=10, slow=30),
        initial_capital=100_000.0,
        risk_manager=RiskManager(max_position_pct=0.05),
    )
    loose = run_backtest(
        data, SMACrossover(["SPY"], fast=10, slow=30),
        initial_capital=100_000.0,
        risk_manager=RiskManager(max_position_pct=0.50),
    )

    if not tight.trade_log.empty and not loose.trade_log.empty:
        tight_max = tight.trade_log["quantity"].abs().max()
        loose_max = loose.trade_log["quantity"].abs().max()
        assert tight_max < loose_max
