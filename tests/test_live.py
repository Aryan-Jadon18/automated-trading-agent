"""Tests for Tier 8: durable state, monitoring, and the live orchestrator."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.trading.backtest.events import Direction
from src.trading.execution.broker import MockBroker
from src.trading.execution.orders import OrderStatus, market_order
from src.trading.live.monitor import (
    Alert,
    HealthStatus,
    Monitor,
    Severity,
    broker_connected_check,
    data_freshness_check,
    drawdown_check,
    position_reconciliation_check,
)
from src.trading.live.orchestrator import CycleResult, LiveTrader
from src.trading.live.state import EquityPoint, StateError, TradingState
from src.trading.risk.manager import RiskManager
from src.trading.strategy.base import Strategy
from src.trading.strategy.technical import SMACrossover


UTC = timezone.utc


# ══ TradingState ══════════════════════════════════════════════════════════════

def test_state_has_unique_session_ids():
    assert TradingState().session_id != TradingState().session_id


def test_state_new_bar_when_never_seen():
    s = TradingState()
    assert s.is_new_bar("SPY", datetime(2024, 1, 1, tzinfo=UTC))


def test_state_marks_bar_processed():
    s = TradingState()
    t = datetime(2024, 1, 1, tzinfo=UTC)
    s.mark_bar_processed("SPY", t)
    assert not s.is_new_bar("SPY", t)


def test_state_older_bar_is_not_new():
    s = TradingState()
    s.mark_bar_processed("SPY", datetime(2024, 1, 5, tzinfo=UTC))
    assert not s.is_new_bar("SPY", datetime(2024, 1, 4, tzinfo=UTC))


def test_state_newer_bar_is_new():
    s = TradingState()
    s.mark_bar_processed("SPY", datetime(2024, 1, 5, tzinfo=UTC))
    assert s.is_new_bar("SPY", datetime(2024, 1, 6, tzinfo=UTC))


def test_state_bars_tracked_per_symbol():
    s = TradingState()
    s.mark_bar_processed("SPY", datetime(2024, 1, 5, tzinfo=UTC))
    assert s.is_new_bar("QQQ", datetime(2024, 1, 1, tzinfo=UTC))


def test_state_records_equity_and_peak():
    s = TradingState()
    s.record_equity(100_000, 50_000)
    s.record_equity(120_000, 50_000)
    s.record_equity(110_000, 50_000)
    assert s.peak_equity == 120_000
    assert s.current_equity == 110_000


def test_state_drawdown_is_negative_below_peak():
    s = TradingState()
    s.record_equity(100_000, 0)
    s.record_equity(90_000, 0)
    assert s.current_drawdown == pytest.approx(-0.10)


def test_state_drawdown_zero_at_peak():
    s = TradingState()
    s.record_equity(100_000, 0)
    assert s.current_drawdown == 0.0


def test_state_drawdown_zero_with_no_history():
    assert TradingState().current_drawdown == 0.0


def test_state_equity_curve_frame():
    s = TradingState()
    for i, eq in enumerate([100_000, 101_000, 99_000]):
        s.record_equity(eq, 0, timestamp=datetime(2024, 1, 1 + i, tzinfo=UTC))
    df = s.equity_curve()
    assert len(df) == 3
    assert "returns" in df.columns


def test_state_empty_equity_curve_has_columns():
    assert list(TradingState().equity_curve().columns) == ["equity", "cash", "returns"]


def test_state_halt_and_resume():
    s = TradingState()
    s.halt("drawdown breach")
    assert s.is_halted and s.halt_reason == "drawdown breach"
    s.resume()
    assert not s.is_halted and s.halt_reason is None


def test_state_records_errors():
    s = TradingState()
    s.record_error("boom")
    assert s.errors == 1 and s.last_error == "boom"


def test_state_roundtrip_to_dict(tmp_path):
    s = TradingState()
    s.mark_bar_processed("SPY", datetime(2024, 3, 1, tzinfo=UTC))
    s.record_equity(105_000, 5_000)
    s.cycles_run = 7
    restored = TradingState.from_dict(s.to_dict())
    assert restored.session_id == s.session_id
    assert restored.cycles_run == 7
    assert restored.last_bar_time["SPY"] == datetime(2024, 3, 1, tzinfo=UTC)
    assert restored.peak_equity == 105_000


def test_state_save_and_load(tmp_path):
    path = tmp_path / "state.json"
    s = TradingState()
    s.mark_bar_processed("SPY", datetime(2024, 3, 1, tzinfo=UTC))
    s.orders_filled = 3
    s.save(path)

    loaded = TradingState.load(path)
    assert loaded.orders_filled == 3
    assert loaded.last_bar_time["SPY"] == datetime(2024, 3, 1, tzinfo=UTC)


def test_state_save_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "deep" / "state.json"
    TradingState().save(path)
    assert path.exists()


def test_state_save_is_atomic_no_temp_left(tmp_path):
    path = tmp_path / "state.json"
    TradingState().save(path)
    assert list(tmp_path.glob("*.tmp")) == []


def test_state_save_overwrites_cleanly(tmp_path):
    path = tmp_path / "state.json"
    s = TradingState()
    s.save(path)
    s.cycles_run = 42
    s.save(path)
    assert TradingState.load(path).cycles_run == 42


def test_state_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TradingState.load(tmp_path / "nope.json")


def test_state_load_corrupt_file_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(StateError):
        TradingState.load(path)


def test_state_load_or_new_returns_fresh_when_missing(tmp_path):
    assert TradingState.load_or_new(tmp_path / "nope.json").cycles_run == 0


def test_state_load_or_new_does_not_swallow_corruption(tmp_path):
    """Corrupt state must raise, not silently reset — resetting risks double-trading."""
    path = tmp_path / "bad.json"
    path.write_text("{corrupt")
    with pytest.raises(StateError):
        TradingState.load_or_new(path)


def test_state_rejects_newer_schema(tmp_path):
    path = tmp_path / "future.json"
    payload = TradingState().to_dict()
    payload["schema_version"] = TradingState.SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload))
    with pytest.raises(StateError, match="newer version"):
        TradingState.load(path)


def test_state_summary_keys():
    s = TradingState().summary()
    for key in ("session_id", "cycles_run", "is_halted", "current_drawdown"):
        assert key in s


# ══ Monitor ═══════════════════════════════════════════════════════════════════

def test_monitor_healthy_with_no_checks():
    assert Monitor(sinks=[]).is_healthy()


def test_monitor_passing_check():
    m = Monitor(sinks=[])
    m.register_check("ok", lambda: (True, "fine"))
    statuses = m.run_checks()
    assert statuses[0].healthy and m.is_healthy(statuses)


def test_monitor_critical_failure_blocks():
    m = Monitor(sinks=[])
    m.register_check("bad", lambda: (False, "down"), Severity.CRITICAL)
    assert not m.is_healthy()


def test_monitor_warning_failure_does_not_block():
    m = Monitor(sinks=[])
    m.register_check("meh", lambda: (False, "slow"), Severity.WARNING)
    assert m.is_healthy()


def test_monitor_check_that_raises_is_unhealthy_not_fatal():
    m = Monitor(sinks=[])

    def boom():
        raise RuntimeError("kaboom")

    m.register_check("explodes", boom, Severity.CRITICAL)
    statuses = m.run_checks()
    assert not statuses[0].healthy
    assert "kaboom" in statuses[0].detail


def test_monitor_dispatches_to_sink():
    seen: list[Alert] = []
    m = Monitor(sinks=[seen.append])
    m.warn("t", "message")
    assert len(seen) == 1 and seen[0].severity == Severity.WARNING


def test_monitor_dedupes_repeat_alerts():
    seen: list[Alert] = []
    m = Monitor(sinks=[seen.append], dedupe_window=timedelta(minutes=10))
    assert m.warn("same", "first") is not None
    assert m.warn("same", "second") is None
    assert len(seen) == 1


def test_monitor_dedupe_is_per_title():
    seen: list[Alert] = []
    m = Monitor(sinks=[seen.append])
    m.warn("a", "x")
    m.warn("b", "y")
    assert len(seen) == 2


def test_monitor_critical_bypasses_dedupe():
    """Suppressing a repeated critical alert is never correct."""
    seen: list[Alert] = []
    m = Monitor(sinks=[seen.append], dedupe_window=timedelta(hours=1))
    m.critical("halt", "one")
    m.critical("halt", "two")
    assert len(seen) == 2


def test_monitor_broken_sink_does_not_propagate():
    def bad_sink(_alert):
        raise RuntimeError("sink down")

    m = Monitor(sinks=[bad_sink])
    assert m.warn("t", "m") is not None       # no exception escapes


def test_monitor_recent_alerts_filters_by_severity():
    m = Monitor(sinks=[])
    m.info("i", "x")
    m.critical("c", "y")
    assert len(m.recent_alerts(Severity.CRITICAL)) == 1


def test_monitor_summary_reports_failing():
    m = Monitor(sinks=[])
    m.register_check("ok", lambda: (True, ""))
    m.register_check("bad", lambda: (False, "nope"), Severity.CRITICAL)
    s = m.summary()
    assert s["n_checks"] == 2 and s["failing"] == ["bad"] and not s["healthy"]


def test_health_status_blocking_only_when_critical():
    assert HealthStatus("x", False, "", Severity.CRITICAL).is_blocking
    assert not HealthStatus("x", False, "", Severity.WARNING).is_blocking
    assert not HealthStatus("x", True, "", Severity.CRITICAL).is_blocking


# ── Standard checks ───────────────────────────────────────────────────────────

def test_broker_connected_check_passes():
    b = MockBroker(prices={"SPY": 100.0})
    b.connect()
    healthy, _ = broker_connected_check(b)()
    assert healthy


def test_drawdown_check_fails_on_breach():
    s = TradingState()
    s.record_equity(100_000, 0)
    s.record_equity(80_000, 0)
    healthy, detail = drawdown_check(s, 0.15)()
    assert not healthy and "20" in detail


def test_drawdown_check_passes_within_limit():
    s = TradingState()
    s.record_equity(100_000, 0)
    s.record_equity(95_000, 0)
    assert drawdown_check(s, 0.15)()[0]


def test_data_freshness_check_detects_stale():
    s = TradingState()
    s.mark_bar_processed("SPY", datetime.now(UTC) - timedelta(days=3))
    assert not data_freshness_check(s, "SPY", timedelta(hours=1))()[0]


def test_data_freshness_check_passes_when_fresh():
    s = TradingState()
    s.mark_bar_processed("SPY", datetime.now(UTC))
    assert data_freshness_check(s, "SPY", timedelta(hours=1))()[0]


def test_data_freshness_check_ok_before_first_bar():
    assert data_freshness_check(TradingState(), "SPY", timedelta(hours=1))()[0]


def test_reconciliation_check_clean():
    b = MockBroker(prices={"SPY": 100.0})
    b.connect()
    b.submit_order(market_order("SPY", 10))
    assert position_reconciliation_check(b)()[0]


def test_reconciliation_check_detects_drift():
    b = MockBroker(prices={"SPY": 100.0})
    b.connect()
    # A position appearing without a tracked order — as if filled elsewhere.
    from src.trading.execution.broker import BrokerPosition
    b._positions["SPY"] = BrokerPosition("SPY", 50, 100.0, 100.0)
    healthy, detail = position_reconciliation_check(b)()
    assert not healthy and "drift" in detail


# ══ LiveTrader ════════════════════════════════════════════════════════════════

def _series(n: int = 260, seed: int = 3) -> pd.DataFrame:
    """Oscillating price series so SMA crossovers actually occur."""
    np.random.seed(seed)
    t = np.arange(n)
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.012)) * (1 + 0.08 * np.sin(t / 12))
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1e6},
        index=idx,
    )


class _Cursor:
    """Feeds bars incrementally, imitating a live data source."""

    def __init__(self, bars: pd.DataFrame, start: int = 60) -> None:
        self.bars = bars
        self.i = start

    def __call__(self, symbol: str) -> pd.DataFrame:
        return self.bars.iloc[: self.i]

    def advance(self) -> float:
        self.i += 1
        return float(self.bars.iloc[self.i - 1]["close"])


@pytest.fixture
def bars() -> pd.DataFrame:
    return _series()


@pytest.fixture
def trader(bars, tmp_path):
    cursor = _Cursor(bars)
    broker = MockBroker(initial_cash=100_000.0,
                        prices={"SPY": float(bars.iloc[59]["close"])})
    lt = LiveTrader(
        symbols=["SPY"],
        strategy=SMACrossover(["SPY"], fast=5, slow=20),
        broker=broker,
        data_provider=cursor,
        risk_manager=RiskManager(max_position_pct=0.20),
        monitor=Monitor(sinks=[]),
        state_path=tmp_path / "state.json",
    )
    lt._cursor = cursor       # test handle
    return lt


# ── Startup ───────────────────────────────────────────────────────────────────

def test_startup_connects_and_records_equity(trader):
    trader.startup()
    assert trader.broker.connected
    assert trader.state.current_equity == pytest.approx(100_000.0)


def test_startup_warms_up_strategy(trader):
    trader.startup()
    assert len(trader.strategy.get_bars("SPY")) == 60


def test_warmup_marks_bars_processed(trader):
    trader.startup()
    assert not trader.state.is_new_bar("SPY", trader._cursor.bars.index[59].to_pydatetime())


def test_warmup_can_be_skipped(trader):
    trader.startup(warmup=False)
    assert trader.strategy.get_bars("SPY").empty


def test_startup_halts_on_position_drift(trader):
    from src.trading.execution.broker import BrokerPosition
    trader.broker._positions["SPY"] = BrokerPosition("SPY", 99, 100.0, 100.0)
    trader.startup()
    assert trader.state.is_halted
    assert "drift" in trader.state.halt_reason


def test_startup_persists_state(trader):
    trader.startup()
    assert trader.state_path.exists()


# ── Cycles ────────────────────────────────────────────────────────────────────

def test_run_once_increments_cycle_count(trader):
    trader.startup()
    trader.run_once()
    assert trader.state.cycles_run == 1


def test_run_once_skips_already_processed_bar(trader):
    trader.startup()
    result = trader.run_once()          # no new bar since warm-up
    assert result.bars_processed == 0
    assert any("already processed" in s for s in result.skipped)


def test_run_once_processes_new_bar(trader):
    trader.startup()
    trader._cursor.advance()
    assert trader.run_once().bars_processed == 1


def test_run_once_is_idempotent_per_bar(trader):
    """Re-running without new data must not re-submit orders."""
    trader.startup()
    trader._cursor.advance()
    trader.run_once()
    submitted = trader.state.orders_submitted
    trader.run_once()
    assert trader.state.orders_submitted == submitted


def test_run_once_halted_state_short_circuits(trader):
    trader.startup()
    trader.state.halt("manual")
    result = trader.run_once()
    assert result.halted and result.bars_processed == 0


def test_run_once_blocked_by_critical_health_check(trader):
    trader.startup()
    trader.monitor.register_check("boom", lambda: (False, "down"), Severity.CRITICAL)
    trader._cursor.advance()
    result = trader.run_once()
    assert not result.healthy
    assert result.bars_processed == 0        # aborted before touching the market


def test_run_once_tolerates_missing_data(trader, tmp_path):
    trader.data_provider = lambda s: pd.DataFrame()
    trader.startup()
    result = trader.run_once()
    assert result.bars_processed == 0
    assert any("no data" in s for s in result.skipped)


def test_run_once_records_provider_errors(trader):
    trader.startup()

    def broken(_symbol):
        raise RuntimeError("feed down")

    trader.data_provider = broken
    result = trader.run_once()
    assert result.errors and trader.state.errors >= 1


def test_full_session_generates_round_trips(trader):
    """Drive the whole series through and confirm real entries and exits."""
    trader.startup()
    for _ in range(199):
        price = trader._cursor.advance()
        trader.broker.set_price("SPY", price)
        trader.run_once()

    assert trader.state.signals_generated > 0
    assert trader.state.orders_filled > 0
    log = trader.broker.orders.to_frame()
    assert set(log["side"]) == {"buy", "sell"}     # both directions occurred


def test_session_respects_risk_position_cap(trader):
    trader.startup()
    for _ in range(199):
        price = trader._cursor.advance()
        trader.broker.set_price("SPY", price)
        trader.run_once()

    equity = trader.state.current_equity
    for pos in trader.broker.get_positions().values():
        assert abs(pos.market_value) <= equity * 0.20 * 1.05    # 5% price-drift tolerance


def test_risk_halt_stops_the_session(trader):
    """A real drawdown in account equity must halt the loop, not merely shrink orders."""
    trader.startup()                       # peak equity = 100k
    trader.broker.cash = 50_000.0          # account genuinely loses half its value
    trader._cursor.advance()
    result = trader.run_once()
    assert result.halted and trader.state.is_halted
    assert "drawdown" in trader.state.halt_reason


def test_risk_halt_clears_when_equity_recovers(trader):
    """
    Equity is re-synced from the broker every cycle, so the guard's hysteresis
    resume is driven by real account value — a halt cannot be faked or stick
    once the account has actually recovered.
    """
    trader.startup()
    trader.broker.cash = 50_000.0
    trader._cursor.advance()
    assert trader.run_once().halted

    trader.state.resume()                  # operator clears the halt
    trader.broker.cash = 100_000.0         # and the account is whole again
    trader._cursor.advance()
    assert not trader.run_once().halted


# ── Dry run ───────────────────────────────────────────────────────────────────

def test_dry_run_never_submits_to_broker(trader):
    trader.dry_run = True
    trader.startup()
    for _ in range(199):
        price = trader._cursor.advance()
        trader.broker.set_price("SPY", price)
        trader.run_once()
    assert len(trader.broker.orders) == 0
    assert trader.broker.get_positions() == {}


def test_dry_run_still_produces_orders_in_result(trader):
    trader.dry_run = True
    trader.startup()
    orders = []
    for _ in range(199):
        price = trader._cursor.advance()
        trader.broker.set_price("SPY", price)
        orders.extend(trader.run_once().orders)
    assert orders                     # decisions still surface for inspection


# ── Persistence across restart ────────────────────────────────────────────────

def test_state_survives_restart(trader, tmp_path):
    trader.startup()
    for _ in range(30):
        price = trader._cursor.advance()
        trader.broker.set_price("SPY", price)
        trader.run_once()

    before = trader.state.summary()
    reloaded = TradingState.load(trader.state_path)
    assert reloaded.session_id == before["session_id"]
    assert reloaded.cycles_run == before["cycles_run"]
    assert reloaded.last_bar_time["SPY"] == trader.state.last_bar_time["SPY"]


def test_restart_does_not_reprocess_bars(bars, tmp_path):
    """The whole point of persistence: a restart must not re-trade old bars."""
    path = tmp_path / "state.json"
    cursor = _Cursor(bars)
    broker = MockBroker(initial_cash=100_000.0, prices={"SPY": 100.0})

    first = LiveTrader(
        symbols=["SPY"], strategy=SMACrossover(["SPY"], fast=5, slow=20),
        broker=broker, data_provider=cursor, monitor=Monitor(sinks=[]),
        state_path=path,
    )
    first.startup()
    for _ in range(20):
        broker.set_price("SPY", cursor.advance())
        first.run_once()
    submitted_before = first.state.orders_submitted

    # New process, same state file and same broker/data.
    second = LiveTrader.from_state_file(
        path,
        symbols=["SPY"], strategy=SMACrossover(["SPY"], fast=5, slow=20),
        broker=broker, data_provider=cursor, monitor=Monitor(sinks=[]),
    )
    second.startup()
    result = second.run_once()

    assert result.bars_processed == 0
    assert second.state.orders_submitted == submitted_before


def test_from_state_file_starts_fresh_when_absent(bars, tmp_path):
    lt = LiveTrader.from_state_file(
        tmp_path / "none.json",
        symbols=["SPY"], strategy=SMACrossover(["SPY"]),
        broker=MockBroker(prices={"SPY": 100.0}),
        data_provider=_Cursor(bars), monitor=Monitor(sinks=[]),
    )
    assert lt.state.cycles_run == 0


# ── Scheduling and shutdown ───────────────────────────────────────────────────

def test_run_forever_stops_at_max_cycles(trader):
    trader.startup()
    slept: list[float] = []
    results = trader.run_forever(
        interval=timedelta(seconds=1), max_cycles=3,
        market_hours_only=False, sleep_fn=slept.append,
    )
    assert len(results) == 3


def test_run_forever_honours_shutdown_request(trader):
    trader.startup()

    def sleeper(_seconds):
        trader.request_shutdown()

    results = trader.run_forever(
        interval=timedelta(seconds=1), max_cycles=100,
        market_hours_only=False, sleep_fn=sleeper,
    )
    assert len(results) == 1


def test_run_forever_skips_cycles_when_market_closed(trader):
    trader.startup()
    trader.broker.market_open = False
    results = trader.run_forever(
        interval=timedelta(seconds=1), max_cycles=3,
        market_hours_only=True, sleep_fn=lambda _s: None,
    )
    assert results == []


def test_run_forever_breaks_on_halt(trader):
    trader.startup()
    trader.state.halt("manual stop")
    results = trader.run_forever(
        interval=timedelta(seconds=1), max_cycles=10,
        market_hours_only=False, sleep_fn=lambda _s: None,
    )
    assert len(results) == 1 and results[0].halted


def test_shutdown_cancels_open_orders(trader):
    from src.trading.execution.orders import limit_order
    trader.startup()
    resting = trader.broker.submit_order(limit_order("SPY", 10, limit_price=1.0))
    trader.shutdown()
    assert resting.status == OrderStatus.CANCELED


def test_shutdown_can_leave_orders_working(trader):
    from src.trading.execution.orders import limit_order
    trader.cancel_orders_on_shutdown = False
    trader.startup()
    resting = trader.broker.submit_order(limit_order("SPY", 10, limit_price=1.0))
    trader.shutdown()
    assert resting.status == OrderStatus.SUBMITTED


def test_shutdown_persists_state(trader, tmp_path):
    trader.startup()
    trader._cursor.advance()
    trader.run_once()
    trader.shutdown()
    assert TradingState.load(trader.state_path).cycles_run == 1


def test_save_state_failure_does_not_crash_loop(trader):
    """Persistence is best-effort — a bad path must not kill a live session."""
    trader.startup()
    trader.state_path = trader.state_path / "not-a-directory" / "x.json"
    trader._cursor.advance()
    trader.run_once()          # must not raise


def test_status_reports_all_subsystems(trader):
    trader.startup()
    s = trader.status()
    assert "monitor" in s and "risk" in s
    assert s["dry_run"] is False


def test_cycle_result_str_variants():
    assert "HALTED" in str(CycleResult(halted=True))
    assert "UNHEALTHY" in str(CycleResult(healthy=False, errors=["x"]))
    assert "bars" in str(CycleResult(bars_processed=2))
