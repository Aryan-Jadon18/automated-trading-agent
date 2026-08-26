"""
Live trading orchestrator — the loop that ties every tier together.

Per cycle, per symbol:

    fetch bars → new-bar check → Strategy.on_bar() → SignalEvent
      → size → RiskManager.evaluate() → Order → Broker.submit_order()
      → record → persist state

Safety properties this loop is built around:

* **Idempotent bars.** A bar is acted on at most once. `TradingState.last_bar_time`
  is the key, and it is persisted after every cycle, so a restart mid-session does
  not re-submit orders for a bar already traded.
* **Health gates before orders.** A failing CRITICAL check aborts the cycle before
  anything is submitted.
* **Risk is not bypassable.** Every order goes through `RiskManager.evaluate()`.
* **Graceful shutdown.** SIGINT/SIGTERM finish the current cycle, then optionally
  cancel working orders and always persist state.
"""
from __future__ import annotations

import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.trading.backtest.events import Direction, MarketEvent, SignalEvent
from src.trading.execution.broker import BrokerBase
from src.trading.execution.orders import Order, OrderStatus, market_order
from src.trading.live.monitor import (
    Monitor,
    Severity,
    broker_connected_check,
    drawdown_check,
    position_reconciliation_check,
)
from src.trading.live.state import TradingState
from src.trading.risk.manager import RiskManager
from src.trading.strategy.base import Strategy
from src.trading.utils.logging import get_logger

log = get_logger(__name__)

#: Given a symbol, return its OHLCV history (newest bar last).
DataProvider = Callable[[str], pd.DataFrame]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class CycleResult:
    """What one pass over all symbols did."""
    timestamp: datetime = field(default_factory=_utcnow)
    bars_processed: int = 0
    signals: list[SignalEvent] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    healthy: bool = True
    halted: bool = False

    @property
    def acted(self) -> bool:
        return bool(self.orders)

    def __str__(self) -> str:
        if self.halted:
            return "cycle: HALTED"
        if not self.healthy:
            return f"cycle: UNHEALTHY ({'; '.join(self.errors)})"
        return (
            f"cycle: {self.bars_processed} bars, {len(self.signals)} signals, "
            f"{len(self.orders)} orders, {len(self.blocked)} blocked"
        )


class LiveTrader:
    """
    Runs a strategy against a live (or paper) broker.

    Construction wires the pieces; `run_once()` executes a single cycle and is the
    unit under test. `run_forever()` schedules it.
    """

    def __init__(
        self,
        symbols: list[str],
        strategy: Strategy,
        broker: BrokerBase,
        data_provider: DataProvider,
        risk_manager: RiskManager | None = None,
        state: TradingState | None = None,
        monitor: Monitor | None = None,
        state_path: Path | str = "data/live_state.json",
        max_drawdown_pct: float = 0.15,
        dry_run: bool = False,
        cancel_orders_on_shutdown: bool = True,
    ) -> None:
        self.symbols = symbols
        self.strategy = strategy
        self.broker = broker
        self.data_provider = data_provider
        self.risk_manager = risk_manager or RiskManager(max_drawdown_pct=max_drawdown_pct)
        self.state = state or TradingState()
        self.monitor = monitor or Monitor()
        self.state_path = Path(state_path)
        self.max_drawdown_pct = max_drawdown_pct
        self.dry_run = dry_run
        self.cancel_orders_on_shutdown = cancel_orders_on_shutdown

        self._shutdown_requested = False
        self._latest_prices: dict[str, float] = {}
        self._register_default_checks()

    # ── Setup ────────────────────────────────────────────────────────────────

    def _register_default_checks(self) -> None:
        self.monitor.register_check(
            "broker_connected", broker_connected_check(self.broker), Severity.CRITICAL
        )
        self.monitor.register_check(
            "position_reconciliation",
            position_reconciliation_check(self.broker),
            Severity.CRITICAL,
        )
        self.monitor.register_check(
            "drawdown", drawdown_check(self.state, self.max_drawdown_pct), Severity.CRITICAL
        )

    @classmethod
    def from_state_file(cls, state_path: Path | str, **kwargs: Any) -> "LiveTrader":
        """Resume a previous session, or start fresh if there is no state file."""
        return cls(state=TradingState.load_or_new(state_path), state_path=state_path, **kwargs)

    # ── Startup ──────────────────────────────────────────────────────────────

    def warmup(self) -> int:
        """
        Replay recent history through the strategy without trading it.

        A freshly-started process has no bar history, so a strategy needing N bars
        is blind for its first N cycles — and worse, it can miss a crossing that
        happened just before startup and then never fire at all. Replaying history
        gives the strategy the same context it would have had if it never stopped.

        Replayed bars are marked processed so the live loop resumes from the next
        genuinely new bar. Returns the number of bars replayed.
        """
        replayed = 0
        for symbol in self.symbols:
            try:
                bars = self.data_provider(symbol)
            except Exception as exc:
                log.warning("warmup_fetch_failed", symbol=symbol, error=str(exc))
                continue
            if bars is None or bars.empty:
                continue

            for ts, row in bars.iterrows():
                bar_time = ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts
                if bar_time.tzinfo is None:
                    bar_time = bar_time.replace(tzinfo=timezone.utc)
                # Signals are intentionally discarded — this is context, not trading.
                self.strategy.on_bar(MarketEvent(
                    datetime=bar_time,
                    symbol=symbol,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                ))
                replayed += 1

            last_time = bars.index[-1]
            if isinstance(last_time, pd.Timestamp):
                last_time = last_time.to_pydatetime()
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=timezone.utc)
            # Only ever advance the marker, never move it backwards.
            if self.state.is_new_bar(symbol, last_time):
                self.state.mark_bar_processed(symbol, last_time)
            self._latest_prices[symbol] = float(bars.iloc[-1]["close"])

        log.info("warmup_complete", bars=replayed, symbols=len(self.symbols))
        return replayed

    def startup(self, warmup: bool = True) -> None:
        """
        Connect, then verify the broker agrees with local state before trading.

        Drift here is not recoverable automatically: it means positions moved
        outside this session, so the loop halts rather than sizing off stale data.
        """
        self.broker.connect()

        if warmup:
            self.warmup()

        account = self.broker.get_account()
        self.state.record_equity(account.equity, account.cash)
        self.risk_manager.update_equity(account.equity)

        drift = self.broker.reconcile()
        if drift:
            msg = f"position drift detected on startup: {drift}"
            self.state.halt(msg)
            self.monitor.critical("startup_drift", msg, drift=str(drift))
        else:
            self.monitor.info(
                "startup",
                f"session {self.state.session_id[:8]} ready",
                equity=account.equity,
                symbols=",".join(self.symbols),
                dry_run=self.dry_run,
            )
        self.save_state()

    # ── One cycle ────────────────────────────────────────────────────────────

    def run_once(self) -> CycleResult:
        """Execute a single trading cycle across all symbols."""
        result = CycleResult()
        self.state.cycles_run += 1

        if self.state.is_halted:
            result.halted = True
            result.errors.append(self.state.halt_reason or "halted")
            self.save_state()
            return result

        statuses = self.monitor.run_checks()
        blocking = [s for s in statuses if s.is_blocking]
        if blocking:
            result.healthy = False
            for s in blocking:
                result.errors.append(f"{s.name}: {s.detail}")
                self.monitor.critical(f"health_{s.name}", s.detail)
            self.save_state()
            return result

        try:
            account = self.broker.get_account()
            self.state.record_equity(account.equity, account.cash)
            self.risk_manager.update_equity(account.equity)
        except Exception as exc:
            result.healthy = False
            result.errors.append(f"account fetch failed: {exc}")
            self.state.record_error(str(exc))
            self.save_state()
            return result

        if self.risk_manager.is_halted:
            reason = (
                f"drawdown {abs(self.risk_manager.drawdown_guard.current_drawdown):.2%} "
                f"breached limit {self.max_drawdown_pct:.2%}"
            )
            self.state.halt(reason)
            self.monitor.critical("risk_halt", reason)
            result.halted = True
            self.save_state()
            return result

        for symbol in self.symbols:
            try:
                self._process_symbol(symbol, result)
            except Exception as exc:
                msg = f"{symbol}: {exc}"
                result.errors.append(msg)
                self.state.record_error(msg)
                self.monitor.warn(f"symbol_error_{symbol}", msg)

        self.save_state()
        return result

    def _process_symbol(self, symbol: str, result: CycleResult) -> None:
        bars = self.data_provider(symbol)
        if bars is None or bars.empty:
            result.skipped.append(f"{symbol}: no data")
            return

        latest = bars.iloc[-1]
        bar_time = bars.index[-1]
        if isinstance(bar_time, pd.Timestamp):
            bar_time = bar_time.to_pydatetime()
        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=timezone.utc)

        # Idempotency: never act on a bar this session already traded.
        if not self.state.is_new_bar(symbol, bar_time):
            result.skipped.append(f"{symbol}: bar already processed")
            return

        price = float(latest["close"])
        self._latest_prices[symbol] = price

        event = MarketEvent(
            datetime=bar_time,
            symbol=symbol,
            open=float(latest["open"]),
            high=float(latest["high"]),
            low=float(latest["low"]),
            close=price,
            volume=float(latest["volume"]),
        )

        signal = self.strategy.on_bar(event)
        self.state.mark_bar_processed(symbol, bar_time)
        result.bars_processed += 1

        if signal is None:
            return

        result.signals.append(signal)
        self.state.signals_generated += 1

        order = self._signal_to_order(signal, price)
        if order is not None:
            result.orders.append(order)
        else:
            result.blocked.append(symbol)

    def _signal_to_order(self, signal: SignalEvent, price: float) -> Optional[Order]:
        """Size a signal, gate it through risk, and submit it."""
        positions = {s: p.quantity for s, p in self.broker.get_positions().items()}
        account = self.broker.get_account()
        current_qty = positions.get(signal.symbol, 0.0)

        if signal.direction == Direction.EXIT:
            if current_qty == 0:
                return None
            proposed = -current_qty
        elif signal.direction == Direction.LONG:
            target = self.risk_manager.kelly_size(
                equity=account.equity, price=price, signal_strength=signal.strength
            )
            proposed = target - current_qty
        elif signal.direction == Direction.SHORT:
            target = -self.risk_manager.kelly_size(
                equity=account.equity, price=price, signal_strength=signal.strength
            )
            proposed = target - current_qty
        else:
            return None

        if abs(proposed) < 1:
            return None

        decision = self.risk_manager.evaluate(
            symbol=signal.symbol,
            proposed_qty=proposed,
            price=price,
            equity=account.equity,
            positions=positions,
            prices=self._latest_prices,
        )

        if not decision.approved:
            self.state.signals_blocked_by_risk += 1
            log.info(
                "order_blocked",
                symbol=signal.symbol,
                reasons="; ".join(decision.reasons),
            )
            return None

        qty = int(decision.approved_quantity)
        if qty == 0:
            self.state.signals_blocked_by_risk += 1
            return None

        if self.dry_run:
            log.info("dry_run_order", symbol=signal.symbol, quantity=qty, price=price)
            return market_order(signal.symbol, qty)

        order = self.broker.submit_order(market_order(signal.symbol, qty))
        self.state.orders_submitted += 1

        if order.status == OrderStatus.FILLED:
            self.state.orders_filled += 1
        elif order.status == OrderStatus.REJECTED:
            self.state.orders_rejected += 1
            self.monitor.warn(
                f"order_rejected_{signal.symbol}",
                order.reject_reason or "rejected",
                symbol=signal.symbol,
            )
        return order

    # ── Scheduling ───────────────────────────────────────────────────────────

    def run_forever(
        self,
        interval: timedelta = timedelta(minutes=5),
        max_cycles: int | None = None,
        market_hours_only: bool = True,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> list[CycleResult]:
        """
        Loop until shutdown is requested or `max_cycles` is reached.

        `max_cycles` and `sleep_fn` exist so tests can drive this deterministically
        without real time passing.
        """
        self._install_signal_handlers()
        results: list[CycleResult] = []
        cycles = 0

        while not self._shutdown_requested:
            if max_cycles is not None and cycles >= max_cycles:
                break

            if market_hours_only and not self._market_is_open():
                log.debug("market_closed_sleeping")
                sleep_fn(interval.total_seconds())
                cycles += 1
                continue

            result = self.run_once()
            results.append(result)
            cycles += 1
            log.info("cycle_complete", summary=str(result))

            if result.halted:
                self.monitor.critical(
                    "loop_halted", self.state.halt_reason or "halted"
                )
                break

            if max_cycles is None or cycles < max_cycles:
                sleep_fn(interval.total_seconds())

        self.shutdown()
        return results

    def _market_is_open(self) -> bool:
        try:
            return self.broker.is_market_open()
        except Exception as exc:
            log.warning("market_clock_failed", error=str(exc))
            return False

    # ── Shutdown ─────────────────────────────────────────────────────────────

    def _install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            log.info("shutdown_signal", signal=signum)
            self._shutdown_requested = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                # Not on the main thread (common under pytest) — skip silently.
                pass

    def request_shutdown(self) -> None:
        self._shutdown_requested = True

    def shutdown(self) -> None:
        """Cancel working orders if configured, then always persist state."""
        if self.cancel_orders_on_shutdown and not self.dry_run:
            try:
                cancelled = self.broker.cancel_all()
                if cancelled:
                    log.info("orders_cancelled", count=len(cancelled))
            except Exception as exc:
                log.warning("cancel_all_failed", error=str(exc))

        self.save_state()
        self.monitor.info("shutdown", "session stopped", **{
            k: v for k, v in self.state.summary().items()
            if k in ("cycles_run", "orders_filled", "errors")
        })

    # ── State ────────────────────────────────────────────────────────────────

    def save_state(self) -> None:
        try:
            self.state.save(self.state_path)
        except Exception as exc:
            # Never let a persistence failure kill a running session.
            log.error("state_save_failed", error=str(exc))

    def status(self) -> dict:
        return {
            **self.state.summary(),
            "monitor": self.monitor.summary(),
            "risk": self.risk_manager.summary(),
            "dry_run": self.dry_run,
        }
