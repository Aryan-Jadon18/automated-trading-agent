"""
Health checks and alerting for the live loop.

`HealthCheck`s are cheap predicates run before each trading cycle; any CRITICAL
failure stops the cycle before an order is placed. Alerts are dispatched to
pluggable sinks (logging by default) so a deployment can add email/Slack/PagerDuty
without touching the trading path.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

from src.trading.utils.logging import get_logger

log = get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    severity: Severity
    title: str
    message: str
    timestamp: datetime = field(default_factory=_utcnow)
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.title}: {self.message}"


@dataclass
class HealthStatus:
    name: str
    healthy: bool
    detail: str = ""
    severity: Severity = Severity.WARNING

    @property
    def is_blocking(self) -> bool:
        """A failed CRITICAL check must stop the cycle."""
        return not self.healthy and self.severity == Severity.CRITICAL


#: A check returns (healthy, detail).
CheckFn = Callable[[], "tuple[bool, str]"]

#: An alert sink receives every dispatched alert.
AlertSink = Callable[[Alert], None]


def log_sink(alert: Alert) -> None:
    """Default sink: route alerts into structured logs by severity."""
    fn = {
        Severity.INFO: log.info,
        Severity.WARNING: log.warning,
        Severity.CRITICAL: log.error,
    }[alert.severity]
    fn("alert", title=alert.title, message=alert.message, **alert.context)


class Monitor:
    """
    Runs registered health checks and fans alerts out to sinks.

    Repeat alerts for the same title are suppressed for `dedupe_window` so a
    persistent failure does not flood the sinks every cycle.
    """

    def __init__(
        self,
        sinks: list[AlertSink] | None = None,
        dedupe_window: timedelta = timedelta(minutes=15),
    ) -> None:
        self.sinks: list[AlertSink] = sinks if sinks is not None else [log_sink]
        self.dedupe_window = dedupe_window
        self._checks: dict[str, tuple[CheckFn, Severity]] = {}
        self._last_alert_at: dict[str, datetime] = {}
        self.alerts: list[Alert] = []

    # ── Checks ───────────────────────────────────────────────────────────────

    def register_check(
        self, name: str, fn: CheckFn, severity: Severity = Severity.WARNING
    ) -> None:
        self._checks[name] = (fn, severity)

    def run_checks(self) -> list[HealthStatus]:
        results: list[HealthStatus] = []
        for name, (fn, severity) in self._checks.items():
            try:
                healthy, detail = fn()
            except Exception as exc:
                # A check that throws is itself a failure — never let it escape
                # into the trading path.
                healthy, detail = False, f"check raised: {exc}"
            results.append(HealthStatus(name, healthy, detail, severity))
        return results

    def is_healthy(self, statuses: list[HealthStatus] | None = None) -> bool:
        """True when no CRITICAL check is failing."""
        statuses = statuses if statuses is not None else self.run_checks()
        return not any(s.is_blocking for s in statuses)

    # ── Alerts ───────────────────────────────────────────────────────────────

    def alert(
        self,
        severity: Severity,
        title: str,
        message: str,
        force: bool = False,
        **context: Any,
    ) -> Optional[Alert]:
        """
        Dispatch an alert. Returns None when suppressed by the dedupe window.
        `force=True` bypasses deduplication.
        """
        now = _utcnow()
        if not force:
            last = self._last_alert_at.get(title)
            if last is not None and now - last < self.dedupe_window:
                return None

        a = Alert(severity=severity, title=title, message=message, context=context)
        self._last_alert_at[title] = now
        self.alerts.append(a)

        for sink in self.sinks:
            try:
                sink(a)
            except Exception as exc:
                # A broken sink must never take down the trading loop.
                log.error("alert_sink_failed", error=str(exc))
        return a

    def info(self, title: str, message: str, **ctx: Any) -> Optional[Alert]:
        return self.alert(Severity.INFO, title, message, **ctx)

    def warn(self, title: str, message: str, **ctx: Any) -> Optional[Alert]:
        return self.alert(Severity.WARNING, title, message, **ctx)

    def critical(self, title: str, message: str, **ctx: Any) -> Optional[Alert]:
        # Critical alerts always fire — suppressing them is never the right call.
        return self.alert(Severity.CRITICAL, title, message, force=True, **ctx)

    # ── Reporting ────────────────────────────────────────────────────────────

    def recent_alerts(self, severity: Severity | None = None, limit: int = 20) -> list[Alert]:
        out = [a for a in self.alerts if severity is None or a.severity == severity]
        return out[-limit:]

    def summary(self) -> dict:
        statuses = self.run_checks()
        return {
            "healthy": self.is_healthy(statuses),
            "n_checks": len(statuses),
            "n_failing": sum(1 for s in statuses if not s.healthy),
            "failing": [s.name for s in statuses if not s.healthy],
            "n_alerts": len(self.alerts),
            "n_critical": sum(1 for a in self.alerts if a.severity == Severity.CRITICAL),
        }


# ── Standard checks ───────────────────────────────────────────────────────────

def broker_connected_check(broker) -> CheckFn:
    """CRITICAL: the broker answers and the account is tradable."""
    def check() -> tuple[bool, str]:
        account = broker.get_account()
        if not account.is_tradable:
            return False, "account is blocked from trading"
        return True, f"equity={account.equity:,.2f}"
    return check


def data_freshness_check(state, symbol: str, max_age: timedelta) -> CheckFn:
    """WARNING: the newest bar acted on is not stale."""
    def check() -> tuple[bool, str]:
        last = state.last_bar_time.get(symbol)
        if last is None:
            return True, "no bars processed yet"
        age = _utcnow() - last
        if age > max_age:
            return False, f"last {symbol} bar is {age} old"
        return True, f"last {symbol} bar {age} ago"
    return check


def drawdown_check(state, max_drawdown_pct: float) -> CheckFn:
    """CRITICAL: drawdown has not breached the configured limit."""
    def check() -> tuple[bool, str]:
        dd = abs(state.current_drawdown)
        if dd >= max_drawdown_pct:
            return False, f"drawdown {dd:.2%} >= limit {max_drawdown_pct:.2%}"
        return True, f"drawdown {dd:.2%}"
    return check


def position_reconciliation_check(broker) -> CheckFn:
    """
    CRITICAL: locally-tracked fills agree with the broker's positions.

    Drift means orders were filled or placed outside this session, so any sizing
    decision made from local state would be wrong.
    """
    def check() -> tuple[bool, str]:
        drift = broker.reconcile()
        if drift:
            return False, f"position drift: {drift}"
        return True, "positions reconciled"
    return check
