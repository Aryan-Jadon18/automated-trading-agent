"""
Durable trading state.

The live loop can be killed at any moment — a deploy, a crash, an OOM. Whatever
it needs on restart lives here and is written to disk after every cycle:
which bars have already been acted on, the equity history, and the halt flag.

Writes are atomic (temp file + rename), so a process killed mid-write leaves the
previous good state intact rather than a truncated file.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


def _parse(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class StateError(RuntimeError):
    """Raised when state cannot be loaded or is incompatible."""


@dataclass
class EquityPoint:
    timestamp: datetime
    equity: float
    cash: float

    def to_dict(self) -> dict:
        return {"timestamp": _iso(self.timestamp), "equity": self.equity, "cash": self.cash}

    @classmethod
    def from_dict(cls, d: dict) -> "EquityPoint":
        return cls(timestamp=_parse(d["timestamp"]), equity=d["equity"], cash=d["cash"])


@dataclass
class TradingState:
    """
    Everything the live loop must remember across a restart.

    `last_bar_time` is the idempotency key: a bar whose timestamp is not strictly
    newer than the stored one has already been acted on and must be skipped, or a
    restart would duplicate every order in the current bar.
    """

    #: Bumped when the on-disk shape changes incompatibly.
    SCHEMA_VERSION = 1

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    last_bar_time: dict[str, datetime] = field(default_factory=dict)
    equity_history: list[EquityPoint] = field(default_factory=list)

    is_halted: bool = False
    halt_reason: Optional[str] = None
    peak_equity: Optional[float] = None

    cycles_run: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    signals_generated: int = 0
    signals_blocked_by_risk: int = 0
    errors: int = 0

    last_error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Idempotency ──────────────────────────────────────────────────────────

    def is_new_bar(self, symbol: str, bar_time: datetime) -> bool:
        """True when `bar_time` is strictly newer than the last bar acted on."""
        seen = self.last_bar_time.get(symbol)
        return seen is None or bar_time > seen

    def mark_bar_processed(self, symbol: str, bar_time: datetime) -> None:
        self.last_bar_time[symbol] = bar_time
        self.updated_at = _utcnow()

    # ── Equity ───────────────────────────────────────────────────────────────

    def record_equity(self, equity: float, cash: float, timestamp: datetime | None = None) -> None:
        self.equity_history.append(
            EquityPoint(timestamp=timestamp or _utcnow(), equity=equity, cash=cash)
        )
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        self.updated_at = _utcnow()

    @property
    def current_equity(self) -> Optional[float]:
        return self.equity_history[-1].equity if self.equity_history else None

    @property
    def current_drawdown(self) -> float:
        """Drawdown from peak as a negative fraction; 0.0 when at or above peak."""
        eq = self.current_equity
        if eq is None or not self.peak_equity:
            return 0.0
        return min(0.0, eq / self.peak_equity - 1.0)

    def equity_curve(self) -> pd.DataFrame:
        if not self.equity_history:
            return pd.DataFrame(columns=["equity", "cash", "returns"])
        df = pd.DataFrame([p.to_dict() for p in self.equity_history])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        df["returns"] = df["equity"].pct_change()
        return df

    # ── Halt ─────────────────────────────────────────────────────────────────

    def halt(self, reason: str) -> None:
        self.is_halted = True
        self.halt_reason = reason
        self.updated_at = _utcnow()

    def resume(self) -> None:
        self.is_halted = False
        self.halt_reason = None
        self.updated_at = _utcnow()

    def record_error(self, message: str) -> None:
        self.errors += 1
        self.last_error = message
        self.updated_at = _utcnow()

    # ── Persistence ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = asdict(self)
        d["schema_version"] = self.SCHEMA_VERSION
        d["started_at"] = _iso(self.started_at)
        d["updated_at"] = _iso(self.updated_at)
        d["last_bar_time"] = {k: _iso(v) for k, v in self.last_bar_time.items()}
        d["equity_history"] = [p.to_dict() for p in self.equity_history]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TradingState":
        version = d.get("schema_version", 0)
        if version > cls.SCHEMA_VERSION:
            raise StateError(
                f"state was written by a newer version (schema {version} > "
                f"{cls.SCHEMA_VERSION}); refusing to load it"
            )
        known = {f for f in cls.__dataclass_fields__}
        payload = {k: v for k, v in d.items() if k in known}
        payload["started_at"] = _parse(d.get("started_at")) or _utcnow()
        payload["updated_at"] = _parse(d.get("updated_at")) or _utcnow()
        payload["last_bar_time"] = {
            k: _parse(v) for k, v in (d.get("last_bar_time") or {}).items()
        }
        payload["equity_history"] = [
            EquityPoint.from_dict(p) for p in (d.get("equity_history") or [])
        ]
        return cls(**payload)

    def save(self, path: Path | str) -> Path:
        """Write state atomically: temp file in the same dir, then rename."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _utcnow()

        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(self.to_dict(), fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)     # atomic on POSIX
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise
        return path

    @classmethod
    def load(cls, path: Path | str) -> "TradingState":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no state file at {path}")
        try:
            return cls.from_dict(json.loads(path.read_text()))
        except json.JSONDecodeError as exc:
            raise StateError(f"state file at {path} is corrupt: {exc}") from exc

    @classmethod
    def load_or_new(cls, path: Path | str) -> "TradingState":
        """
        Load existing state, or start fresh when there is none.

        A corrupt file is *not* silently discarded — losing the record of which
        bars were already traded could double-submit orders — so it raises.
        """
        try:
            return cls.load(path)
        except FileNotFoundError:
            return cls()

    # ── Reporting ────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "cycles_run": self.cycles_run,
            "signals_generated": self.signals_generated,
            "signals_blocked_by_risk": self.signals_blocked_by_risk,
            "orders_submitted": self.orders_submitted,
            "orders_filled": self.orders_filled,
            "orders_rejected": self.orders_rejected,
            "errors": self.errors,
            "is_halted": self.is_halted,
            "halt_reason": self.halt_reason,
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "current_drawdown": self.current_drawdown,
        }
