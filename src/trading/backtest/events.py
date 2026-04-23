"""Event types for the event-driven backtest engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal


class EventType(str, Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"


class OrderType(str, Enum):
    MARKET = "MKT"
    LIMIT = "LMT"


@dataclass
class MarketEvent:
    """Fired when a new bar of OHLCV data becomes available."""
    type: EventType = field(default=EventType.MARKET, init=False)
    datetime: datetime = field(default_factory=datetime.utcnow)
    symbol: str = ""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0


@dataclass
class SignalEvent:
    """Strategy emits a directional signal for a symbol."""
    type: EventType = field(default=EventType.SIGNAL, init=False)
    datetime: datetime = field(default_factory=datetime.utcnow)
    symbol: str = ""
    direction: Direction = Direction.LONG
    strength: float = 1.0          # 0–1 conviction score


@dataclass
class OrderEvent:
    """Portfolio converts a signal into a sized order."""
    type: EventType = field(default=EventType.ORDER, init=False)
    datetime: datetime = field(default_factory=datetime.utcnow)
    symbol: str = ""
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0          # shares (positive = buy, negative = sell)
    direction: Direction = Direction.LONG
    limit_price: float | None = None


@dataclass
class FillEvent:
    """Simulated broker confirms an order execution."""
    type: EventType = field(default=EventType.FILL, init=False)
    datetime: datetime = field(default_factory=datetime.utcnow)
    symbol: str = ""
    quantity: float = 0.0          # actual shares filled
    fill_price: float = 0.0
    commission: float = 0.0

    @property
    def cost(self) -> float:
        return self.quantity * self.fill_price + self.commission
