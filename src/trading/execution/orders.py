"""
Order models and lifecycle management for live/paper execution.

Distinct from backtest/events.py: those are internal simulation events, these
are real orders with broker IDs, partial fills, and a status state machine.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import pandas as pd


# ── Enums ─────────────────────────────────────────────────────────────────────

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"     # good-till-cancelled
    IOC = "ioc"     # immediate-or-cancel
    FOK = "fok"     # fill-or-kill


class OrderStatus(str, Enum):
    NEW = "new"                          # created locally, not yet sent
    SUBMITTED = "submitted"              # accepted by broker
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


#: Statuses that can no longer change.
TERMINAL_STATUSES = frozenset({
    OrderStatus.FILLED,
    OrderStatus.CANCELED,
    OrderStatus.REJECTED,
    OrderStatus.EXPIRED,
})

#: Legal status transitions. Anything else is a bug or a broker desync.
_ALLOWED_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.NEW: frozenset({
        OrderStatus.SUBMITTED, OrderStatus.REJECTED, OrderStatus.CANCELED,
    }),
    OrderStatus.SUBMITTED: frozenset({
        OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
        OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED,
    }),
    OrderStatus.PARTIALLY_FILLED: frozenset({
        OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
        OrderStatus.CANCELED, OrderStatus.EXPIRED,
    }),
    OrderStatus.FILLED: frozenset(),
    OrderStatus.CANCELED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
    OrderStatus.EXPIRED: frozenset(),
}


class InvalidOrderTransition(RuntimeError):
    """Raised when an order is moved to a status it cannot legally reach."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Order ─────────────────────────────────────────────────────────────────────

@dataclass
class Order:
    """
    A single order tracked from creation through to a terminal status.

    `quantity` is always positive; direction lives in `side`.
    """
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY

    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    broker_order_id: Optional[str] = None

    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0

    created_at: datetime = field(default_factory=_utcnow)
    submitted_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    reject_reason: Optional[str] = None
    tags: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and self.limit_price is None:
            raise ValueError(f"{self.order_type.value} order requires limit_price")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError(f"{self.order_type.value} order requires stop_price")

    # ── Derived state ────────────────────────────────────────────────────────

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def is_open(self) -> bool:
        return not self.is_terminal

    @property
    def remaining_quantity(self) -> float:
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def signed_filled_quantity(self) -> float:
        """Filled quantity signed by direction (negative for sells)."""
        sign = 1.0 if self.side == OrderSide.BUY else -1.0
        return sign * self.filled_quantity

    @property
    def filled_notional(self) -> float:
        return self.filled_quantity * self.avg_fill_price

    # ── Transitions ──────────────────────────────────────────────────────────

    def transition_to(self, new_status: OrderStatus) -> None:
        """Move to `new_status`, rejecting illegal transitions."""
        if new_status not in _ALLOWED_TRANSITIONS[self.status]:
            raise InvalidOrderTransition(
                f"{self.symbol} {self.client_order_id[:8]}: "
                f"cannot go {self.status.value} → {new_status.value}"
            )
        self.status = new_status
        if new_status == OrderStatus.SUBMITTED and self.submitted_at is None:
            self.submitted_at = _utcnow()
        if new_status in TERMINAL_STATUSES:
            self.closed_at = _utcnow()

    def apply_fill(self, quantity: float, price: float, commission: float = 0.0) -> None:
        """
        Record a (possibly partial) fill, updating the volume-weighted average price
        and advancing the status accordingly.
        """
        if quantity <= 0:
            raise ValueError(f"fill quantity must be positive, got {quantity}")
        if self.is_terminal:
            raise InvalidOrderTransition(
                f"cannot fill order in terminal status {self.status.value}"
            )
        if quantity > self.remaining_quantity + 1e-9:
            raise ValueError(
                f"fill {quantity} exceeds remaining {self.remaining_quantity}"
            )

        prior_notional = self.filled_quantity * self.avg_fill_price
        self.filled_quantity += quantity
        self.avg_fill_price = (prior_notional + quantity * price) / self.filled_quantity
        self.commission += commission

        # Float tolerance: treat near-complete as complete.
        if self.remaining_quantity <= 1e-9:
            self.transition_to(OrderStatus.FILLED)
        else:
            self.transition_to(OrderStatus.PARTIALLY_FILLED)

    def reject(self, reason: str) -> None:
        self.reject_reason = reason
        self.transition_to(OrderStatus.REJECTED)

    def cancel(self) -> None:
        self.transition_to(OrderStatus.CANCELED)

    def to_dict(self) -> dict:
        return {
            "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "commission": self.commission,
            "created_at": self.created_at,
            "submitted_at": self.submitted_at,
            "closed_at": self.closed_at,
            "reject_reason": self.reject_reason,
        }

    def __str__(self) -> str:
        base = (
            f"{self.side.value.upper()} {self.quantity:g} {self.symbol} "
            f"@ {self.order_type.value}"
        )
        if self.limit_price is not None:
            base += f" {self.limit_price:.2f}"
        base += f" [{self.status.value}]"
        if self.filled_quantity > 0:
            base += f" filled {self.filled_quantity:g}@{self.avg_fill_price:.2f}"
        return base


# ── Factory helpers ───────────────────────────────────────────────────────────

def market_order(symbol: str, quantity: float, **kw) -> Order:
    """Build a market order; sign of `quantity` sets the side."""
    side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
    return Order(symbol, side, abs(quantity), OrderType.MARKET, **kw)


def limit_order(symbol: str, quantity: float, limit_price: float, **kw) -> Order:
    side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
    return Order(symbol, side, abs(quantity), OrderType.LIMIT,
                 limit_price=limit_price, **kw)


def stop_order(symbol: str, quantity: float, stop_price: float, **kw) -> Order:
    side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
    return Order(symbol, side, abs(quantity), OrderType.STOP,
                 stop_price=stop_price, **kw)


# ── OrderManager ──────────────────────────────────────────────────────────────

class OrderManager:
    """
    Tracks every order this session has created and reconciles local state
    against the broker's view of the world.
    """

    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}          # client_order_id → Order
        self._by_broker_id: dict[str, str] = {}      # broker_id → client_order_id

    def __len__(self) -> int:
        return len(self._orders)

    def __contains__(self, client_order_id: str) -> bool:
        return client_order_id in self._orders

    # ── Registration ─────────────────────────────────────────────────────────

    def track(self, order: Order) -> Order:
        """Begin tracking an order."""
        self._orders[order.client_order_id] = order
        if order.broker_order_id:
            self._by_broker_id[order.broker_order_id] = order.client_order_id
        return order

    def link_broker_id(self, client_order_id: str, broker_order_id: str) -> None:
        order = self.get(client_order_id)
        if order is None:
            raise KeyError(f"unknown client_order_id {client_order_id}")
        order.broker_order_id = broker_order_id
        self._by_broker_id[broker_order_id] = client_order_id

    # ── Lookup ───────────────────────────────────────────────────────────────

    def get(self, client_order_id: str) -> Optional[Order]:
        return self._orders.get(client_order_id)

    def get_by_broker_id(self, broker_order_id: str) -> Optional[Order]:
        cid = self._by_broker_id.get(broker_order_id)
        return self._orders.get(cid) if cid else None

    def all_orders(self) -> list[Order]:
        return list(self._orders.values())

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        return [
            o for o in self._orders.values()
            if o.is_open and (symbol is None or o.symbol == symbol)
        ]

    def closed_orders(self, symbol: str | None = None) -> list[Order]:
        return [
            o for o in self._orders.values()
            if o.is_terminal and (symbol is None or o.symbol == symbol)
        ]

    def filled_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if o.status == OrderStatus.FILLED]

    # ── Aggregate views ──────────────────────────────────────────────────────

    def net_filled_quantity(self, symbol: str) -> float:
        """Signed net quantity filled for `symbol` across all orders."""
        return sum(
            o.signed_filled_quantity
            for o in self._orders.values()
            if o.symbol == symbol
        )

    def pending_exposure(self, symbol: str) -> float:
        """Signed quantity still working at the broker for `symbol`."""
        total = 0.0
        for o in self._orders.values():
            if o.symbol != symbol or not o.is_open:
                continue
            sign = 1.0 if o.side == OrderSide.BUY else -1.0
            total += sign * o.remaining_quantity
        return total

    # ── Reconciliation ───────────────────────────────────────────────────────

    def reconcile(self, broker_positions: dict[str, float]) -> dict[str, float]:
        """
        Compare locally-tracked net fills against the broker's actual positions.

        Returns {symbol: drift} for every symbol that disagrees, where
        drift = broker_quantity − locally_tracked_quantity. An empty dict means
        local state matches the broker. Non-empty means orders were filled,
        cancelled, or placed outside this session — investigate before trading.
        """
        symbols = set(broker_positions) | {o.symbol for o in self._orders.values()}
        drift: dict[str, float] = {}
        for sym in symbols:
            local = self.net_filled_quantity(sym)
            remote = broker_positions.get(sym, 0.0)
            if abs(remote - local) > 1e-6:
                drift[sym] = remote - local
        return drift

    # ── Reporting ────────────────────────────────────────────────────────────

    def to_frame(self) -> pd.DataFrame:
        if not self._orders:
            return pd.DataFrame(columns=[
                "client_order_id", "broker_order_id", "symbol", "side",
                "quantity", "order_type", "status", "filled_quantity",
                "avg_fill_price", "commission",
            ])
        return pd.DataFrame([o.to_dict() for o in self._orders.values()])

    def summary(self) -> dict:
        orders = list(self._orders.values())
        filled = [o for o in orders if o.status == OrderStatus.FILLED]
        return {
            "n_orders": len(orders),
            "n_open": sum(1 for o in orders if o.is_open),
            "n_filled": len(filled),
            "n_rejected": sum(1 for o in orders if o.status == OrderStatus.REJECTED),
            "n_canceled": sum(1 for o in orders if o.status == OrderStatus.CANCELED),
            "fill_rate": len(filled) / len(orders) if orders else 0.0,
            "total_commission": sum(o.commission for o in orders),
            "total_notional": sum(o.filled_notional for o in orders),
        }
