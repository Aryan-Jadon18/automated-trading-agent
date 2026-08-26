"""
Broker abstraction: a common interface over Alpaca and an in-memory mock.

  BrokerBase   — the interface every broker implements
  MockBroker   — deterministic in-memory broker for tests and dry runs
  AlpacaBroker — real paper/live trading via alpaca-py

Live trading is gated: AlpacaBroker refuses mode="live" unless the caller
passes allow_live=True *and* sets TRADING_ALLOW_LIVE=1 in the environment.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.trading.execution.orders import (
    Order,
    OrderManager,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from src.trading.utils.logging import get_logger

log = get_logger(__name__)


class BrokerError(RuntimeError):
    """Raised when a broker operation fails."""


class LiveTradingBlocked(RuntimeError):
    """Raised when live trading is attempted without explicit authorisation."""


# ── Value objects ─────────────────────────────────────────────────────────────

@dataclass
class Account:
    cash: float
    equity: float
    buying_power: float
    currency: str = "USD"
    pattern_day_trader: bool = False
    trading_blocked: bool = False

    @property
    def is_tradable(self) -> bool:
        return not self.trading_blocked


@dataclass
class BrokerPosition:
    symbol: str
    quantity: float                 # signed: negative = short
    avg_entry_price: float
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pl(self) -> float:
        return self.quantity * (self.current_price - self.avg_entry_price)

    @property
    def unrealized_pl_pct(self) -> float:
        cost = abs(self.quantity * self.avg_entry_price)
        return self.unrealized_pl / cost if cost > 0 else 0.0


# ── Interface ─────────────────────────────────────────────────────────────────

class BrokerBase(ABC):
    """Every broker implementation exposes this surface."""

    def __init__(self, order_manager: OrderManager | None = None) -> None:
        self.orders = order_manager or OrderManager()

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def get_account(self) -> Account: ...

    @abstractmethod
    def get_positions(self) -> dict[str, BrokerPosition]: ...

    @abstractmethod
    def submit_order(self, order: Order) -> Order: ...

    @abstractmethod
    def cancel_order(self, order: Order) -> Order: ...

    @abstractmethod
    def sync_order(self, order: Order) -> Order:
        """Refresh one order's status from the broker."""

    @abstractmethod
    def is_market_open(self) -> bool: ...

    # ── Shared conveniences ──────────────────────────────────────────────────

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        return self.get_positions().get(symbol)

    def cancel_all(self, symbol: str | None = None) -> list[Order]:
        """Cancel every working order (optionally for one symbol)."""
        return [self.cancel_order(o) for o in self.orders.open_orders(symbol)]

    def sync_all(self) -> list[Order]:
        return [self.sync_order(o) for o in self.orders.open_orders()]

    def close_position(self, symbol: str) -> Optional[Order]:
        """Submit a market order flattening `symbol`. Returns None if flat."""
        pos = self.get_position(symbol)
        if pos is None or pos.quantity == 0:
            return None
        side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
        return self.submit_order(
            Order(symbol, side, abs(pos.quantity), OrderType.MARKET)
        )

    def close_all_positions(self) -> list[Order]:
        out = []
        for sym in list(self.get_positions()):
            order = self.close_position(sym)
            if order is not None:
                out.append(order)
        return out

    def reconcile(self) -> dict[str, float]:
        """Drift between locally-tracked fills and broker positions."""
        return self.orders.reconcile(
            {s: p.quantity for s, p in self.get_positions().items()}
        )


# ── Mock broker ───────────────────────────────────────────────────────────────

class MockBroker(BrokerBase):
    """
    Deterministic in-memory broker. No network, no credentials.

    Market orders fill immediately at the current price plus slippage.
    Limit orders rest until `set_price()` moves the market through the limit.
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        prices: dict[str, float] | None = None,
        commission: float = 0.0,
        slippage: float = 0.0,
        market_open: bool = True,
        order_manager: OrderManager | None = None,
    ) -> None:
        super().__init__(order_manager)
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.prices: dict[str, float] = dict(prices or {})
        self.commission = commission
        self.slippage = slippage
        self.market_open = market_open
        self.connected = False
        self._positions: dict[str, BrokerPosition] = {}
        self._next_id = 1
        self.reject_next: str | None = None    # test hook: force a rejection

    # ── Test controls ────────────────────────────────────────────────────────

    def set_price(self, symbol: str, price: float) -> None:
        """Move the market and try to fill any resting orders."""
        self.prices[symbol] = price
        for pos in self._positions.values():
            if pos.symbol == symbol:
                pos.current_price = price
        self._try_fill_resting(symbol)

    def _try_fill_resting(self, symbol: str) -> None:
        for order in self.orders.open_orders(symbol):
            if order.status != OrderStatus.SUBMITTED:
                continue
            if order.order_type != OrderType.LIMIT:
                continue
            price = self.prices[symbol]
            crossed = (
                (order.side == OrderSide.BUY and price <= order.limit_price)
                or (order.side == OrderSide.SELL and price >= order.limit_price)
            )
            if crossed:
                self._fill(order, order.limit_price)

    # ── Interface ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        self.connected = True

    def get_account(self) -> Account:
        equity = self.cash + sum(p.market_value for p in self._positions.values())
        return Account(
            cash=self.cash,
            equity=equity,
            buying_power=max(0.0, self.cash),
        )

    def get_positions(self) -> dict[str, BrokerPosition]:
        return {s: p for s, p in self._positions.items() if p.quantity != 0}

    def is_market_open(self) -> bool:
        return self.market_open

    def submit_order(self, order: Order) -> Order:
        self.orders.track(order)

        if self.reject_next is not None:
            reason, self.reject_next = self.reject_next, None
            order.reject(reason)
            return order

        if not self.market_open and order.order_type == OrderType.MARKET:
            order.reject("market closed")
            return order

        if order.symbol not in self.prices:
            order.reject(f"no price for {order.symbol}")
            return order

        broker_id = f"mock-{self._next_id}"
        self._next_id += 1
        order.transition_to(OrderStatus.SUBMITTED)
        self.orders.link_broker_id(order.client_order_id, broker_id)

        if order.order_type == OrderType.MARKET:
            sign = 1 if order.side == OrderSide.BUY else -1
            fill_price = self.prices[order.symbol] * (1 + sign * self.slippage)
            self._fill(order, fill_price)
        else:
            self._try_fill_resting(order.symbol)

        return order

    def cancel_order(self, order: Order) -> Order:
        if order.is_terminal:
            raise BrokerError(f"cannot cancel {order.status.value} order")
        order.cancel()
        return order

    def sync_order(self, order: Order) -> Order:
        return order        # mock state is always authoritative

    # ── Fill mechanics ───────────────────────────────────────────────────────

    def _fill(self, order: Order, price: float) -> None:
        qty = order.remaining_quantity
        commission = qty * price * self.commission
        signed = qty if order.side == OrderSide.BUY else -qty

        order.apply_fill(qty, price, commission)
        self.cash -= signed * price + commission

        pos = self._positions.get(order.symbol)
        if pos is None:
            self._positions[order.symbol] = BrokerPosition(
                symbol=order.symbol,
                quantity=signed,
                avg_entry_price=price,
                current_price=self.prices.get(order.symbol, price),
            )
            return

        new_qty = pos.quantity + signed
        if pos.quantity == 0 or (pos.quantity > 0) != (signed > 0):
            # Opening, closing, or flipping — reset the cost basis.
            pos.avg_entry_price = price if new_qty != 0 else 0.0
        else:
            pos.avg_entry_price = (
                pos.quantity * pos.avg_entry_price + signed * price
            ) / new_qty
        pos.quantity = new_qty
        pos.current_price = self.prices.get(order.symbol, price)


# ── Alpaca broker ─────────────────────────────────────────────────────────────

class AlpacaBroker(BrokerBase):
    """
    Alpaca paper/live trading via alpaca-py.

    alpaca-py is imported lazily so this module stays importable without it.

    Live trading requires BOTH:
      - allow_live=True passed explicitly at construction, and
      - TRADING_ALLOW_LIVE=1 in the environment
    Two independent switches make it hard to reach live by accident.
    """

    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        mode: str = "paper",
        allow_live: bool = False,
        order_manager: OrderManager | None = None,
    ) -> None:
        super().__init__(order_manager)
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self.mode = mode.lower()

        if self.mode not in ("paper", "live"):
            raise ValueError(f"mode must be 'paper' or 'live', got {mode!r}")

        if self.mode == "live":
            env_ok = os.environ.get("TRADING_ALLOW_LIVE") == "1"
            if not (allow_live and env_ok):
                raise LiveTradingBlocked(
                    "Live trading is blocked. It requires BOTH allow_live=True and "
                    "TRADING_ALLOW_LIVE=1 in the environment. Use mode='paper' instead."
                )

        self._client = None

    @property
    def is_paper(self) -> bool:
        return self.mode == "paper"

    @classmethod
    def from_config(cls, exec_cfg, **kw) -> "AlpacaBroker":
        """Build from an ExecutionConfig (config/settings.py)."""
        return cls(
            api_key=exec_cfg.alpaca_api_key or None,
            secret_key=exec_cfg.alpaca_secret_key or None,
            mode=exec_cfg.mode,
            **kw,
        )

    # ── Connection ───────────────────────────────────────────────────────────

    def connect(self) -> None:
        if not self.api_key or not self.secret_key:
            raise BrokerError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY in the environment."
            )
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:
            raise BrokerError(
                "alpaca-py is not installed. Run: pip install alpaca-py"
            ) from exc

        self._client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.is_paper,
        )
        log.info("alpaca_connected", mode=self.mode)

    def _require_client(self):
        if self._client is None:
            raise BrokerError("Not connected. Call connect() first.")
        return self._client

    # ── Account and positions ────────────────────────────────────────────────

    def get_account(self) -> Account:
        raw = self._require_client().get_account()
        return Account(
            cash=float(raw.cash),
            equity=float(raw.equity),
            buying_power=float(raw.buying_power),
            currency=getattr(raw, "currency", "USD"),
            pattern_day_trader=bool(getattr(raw, "pattern_day_trader", False)),
            trading_blocked=bool(getattr(raw, "trading_blocked", False)),
        )

    def get_positions(self) -> dict[str, BrokerPosition]:
        out: dict[str, BrokerPosition] = {}
        for p in self._require_client().get_all_positions():
            out[p.symbol] = BrokerPosition(
                symbol=p.symbol,
                quantity=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(getattr(p, "current_price", 0) or 0),
            )
        return out

    def is_market_open(self) -> bool:
        return bool(self._require_client().get_clock().is_open)

    # ── Order translation ────────────────────────────────────────────────────

    def _build_request(self, order: Order):
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import TimeInForce as AlpacaTIF
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLimitOrderRequest,
            StopOrderRequest,
        )

        side = AlpacaSide.BUY if order.side == OrderSide.BUY else AlpacaSide.SELL
        tif = {
            TimeInForce.DAY: AlpacaTIF.DAY,
            TimeInForce.GTC: AlpacaTIF.GTC,
            TimeInForce.IOC: AlpacaTIF.IOC,
            TimeInForce.FOK: AlpacaTIF.FOK,
        }[order.time_in_force]

        common = dict(
            symbol=order.symbol,
            qty=order.quantity,
            side=side,
            time_in_force=tif,
            client_order_id=order.client_order_id,
        )

        if order.order_type == OrderType.MARKET:
            return MarketOrderRequest(**common)
        if order.order_type == OrderType.LIMIT:
            return LimitOrderRequest(limit_price=order.limit_price, **common)
        if order.order_type == OrderType.STOP:
            return StopOrderRequest(stop_price=order.stop_price, **common)
        return StopLimitOrderRequest(
            limit_price=order.limit_price, stop_price=order.stop_price, **common
        )

    _STATUS_MAP = {
        "new": OrderStatus.SUBMITTED,
        "accepted": OrderStatus.SUBMITTED,
        "pending_new": OrderStatus.SUBMITTED,
        "accepted_for_bidding": OrderStatus.SUBMITTED,
        "partially_filled": OrderStatus.PARTIALLY_FILLED,
        "filled": OrderStatus.FILLED,
        "canceled": OrderStatus.CANCELED,
        "pending_cancel": OrderStatus.SUBMITTED,
        "expired": OrderStatus.EXPIRED,
        "rejected": OrderStatus.REJECTED,
        "suspended": OrderStatus.SUBMITTED,
        "done_for_day": OrderStatus.EXPIRED,
        "replaced": OrderStatus.CANCELED,
        "stopped": OrderStatus.SUBMITTED,
        "calculated": OrderStatus.SUBMITTED,
    }

    def _apply_remote(self, order: Order, raw) -> Order:
        """Fold the broker's view of an order back into local state."""
        raw_status = str(getattr(raw, "status", "")).lower().split(".")[-1]
        mapped = self._STATUS_MAP.get(raw_status)

        filled_qty = float(getattr(raw, "filled_qty", 0) or 0)
        avg_price = float(getattr(raw, "filled_avg_price", 0) or 0)

        if filled_qty > order.filled_quantity:
            order.filled_quantity = filled_qty
            order.avg_fill_price = avg_price

        if mapped is not None and mapped != order.status and not order.is_terminal:
            try:
                order.transition_to(mapped)
            except Exception:
                # Broker is authoritative; a transition we consider illegal means
                # our local view drifted. Take the remote status and log it.
                log.warning(
                    "order_status_drift",
                    symbol=order.symbol,
                    local=order.status.value,
                    remote=mapped.value,
                )
                order.status = mapped
        return order

    # ── Order operations ─────────────────────────────────────────────────────

    def submit_order(self, order: Order) -> Order:
        client = self._require_client()
        self.orders.track(order)
        try:
            raw = client.submit_order(self._build_request(order))
        except Exception as exc:
            order.reject(str(exc))
            log.error("order_rejected", symbol=order.symbol, error=str(exc))
            return order

        order.transition_to(OrderStatus.SUBMITTED)
        self.orders.link_broker_id(order.client_order_id, str(raw.id))
        return self._apply_remote(order, raw)

    def cancel_order(self, order: Order) -> Order:
        if order.is_terminal:
            raise BrokerError(f"cannot cancel {order.status.value} order")
        if not order.broker_order_id:
            raise BrokerError("order has no broker_order_id")
        self._require_client().cancel_order_by_id(order.broker_order_id)
        order.cancel()
        return order

    def sync_order(self, order: Order) -> Order:
        if not order.broker_order_id:
            return order
        raw = self._require_client().get_order_by_id(order.broker_order_id)
        return self._apply_remote(order, raw)
