"""Portfolio — tracks positions, cash, equity curve, and generates orders."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from src.trading.backtest.events import (
    Direction,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderType,
    SignalEvent,
)
from src.trading.utils.metrics import summary as perf_summary


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0          # positive = long, negative = short
    avg_price: float = 0.0
    realized_pnl: float = 0.0

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealized_pnl(self, price: float) -> float:
        return self.quantity * (price - self.avg_price)

    def update_fill(self, quantity: float, price: float) -> None:
        """Update avg cost on a new fill (handles partial fills and exits)."""
        new_qty = self.quantity + quantity
        if self.quantity == 0 or (self.quantity > 0) != (quantity > 0):
            # Opening or flipping — reset avg price
            if new_qty != 0:
                self.avg_price = price
            else:
                self.realized_pnl += self.quantity * (price - self.avg_price)
                self.avg_price = 0.0
        else:
            # Adding to existing position — weighted avg
            total_cost = self.quantity * self.avg_price + quantity * price
            self.avg_price = total_cost / new_qty
        self.quantity = new_qty


class Portfolio:
    """
    Converts SignalEvents → OrderEvents.
    Applies FillEvents to update positions and cash.
    Records the equity curve bar-by-bar.
    """

    def __init__(
        self,
        symbols: list[str],
        initial_capital: float = 100_000.0,
        max_position_pct: float = 0.20,
        commission: float = 0.001,
    ) -> None:
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.commission = commission

        self.cash: float = initial_capital
        self.positions: dict[str, Position] = {s: Position(s) for s in symbols}
        self._latest_prices: dict[str, float] = {}

        self._equity_records: list[dict] = []
        self._trade_log: list[dict] = []

    def reset(self) -> None:
        self.cash = self.initial_capital
        self.positions = {s: Position(s) for s in self.symbols}
        self._latest_prices = {}
        self._equity_records = []
        self._trade_log = []

    # ── Market update ─────────────────────────────────────────────────────────

    def on_market(self, event: MarketEvent) -> None:
        self._latest_prices[event.symbol] = event.close
        equity = self._total_equity()
        self._equity_records.append({"datetime": event.datetime, "equity": equity})

    # ── Signal → Order ────────────────────────────────────────────────────────

    def on_signal(self, signal: SignalEvent) -> Optional[OrderEvent]:
        price = self._latest_prices.get(signal.symbol)
        if price is None or price <= 0:
            return None

        equity = self._total_equity()
        pos = self.positions[signal.symbol]

        if signal.direction == Direction.EXIT:
            if pos.quantity == 0:
                return None
            quantity = -pos.quantity  # close entire position
            return OrderEvent(
                datetime=signal.datetime,
                symbol=signal.symbol,
                order_type=OrderType.MARKET,
                quantity=quantity,
                direction=Direction.EXIT,
            )

        if signal.direction == Direction.LONG:
            # Size = min(max_position_pct of equity, available cash) × signal strength
            target_value = equity * self.max_position_pct * signal.strength
            current_value = pos.quantity * price
            delta_value = target_value - current_value
            if delta_value < price:         # less than 1 share — skip
                return None
            quantity = int(delta_value / price)
            if quantity * price > self.cash:
                quantity = int(self.cash / price)
            if quantity <= 0:
                return None
            return OrderEvent(
                datetime=signal.datetime,
                symbol=signal.symbol,
                order_type=OrderType.MARKET,
                quantity=float(quantity),
                direction=Direction.LONG,
            )

        if signal.direction == Direction.SHORT:
            target_value = equity * self.max_position_pct * signal.strength
            quantity = -int(target_value / price)
            return OrderEvent(
                datetime=signal.datetime,
                symbol=signal.symbol,
                order_type=OrderType.MARKET,
                quantity=float(quantity),
                direction=Direction.SHORT,
            )

        return None

    # ── Fill → Position + Cash ────────────────────────────────────────────────

    def on_fill(self, fill: FillEvent) -> None:
        pos = self.positions[fill.symbol]
        pos.update_fill(fill.quantity, fill.fill_price)
        self.cash -= fill.quantity * fill.fill_price + fill.commission
        self._trade_log.append({
            "datetime": fill.datetime,
            "symbol": fill.symbol,
            "quantity": fill.quantity,
            "price": fill.fill_price,
            "commission": fill.commission,
            "cash_after": self.cash,
        })

    # ── Reporting ─────────────────────────────────────────────────────────────

    def _total_equity(self) -> float:
        market_value = sum(
            pos.quantity * self._latest_prices.get(sym, 0.0)
            for sym, pos in self.positions.items()
        )
        return self.cash + market_value

    def equity_curve(self) -> pd.DataFrame:
        df = pd.DataFrame(self._equity_records).set_index("datetime")
        df["returns"] = df["equity"].pct_change()
        return df

    def trade_log(self) -> pd.DataFrame:
        return pd.DataFrame(self._trade_log)

    def performance(self) -> dict:
        ec = self.equity_curve().dropna()
        if ec.empty:
            return {}
        return perf_summary(ec["equity"], ec["returns"])
