"""Event-driven backtest engine."""
from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from typing import Optional

import pandas as pd

from src.trading.backtest.events import (
    FillEvent,
    MarketEvent,
    OrderEvent,
)
from src.trading.backtest.portfolio import Portfolio
from src.trading.strategy.base import Strategy
from src.trading.utils.logging import get_logger

log = get_logger(__name__)


class SimulatedBroker:
    """
    Fills orders at next-bar open with slippage and commission.
    Realistic: you can only trade at the next bar's open after signal.
    """

    def __init__(self, commission: float = 0.001, slippage: float = 0.0005) -> None:
        self.commission = commission
        self.slippage = slippage

    def execute(self, order: OrderEvent, next_open: float) -> FillEvent:
        direction_sign = 1 if order.quantity > 0 else -1
        fill_price = next_open * (1 + direction_sign * self.slippage)
        commission = abs(order.quantity) * fill_price * self.commission
        return FillEvent(
            datetime=order.datetime,
            symbol=order.symbol,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
        )


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    performance: dict
    n_bars: int
    n_signals: int
    n_fills: int

    def __str__(self) -> str:
        p = self.performance
        lines = [
            f"  Bars:          {self.n_bars}",
            f"  Signals:       {self.n_signals}",
            f"  Fills:         {self.n_fills}",
            f"  Total Return:  {p.get('total_return', 0):.2%}",
            f"  CAGR:          {p.get('cagr', 0):.2%}",
            f"  Sharpe:        {p.get('sharpe', 0):.2f}",
            f"  Max Drawdown:  {p.get('max_drawdown', 0):.2%}",
            f"  Win Rate:      {p.get('win_rate', 0):.2%}",
        ]
        return "\n".join(lines)


class BacktestEngine:
    """
    Feeds historical OHLCV data bar-by-bar through an event queue:

        MarketEvent → Strategy.on_bar() → SignalEvent
        SignalEvent  → Portfolio.on_signal() → OrderEvent
        OrderEvent   → SimulatedBroker.execute() → FillEvent
        FillEvent    → Portfolio.on_fill()

    Orders execute at the *next* bar's open (no look-ahead bias).
    """

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        strategy: Strategy,
        portfolio: Portfolio,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ) -> None:
        self.data = data
        self.strategy = strategy
        self.portfolio = portfolio
        self.broker = SimulatedBroker(commission=commission, slippage=slippage)
        self._queue: Queue = Queue()

    def _build_timeline(self) -> pd.DatetimeIndex:
        """Merge all symbol timestamps into a single sorted timeline."""
        all_idx = pd.concat(
            [df.index.to_series() for df in self.data.values()]
        ).drop_duplicates().sort_values()
        return pd.DatetimeIndex(all_idx)

    def run(self) -> BacktestResult:
        self.strategy.reset()
        self.portfolio.reset()

        timeline = self._build_timeline()
        n_signals = 0
        n_fills = 0

        # Pending orders waiting for next bar's open
        pending_orders: list[tuple[str, OrderEvent]] = []

        for i, dt in enumerate(timeline):
            # ── Execute pending orders at this bar's open ──────────────────
            new_pending = []
            for sym, order in pending_orders:
                if sym in self.data and dt in self.data[sym].index:
                    next_open = float(self.data[sym].loc[dt, "open"])
                    fill = self.broker.execute(order, next_open)
                    self.portfolio.on_fill(fill)
                    n_fills += 1
                    log.debug("fill", symbol=sym, qty=fill.quantity, price=f"{fill.fill_price:.2f}")
                else:
                    new_pending.append((sym, order))  # carry forward
            pending_orders = new_pending

            # ── Feed market data for this bar ──────────────────────────────
            for sym, df in self.data.items():
                if dt not in df.index:
                    continue
                row = df.loc[dt]
                market_evt = MarketEvent(
                    datetime=dt.to_pydatetime(),
                    symbol=sym,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
                self.portfolio.on_market(market_evt)

                signal = self.strategy.on_bar(market_evt)
                if signal is not None:
                    n_signals += 1
                    order = self.portfolio.on_signal(signal)
                    if order is not None:
                        pending_orders.append((sym, order))

        return BacktestResult(
            equity_curve=self.portfolio.equity_curve(),
            trade_log=self.portfolio.trade_log(),
            performance=self.portfolio.performance(),
            n_bars=len(timeline),
            n_signals=n_signals,
            n_fills=n_fills,
        )


def run_backtest(
    data: dict[str, pd.DataFrame],
    strategy: Strategy,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    max_position_pct: float = 0.20,
) -> BacktestResult:
    """Convenience wrapper — creates Portfolio + Engine and runs in one call."""
    portfolio = Portfolio(
        symbols=list(data.keys()),
        initial_capital=initial_capital,
        commission=commission,
        max_position_pct=max_position_pct,
    )
    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        portfolio=portfolio,
        commission=commission,
        slippage=slippage,
    )
    return engine.run()
