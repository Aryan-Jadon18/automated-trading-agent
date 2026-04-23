"""Abstract base class for all trading strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

import pandas as pd

from src.trading.backtest.events import Direction, MarketEvent, SignalEvent


class Strategy(ABC):
    """
    Strategies receive MarketEvents bar-by-bar and emit SignalEvents.
    They must be stateless between separate backtest runs — reset() is called
    before each run.
    """

    def __init__(self, symbols: list[str]) -> None:
        self.symbols = symbols
        self._bars: dict[str, deque] = {s: deque() for s in symbols}

    def reset(self) -> None:
        """Clear internal state before a new backtest run."""
        self._bars = {s: deque() for s in self.symbols}

    def update_bar(self, event: MarketEvent) -> None:
        """Append a new OHLCV bar to internal history."""
        self._bars[event.symbol].append({
            "datetime": event.datetime,
            "open": event.open,
            "high": event.high,
            "low": event.low,
            "close": event.close,
            "volume": event.volume,
        })

    def get_bars(self, symbol: str, n: Optional[int] = None) -> pd.DataFrame:
        """Return the last *n* bars for *symbol* as a DataFrame."""
        bars = list(self._bars[symbol])
        if n is not None:
            bars = bars[-n:]
        return pd.DataFrame(bars).set_index("datetime") if bars else pd.DataFrame()

    def _signal(
        self,
        event: MarketEvent,
        direction: Direction,
        strength: float = 1.0,
    ) -> SignalEvent:
        return SignalEvent(
            datetime=event.datetime,
            symbol=event.symbol,
            direction=direction,
            strength=strength,
        )

    @abstractmethod
    def on_bar(self, event: MarketEvent) -> Optional[SignalEvent]:
        """
        Called for every new bar. Return a SignalEvent or None.
        Implementations should call update_bar(event) first.
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
