"""Technical strategies: SMA crossover + RSI filter."""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.trading.backtest.events import Direction, MarketEvent, SignalEvent
from src.trading.strategy.base import Strategy


class SMACrossover(Strategy):
    """
    Classic dual-SMA crossover with optional RSI filter.

    Signal rules:
      LONG  — fast SMA crosses above slow SMA AND RSI < overbought
      EXIT  — fast SMA crosses below slow SMA
      SHORT — disabled by default (long-only)
    """

    def __init__(
        self,
        symbols: list[str],
        fast: int = 20,
        slow: int = 50,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        long_only: bool = True,
    ) -> None:
        super().__init__(symbols)
        self.fast = fast
        self.slow = slow
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.long_only = long_only
        self._position: dict[str, Direction | None] = {s: None for s in symbols}

    def reset(self) -> None:
        super().reset()
        self._position = {s: None for s in self.symbols}

    def _sma(self, closes: list[float], window: int) -> Optional[float]:
        if len(closes) < window:
            return None
        return float(np.mean(closes[-window:]))

    def _rsi(self, closes: list[float]) -> Optional[float]:
        n = self.rsi_period
        if len(closes) < n + 1:
            return None
        deltas = np.diff(closes[-(n + 1):])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def on_bar(self, event: MarketEvent) -> Optional[SignalEvent]:
        self.update_bar(event)
        bars = self.get_bars(event.symbol)
        if bars.empty or len(bars) < self.slow + 1:
            return None

        closes = bars["close"].tolist()
        fast_sma = self._sma(closes, self.fast)
        slow_sma = self._sma(closes, self.slow)
        prev_fast = self._sma(closes[:-1], self.fast)
        prev_slow = self._sma(closes[:-1], self.slow)
        rsi_val = self._rsi(closes)

        if None in (fast_sma, slow_sma, prev_fast, prev_slow):
            return None

        current_pos = self._position[event.symbol]

        # Golden cross → LONG
        crossed_up = prev_fast <= prev_slow and fast_sma > slow_sma
        if crossed_up and (rsi_val is None or rsi_val < self.rsi_overbought):
            if current_pos != Direction.LONG:
                self._position[event.symbol] = Direction.LONG
                return self._signal(event, Direction.LONG)

        # Death cross → EXIT (or SHORT if not long_only)
        crossed_down = prev_fast >= prev_slow and fast_sma < slow_sma
        if crossed_down and current_pos is not None:
            if self.long_only:
                self._position[event.symbol] = None
                return self._signal(event, Direction.EXIT)
            else:
                self._position[event.symbol] = Direction.SHORT
                return self._signal(event, Direction.SHORT)

        return None


class MeanReversion(Strategy):
    """
    Bollinger Band mean-reversion strategy.

    LONG  — close < lower band (oversold)
    EXIT  — close > middle band
    SHORT — close > upper band (if not long_only)
    """

    def __init__(
        self,
        symbols: list[str],
        window: int = 20,
        n_std: float = 2.0,
        long_only: bool = True,
    ) -> None:
        super().__init__(symbols)
        self.window = window
        self.n_std = n_std
        self.long_only = long_only
        self._position: dict[str, Direction | None] = {s: None for s in symbols}

    def reset(self) -> None:
        super().reset()
        self._position = {s: None for s in self.symbols}

    def on_bar(self, event: MarketEvent) -> Optional[SignalEvent]:
        self.update_bar(event)
        bars = self.get_bars(event.symbol)
        if len(bars) < self.window:
            return None

        closes = bars["close"].values[-self.window:]
        mid = closes.mean()
        std = closes.std(ddof=1)
        upper = mid + self.n_std * std
        lower = mid - self.n_std * std
        price = event.close
        current_pos = self._position[event.symbol]

        if price < lower and current_pos != Direction.LONG:
            self._position[event.symbol] = Direction.LONG
            strength = min(1.0, (lower - price) / (self.n_std * std))
            return self._signal(event, Direction.LONG, strength)

        if price > mid and current_pos == Direction.LONG:
            self._position[event.symbol] = None
            return self._signal(event, Direction.EXIT)

        if not self.long_only and price > upper and current_pos != Direction.SHORT:
            self._position[event.symbol] = Direction.SHORT
            return self._signal(event, Direction.SHORT)

        return None
