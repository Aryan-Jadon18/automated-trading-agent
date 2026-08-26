"""
Risk manager — position sizing, drawdown circuit-breaker, exposure limits.

Decision pipeline:
    proposed order → RiskManager.evaluate() → RiskDecision(approved_quantity, reasons)

Components:
  kelly_fraction()    — optimal bet size from win-rate + payoff ratio
  DrawdownGuard       — halts trading when equity drawdown breaches a limit
  ExposureLimits      — caps per-position, gross, and net exposure
  CorrelationLimiter  — penalises adding to an already-correlated book
  RiskManager         — orchestrates all of the above
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ── Kelly sizing ──────────────────────────────────────────────────────────────

def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Kelly criterion:  f* = (p·b − q) / b
      p = win probability, q = 1 − p, b = payoff ratio (avg_win / |avg_loss|)

    Returns the full-Kelly fraction clipped to [0, 1].
    Returns 0.0 when the edge is negative or inputs are degenerate.
    """
    if not (0.0 <= win_rate <= 1.0):
        raise ValueError(f"win_rate must be in [0, 1], got {win_rate}")

    loss = abs(avg_loss)
    if loss <= 0 or avg_win <= 0:
        return 0.0

    b = avg_win / loss
    p = win_rate
    q = 1.0 - p
    f = (p * b - q) / b
    return float(np.clip(f, 0.0, 1.0))


def kelly_from_returns(returns: pd.Series) -> float:
    """Estimate the full-Kelly fraction from a realised return series."""
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return 0.0

    wins = r[r > 0]
    losses = r[r < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0

    return kelly_fraction(
        win_rate=len(wins) / len(r),
        avg_win=float(wins.mean()),
        avg_loss=float(losses.mean()),
    )


# ── Drawdown circuit-breaker ──────────────────────────────────────────────────

class DrawdownGuard:
    """
    Tracks peak equity and halts trading when drawdown breaches `max_drawdown_pct`.

    Once halted, trading stays halted until equity recovers to within
    `resume_drawdown_pct` of the peak (hysteresis prevents flip-flopping).
    """

    def __init__(
        self,
        max_drawdown_pct: float = 0.15,
        resume_drawdown_pct: float = 0.05,
    ) -> None:
        if not 0.0 < max_drawdown_pct <= 1.0:
            raise ValueError("max_drawdown_pct must be in (0, 1]")
        if resume_drawdown_pct > max_drawdown_pct:
            raise ValueError("resume_drawdown_pct must be <= max_drawdown_pct")

        self.max_drawdown_pct = max_drawdown_pct
        self.resume_drawdown_pct = resume_drawdown_pct

        self.peak: float | None = None
        self.current_equity: float | None = None
        self.is_halted: bool = False
        self.halt_count: int = 0

    def reset(self) -> None:
        self.peak = None
        self.current_equity = None
        self.is_halted = False
        self.halt_count = 0

    @property
    def current_drawdown(self) -> float:
        """Current drawdown as a negative fraction (0.0 when at peak)."""
        if self.peak is None or self.current_equity is None or self.peak <= 0:
            return 0.0
        return self.current_equity / self.peak - 1.0

    def update(self, equity: float) -> bool:
        """
        Feed the latest equity value. Returns True if trading is halted.
        """
        self.current_equity = equity
        if self.peak is None or equity > self.peak:
            self.peak = equity

        dd = abs(self.current_drawdown)

        if not self.is_halted and dd >= self.max_drawdown_pct:
            self.is_halted = True
            self.halt_count += 1
        elif self.is_halted and dd <= self.resume_drawdown_pct:
            self.is_halted = False

        return self.is_halted


# ── Exposure limits ───────────────────────────────────────────────────────────

class ExposureLimits:
    """
    Caps order size against per-position, gross, net, and count limits.

    All percentages are fractions of total equity.
    """

    def __init__(
        self,
        max_position_pct: float = 0.20,
        max_gross_exposure: float = 1.0,
        max_net_exposure: float = 1.0,
        max_open_positions: int = 10,
    ) -> None:
        self.max_position_pct = max_position_pct
        self.max_gross_exposure = max_gross_exposure
        self.max_net_exposure = max_net_exposure
        self.max_open_positions = max_open_positions

    def cap_quantity(
        self,
        symbol: str,
        proposed_qty: float,
        price: float,
        equity: float,
        positions: dict[str, float],
        prices: dict[str, float],
    ) -> tuple[float, list[str]]:
        """
        Shrink `proposed_qty` until every exposure limit is satisfied.

        positions: symbol → signed share quantity (positive long, negative short)
        prices:    symbol → latest price

        Returns (approved_qty, reasons). Reasons list every binding constraint.
        """
        reasons: list[str] = []
        if price <= 0 or equity <= 0:
            return 0.0, ["invalid price or equity"]

        qty = proposed_qty
        current_qty = positions.get(symbol, 0.0)
        is_reducing = current_qty != 0 and np.sign(qty) != np.sign(current_qty)

        # Reducing or closing an existing position is always allowed.
        if is_reducing and abs(qty) <= abs(current_qty):
            return qty, reasons

        # 1. Per-position cap
        max_pos_value = equity * self.max_position_pct
        resulting_value = abs((current_qty + qty) * price)
        if resulting_value > max_pos_value:
            allowed_shares = max_pos_value / price - abs(current_qty)
            capped = max(0.0, allowed_shares) * np.sign(qty)
            if abs(capped) < abs(qty):
                reasons.append(
                    f"position cap {self.max_position_pct:.0%} of equity"
                )
                qty = capped

        # 2. Open-position count cap (only blocks *new* symbols)
        if current_qty == 0 and qty != 0:
            n_open = sum(1 for q in positions.values() if q != 0)
            if n_open >= self.max_open_positions:
                return 0.0, reasons + [
                    f"max open positions ({self.max_open_positions}) reached"
                ]

        # 3. Gross exposure cap (sum of |value| across the book)
        gross = sum(
            abs(q * prices.get(s, 0.0)) for s, q in positions.items() if s != symbol
        )
        max_gross_value = equity * self.max_gross_exposure
        room = max_gross_value - gross - abs(current_qty * price)
        if room <= 0:
            return 0.0, reasons + [
                f"gross exposure cap {self.max_gross_exposure:.0%} reached"
            ]
        if abs(qty * price) > room:
            qty = (room / price) * np.sign(qty)
            reasons.append(f"gross exposure cap {self.max_gross_exposure:.0%}")

        # 4. Net exposure cap (signed sum — limits directional tilt)
        net = sum(
            q * prices.get(s, 0.0) for s, q in positions.items() if s != symbol
        ) + current_qty * price
        max_net_value = equity * self.max_net_exposure
        projected_net = net + qty * price
        if abs(projected_net) > max_net_value:
            allowed_value = np.sign(projected_net) * max_net_value - net
            qty = allowed_value / price
            reasons.append(f"net exposure cap {self.max_net_exposure:.0%}")

        # Never flip direction through capping
        if np.sign(qty) != np.sign(proposed_qty):
            qty = 0.0

        return float(qty), reasons


# ── Correlation-aware allocation ──────────────────────────────────────────────

class CorrelationLimiter:
    """
    Limits total exposure to a cluster of highly-correlated symbols.

    Two symbols belong to the same cluster when |corr| >= corr_threshold.
    The combined value of a cluster may not exceed `max_cluster_pct` of equity.
    """

    def __init__(
        self,
        corr: pd.DataFrame | None = None,
        corr_threshold: float = 0.7,
        max_cluster_pct: float = 0.40,
    ) -> None:
        self.corr = corr
        self.corr_threshold = corr_threshold
        self.max_cluster_pct = max_cluster_pct

    def cluster_of(self, symbol: str) -> set[str]:
        """Symbols correlated with `symbol` at or above the threshold."""
        if self.corr is None or symbol not in self.corr.columns:
            return {symbol}
        row = self.corr[symbol].abs()
        return set(row[row >= self.corr_threshold].index) | {symbol}

    def cap_quantity(
        self,
        symbol: str,
        proposed_qty: float,
        price: float,
        equity: float,
        positions: dict[str, float],
        prices: dict[str, float],
    ) -> tuple[float, list[str]]:
        if self.corr is None or proposed_qty == 0 or equity <= 0:
            return proposed_qty, []

        cluster = self.cluster_of(symbol)
        cluster_value = sum(
            abs(q * prices.get(s, 0.0))
            for s, q in positions.items()
            if s in cluster and s != symbol
        )
        current_value = abs(positions.get(symbol, 0.0) * price)
        max_value = equity * self.max_cluster_pct
        room = max_value - cluster_value - current_value

        if room <= 0:
            return 0.0, [
                f"correlated cluster cap {self.max_cluster_pct:.0%} "
                f"({len(cluster)} symbols)"
            ]

        if abs(proposed_qty * price) > room:
            capped = (room / price) * np.sign(proposed_qty)
            return float(capped), [
                f"correlated cluster cap {self.max_cluster_pct:.0%}"
            ]

        return proposed_qty, []


# ── Orchestrator ──────────────────────────────────────────────────────────────

@dataclass
class RiskDecision:
    """Outcome of a risk evaluation for one proposed order."""
    symbol: str
    approved: bool
    original_quantity: float
    approved_quantity: float
    reasons: list[str] = field(default_factory=list)

    @property
    def was_reduced(self) -> bool:
        return abs(self.approved_quantity) < abs(self.original_quantity)

    def __str__(self) -> str:
        status = "APPROVED" if self.approved else "REJECTED"
        line = (
            f"[{status}] {self.symbol}: "
            f"{self.original_quantity:.0f} → {self.approved_quantity:.0f}"
        )
        return line + (f"  ({'; '.join(self.reasons)})" if self.reasons else "")


class RiskManager:
    """
    Central risk gate. Every order passes through `evaluate()` before execution.

    Typical wiring:
        rm = RiskManager(max_drawdown_pct=0.15, kelly_fraction=0.25)
        rm.update_equity(portfolio_equity)          # once per bar
        decision = rm.evaluate(symbol, qty, price, equity, positions, prices)
        if decision.approved:
            send_order(decision.approved_quantity)
    """

    def __init__(
        self,
        max_position_pct: float = 0.20,
        max_drawdown_pct: float = 0.15,
        resume_drawdown_pct: float = 0.05,
        kelly_fraction_scale: float = 0.25,
        max_gross_exposure: float = 1.0,
        max_net_exposure: float = 1.0,
        max_open_positions: int = 10,
        corr: pd.DataFrame | None = None,
        corr_threshold: float = 0.7,
        max_cluster_pct: float = 0.40,
    ) -> None:
        self.kelly_fraction_scale = kelly_fraction_scale
        self.drawdown_guard = DrawdownGuard(max_drawdown_pct, resume_drawdown_pct)
        self.exposure_limits = ExposureLimits(
            max_position_pct=max_position_pct,
            max_gross_exposure=max_gross_exposure,
            max_net_exposure=max_net_exposure,
            max_open_positions=max_open_positions,
        )
        self.correlation_limiter = CorrelationLimiter(
            corr=corr,
            corr_threshold=corr_threshold,
            max_cluster_pct=max_cluster_pct,
        )
        self.decisions: list[RiskDecision] = []

    @classmethod
    def from_config(cls, risk_cfg, corr: pd.DataFrame | None = None) -> "RiskManager":
        """Build from a RiskConfig (config/settings.py)."""
        return cls(
            max_position_pct=risk_cfg.max_position_pct,
            max_drawdown_pct=risk_cfg.max_drawdown_pct,
            kelly_fraction_scale=risk_cfg.kelly_fraction,
            corr=corr,
        )

    def reset(self) -> None:
        self.drawdown_guard.reset()
        self.decisions = []

    # ── Equity tracking ───────────────────────────────────────────────────────

    def update_equity(self, equity: float) -> bool:
        """Feed latest equity. Returns True if trading is currently halted."""
        return self.drawdown_guard.update(equity)

    @property
    def is_halted(self) -> bool:
        return self.drawdown_guard.is_halted

    # ── Sizing ────────────────────────────────────────────────────────────────

    def kelly_size(
        self,
        equity: float,
        price: float,
        returns: pd.Series | None = None,
        win_rate: float | None = None,
        avg_win: float | None = None,
        avg_loss: float | None = None,
        signal_strength: float = 1.0,
    ) -> float:
        """
        Fractional-Kelly position size in shares.

        Supply either a realised `returns` series, or explicit
        (win_rate, avg_win, avg_loss). Falls back to the per-position cap
        when no edge statistics are available.
        """
        if price <= 0 or equity <= 0:
            return 0.0

        if returns is not None:
            f = kelly_from_returns(returns)
        elif None not in (win_rate, avg_win, avg_loss):
            f = kelly_fraction(win_rate, avg_win, avg_loss)
        else:
            f = self.exposure_limits.max_position_pct / self.kelly_fraction_scale

        scaled = f * self.kelly_fraction_scale * float(np.clip(signal_strength, 0.0, 1.0))
        scaled = min(scaled, self.exposure_limits.max_position_pct)
        return float(equity * scaled / price)

    # ── Full evaluation ───────────────────────────────────────────────────────

    def evaluate(
        self,
        symbol: str,
        proposed_qty: float,
        price: float,
        equity: float,
        positions: dict[str, float] | None = None,
        prices: dict[str, float] | None = None,
    ) -> RiskDecision:
        """
        Run a proposed order through every risk check.

        Returns a RiskDecision whose `approved_quantity` is the largest size
        that satisfies all constraints (0 when the order is rejected outright).
        """
        positions = positions or {}
        prices = dict(prices or {})
        prices.setdefault(symbol, price)

        reasons: list[str] = []
        current_qty = positions.get(symbol, 0.0)
        is_reducing = current_qty != 0 and np.sign(proposed_qty) != np.sign(current_qty)

        # Halt blocks new risk, but never blocks closing existing risk.
        if self.drawdown_guard.is_halted and not is_reducing:
            decision = RiskDecision(
                symbol=symbol,
                approved=False,
                original_quantity=proposed_qty,
                approved_quantity=0.0,
                reasons=[
                    f"trading halted: drawdown "
                    f"{abs(self.drawdown_guard.current_drawdown):.1%} >= "
                    f"{self.drawdown_guard.max_drawdown_pct:.1%}"
                ],
            )
            self.decisions.append(decision)
            return decision

        qty = proposed_qty

        qty, exposure_reasons = self.exposure_limits.cap_quantity(
            symbol, qty, price, equity, positions, prices
        )
        reasons.extend(exposure_reasons)

        if qty != 0 and not is_reducing:
            qty, corr_reasons = self.correlation_limiter.cap_quantity(
                symbol, qty, price, equity, positions, prices
            )
            reasons.extend(corr_reasons)

        # Drop sub-share residuals
        if abs(qty) < 1.0:
            qty = 0.0

        decision = RiskDecision(
            symbol=symbol,
            approved=qty != 0.0,
            original_quantity=proposed_qty,
            approved_quantity=float(qty),
            reasons=reasons,
        )
        self.decisions.append(decision)
        return decision

    # ── Reporting ─────────────────────────────────────────────────────────────

    def decision_log(self) -> pd.DataFrame:
        """All decisions made so far as a DataFrame."""
        if not self.decisions:
            return pd.DataFrame(
                columns=["symbol", "approved", "original_quantity",
                         "approved_quantity", "reasons"]
            )
        return pd.DataFrame([
            {
                "symbol": d.symbol,
                "approved": d.approved,
                "original_quantity": d.original_quantity,
                "approved_quantity": d.approved_quantity,
                "reasons": "; ".join(d.reasons),
            }
            for d in self.decisions
        ])

    def summary(self) -> dict:
        n = len(self.decisions)
        approved = sum(1 for d in self.decisions if d.approved)
        reduced = sum(1 for d in self.decisions if d.was_reduced and d.approved)
        return {
            "n_decisions": n,
            "n_approved": approved,
            "n_rejected": n - approved,
            "n_reduced": reduced,
            "approval_rate": approved / n if n else 0.0,
            "halt_count": self.drawdown_guard.halt_count,
            "current_drawdown": self.drawdown_guard.current_drawdown,
            "is_halted": self.drawdown_guard.is_halted,
        }
