"""
Risk analytics on Monte Carlo path arrays.

Input convention:  paths shape (n_sims, n_days+1, n_assets)
                   or (n_sims, n_days+1) for single-asset.

All dollar / return values are expressed in the same units as S0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ── Return extraction ─────────────────────────────────────────────────────────

def terminal_returns(paths: np.ndarray, asset_idx: int = 0) -> np.ndarray:
    """Final period return for each simulation: S_T/S_0 - 1."""
    if paths.ndim == 3:
        s = paths[:, :, asset_idx]
    else:
        s = paths
    return s[:, -1] / s[:, 0] - 1


def period_log_returns(paths: np.ndarray, asset_idx: int = 0) -> np.ndarray:
    """Daily log returns shape (n_sims, n_days)."""
    if paths.ndim == 3:
        s = paths[:, :, asset_idx]
    else:
        s = paths
    return np.log(s[:, 1:] / s[:, :-1])


# ── Drawdown ──────────────────────────────────────────────────────────────────

def path_max_drawdown(paths: np.ndarray, asset_idx: int = 0) -> np.ndarray:
    """
    Max drawdown for each simulation path.
    Returns 1-D array of shape (n_sims,) with values in (-1, 0].
    """
    if paths.ndim == 3:
        s = paths[:, :, asset_idx]
    else:
        s = paths
    rolling_max = np.maximum.accumulate(s, axis=1)
    drawdowns = s / rolling_max - 1          # (n_sims, n_days+1)
    return drawdowns.min(axis=1)             # worst per path


def drawdown_distribution(
    paths: np.ndarray,
    confidence_levels: list[float] = [0.95, 0.99],
    asset_idx: int = 0,
) -> dict:
    """Summary statistics for the max-drawdown distribution."""
    mdd = path_max_drawdown(paths, asset_idx)
    result = {
        "mean_mdd": float(mdd.mean()),
        "median_mdd": float(np.median(mdd)),
        "worst_mdd": float(mdd.min()),
    }
    for cl in confidence_levels:
        result[f"mdd_at_{int(cl*100)}pct"] = float(np.percentile(mdd, (1 - cl) * 100))
    return result


# ── VaR / CVaR ────────────────────────────────────────────────────────────────

def var(
    paths: np.ndarray,
    confidence: float = 0.95,
    horizon_days: int | None = None,
    asset_idx: int = 0,
) -> float:
    """
    Historical-simulation VaR from MC paths.

    horizon_days=None → uses terminal (full-period) returns.
    Returns a positive number representing the loss at the given confidence.
    """
    if horizon_days is not None:
        if paths.ndim == 3:
            s = paths[:, :, asset_idx]
        else:
            s = paths
        end = min(horizon_days, s.shape[1] - 1)
        rets = s[:, end] / s[:, 0] - 1
    else:
        rets = terminal_returns(paths, asset_idx)

    return float(-np.percentile(rets, (1 - confidence) * 100))


def cvar(
    paths: np.ndarray,
    confidence: float = 0.95,
    horizon_days: int | None = None,
    asset_idx: int = 0,
) -> float:
    """
    Conditional VaR (Expected Shortfall) — mean loss beyond the VaR threshold.
    Returns a positive number.
    """
    if horizon_days is not None:
        if paths.ndim == 3:
            s = paths[:, :, asset_idx]
        else:
            s = paths
        end = min(horizon_days, s.shape[1] - 1)
        rets = s[:, end] / s[:, 0] - 1
    else:
        rets = terminal_returns(paths, asset_idx)

    threshold = np.percentile(rets, (1 - confidence) * 100)
    tail = rets[rets <= threshold]
    return float(-tail.mean()) if len(tail) > 0 else 0.0


# ── Return distribution ───────────────────────────────────────────────────────

def return_distribution(
    paths: np.ndarray,
    confidence_levels: list[float] = [0.05, 0.25, 0.50, 0.75, 0.95],
    asset_idx: int = 0,
) -> dict:
    """Percentile breakdown of terminal returns."""
    rets = terminal_returns(paths, asset_idx)
    result = {
        "mean": float(rets.mean()),
        "std": float(rets.std()),
        "skew": float(pd.Series(rets).skew()),
        "kurtosis": float(pd.Series(rets).kurtosis()),
        "prob_profit": float((rets > 0).mean()),
        "prob_loss_10pct": float((rets < -0.10).mean()),
        "prob_loss_20pct": float((rets < -0.20).mean()),
    }
    for p in confidence_levels:
        result[f"p{int(p*100)}"] = float(np.percentile(rets, p * 100))
    return result


# ── Strategy robustness ───────────────────────────────────────────────────────

def strategy_var_overlay(
    backtest_equity: pd.Series,
    paths: np.ndarray,
    confidence: float = 0.95,
    asset_idx: int = 0,
) -> dict:
    """
    Combine realised backtest equity with MC-forward paths to produce
    forward-looking VaR / CVaR estimates from the current equity level.
    """
    current_equity = float(backtest_equity.iloc[-1])
    rets = terminal_returns(paths, asset_idx)
    forward_equity = current_equity * (1 + rets)
    losses = current_equity - forward_equity

    threshold = np.percentile(losses, confidence * 100)
    tail_losses = losses[losses >= threshold]

    return {
        "current_equity": current_equity,
        "forward_var": float(np.percentile(losses, confidence * 100)),
        "forward_cvar": float(tail_losses.mean()) if len(tail_losses) > 0 else 0.0,
        "expected_forward_equity": float(forward_equity.mean()),
        "prob_ruin_50pct": float((forward_equity < current_equity * 0.5).mean()),
    }


# ── Master summary ────────────────────────────────────────────────────────────

@dataclass
class MCResult:
    model: str
    n_simulations: int
    n_days: int
    symbols: list[str]
    paths: np.ndarray                          # (n_sims, n_days+1, n_assets)
    var_95: list[float] = field(default_factory=list)
    var_99: list[float] = field(default_factory=list)
    cvar_95: list[float] = field(default_factory=list)
    cvar_99: list[float] = field(default_factory=list)
    drawdown_stats: list[dict] = field(default_factory=list)
    return_stats: list[dict] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        rows = []
        for i, sym in enumerate(self.symbols):
            rows.append({
                "symbol": sym,
                "var_95": self.var_95[i] if i < len(self.var_95) else None,
                "var_99": self.var_99[i] if i < len(self.var_99) else None,
                "cvar_95": self.cvar_95[i] if i < len(self.cvar_95) else None,
                "cvar_99": self.cvar_99[i] if i < len(self.cvar_99) else None,
                "mean_mdd": self.drawdown_stats[i].get("mean_mdd") if i < len(self.drawdown_stats) else None,
                "worst_mdd": self.drawdown_stats[i].get("worst_mdd") if i < len(self.drawdown_stats) else None,
                "prob_profit": self.return_stats[i].get("prob_profit") if i < len(self.return_stats) else None,
                "mean_return": self.return_stats[i].get("mean") if i < len(self.return_stats) else None,
            })
        return pd.DataFrame(rows).set_index("symbol")


def run_analysis(
    paths: np.ndarray,
    symbols: list[str],
    model: str,
    confidence_levels: list[float] = [0.95, 0.99],
) -> MCResult:
    """Run all risk analytics on a paths array and return an MCResult."""
    n_sims, n_days_plus1, n_assets = paths.shape
    result = MCResult(
        model=model,
        n_simulations=n_sims,
        n_days=n_days_plus1 - 1,
        symbols=symbols,
        paths=paths,
    )
    for a in range(n_assets):
        result.var_95.append(var(paths, 0.95, asset_idx=a))
        result.var_99.append(var(paths, 0.99, asset_idx=a))
        result.cvar_95.append(cvar(paths, 0.95, asset_idx=a))
        result.cvar_99.append(cvar(paths, 0.99, asset_idx=a))
        result.drawdown_stats.append(drawdown_distribution(paths, confidence_levels, asset_idx=a))
        result.return_stats.append(return_distribution(paths, asset_idx=a))
    return result
