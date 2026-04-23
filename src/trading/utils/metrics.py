"""Performance metrics for backtesting and live monitoring."""
from __future__ import annotations

import numpy as np
import pandas as pd


def total_return(equity: pd.Series) -> float:
    return equity.iloc[-1] / equity.iloc[0] - 1


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    n_years = len(equity) / periods_per_year
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1


def sharpe(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - risk_free / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - risk_free / periods_per_year
    downside_std = excess[excess < 0].std()
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1
    return float(drawdown.min())


def calmar(equity: pd.Series, periods_per_year: int = 252) -> float:
    mdd = abs(max_drawdown(equity))
    return cagr(equity, periods_per_year) / mdd if mdd > 0 else np.inf


def win_rate(returns: pd.Series) -> float:
    return float((returns > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return gains / losses if losses > 0 else np.inf


def summary(equity: pd.Series, returns: pd.Series, periods_per_year: int = 252) -> dict:
    return {
        "total_return": total_return(equity),
        "cagr": cagr(equity, periods_per_year),
        "sharpe": sharpe(returns, periods_per_year=periods_per_year),
        "sortino": sortino(returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar(equity, periods_per_year),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
    }
