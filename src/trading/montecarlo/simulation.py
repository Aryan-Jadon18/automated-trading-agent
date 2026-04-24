"""
Monte Carlo simulation engine.

Three return models:
  - gbm        : Geometric Brownian Motion (normal returns)
  - fat_tail   : GBM with Student-t shocks (captures fat tails)
  - regime     : 2-state Markov regime-switching (bull / bear)

Multi-asset support via Cholesky decomposition of the correlation matrix.
All simulations return shape (n_simulations, n_days+1, n_assets) price paths.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ── Parameter estimation ──────────────────────────────────────────────────────

@dataclass
class AssetParams:
    symbol: str
    mu: float       # annualised drift
    sigma: float    # annualised volatility
    nu: float = 5.0 # Student-t degrees of freedom (fat-tail model)


def estimate_params(
    prices: pd.Series,
    trading_days: int = 252,
) -> AssetParams:
    """Estimate drift + vol from historical close prices."""
    log_ret = np.log(prices / prices.shift(1)).dropna()
    daily_mu = float(log_ret.mean())
    daily_sigma = float(log_ret.std(ddof=1))
    # Fit Student-t df via method-of-moments (excess kurtosis → df)
    excess_kurt = float(log_ret.kurtosis())
    nu = max(2.5, 6.0 / excess_kurt + 4.0) if excess_kurt > 0 else 30.0
    return AssetParams(
        symbol=str(prices.name or "ASSET"),
        mu=daily_mu * trading_days,
        sigma=daily_sigma * np.sqrt(trading_days),
        nu=nu,
    )


def estimate_correlation(prices_df: pd.DataFrame) -> np.ndarray:
    """Pearson correlation matrix of log returns."""
    log_ret = np.log(prices_df / prices_df.shift(1)).dropna()
    return log_ret.corr().values


# ── Correlated shock generator ────────────────────────────────────────────────

def _cholesky_shocks(
    n_sims: int,
    n_days: int,
    n_assets: int,
    corr: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate correlated standard-normal shocks via Cholesky.
    Returns shape (n_sims, n_days, n_assets).
    """
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_sims, n_days, n_assets))
    # Apply correlation: each day slice → (n_sims, n_assets) @ L.T
    return z @ L.T


def _cholesky_t_shocks(
    n_sims: int,
    n_days: int,
    n_assets: int,
    corr: np.ndarray,
    nu: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Correlated Student-t shocks (shared df across assets for simplicity).
    Returns shape (n_sims, n_days, n_assets).
    """
    z = _cholesky_shocks(n_sims, n_days, n_assets, corr, rng)
    chi2 = rng.chisquare(df=nu, size=(n_sims, n_days, 1))
    return z / np.sqrt(chi2 / nu)


# ── GBM ───────────────────────────────────────────────────────────────────────

def simulate_gbm(
    params: list[AssetParams],
    corr: np.ndarray,
    n_simulations: int = 10_000,
    n_days: int = 252,
    S0: list[float] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Multi-asset GBM with Cholesky-correlated Brownian shocks.

    Returns price paths of shape (n_simulations, n_days+1, n_assets).
    S0 defaults to 1.0 per asset (returns base-1 paths).
    """
    rng = np.random.default_rng(seed)
    n_assets = len(params)
    dt = 1.0 / 252

    if S0 is None:
        S0 = [1.0] * n_assets

    # (n_sims, n_days, n_assets)
    z = _cholesky_shocks(n_simulations, n_days, n_assets, corr, rng)

    paths = np.empty((n_simulations, n_days + 1, n_assets))
    paths[:, 0, :] = S0

    for a, p in enumerate(params):
        daily_mu = p.mu / 252
        daily_sigma = p.sigma / np.sqrt(252)
        drift = (daily_mu - 0.5 * daily_sigma ** 2) * dt
        diffusion = daily_sigma * np.sqrt(dt) * z[:, :, a]
        log_increments = drift + diffusion           # (n_sims, n_days)
        paths[:, 1:, a] = S0[a] * np.exp(np.cumsum(log_increments, axis=1))

    return paths


# ── Fat-tail (Student-t GBM) ──────────────────────────────────────────────────

def simulate_fat_tail(
    params: list[AssetParams],
    corr: np.ndarray,
    n_simulations: int = 10_000,
    n_days: int = 252,
    S0: list[float] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    GBM variant where shocks follow a multivariate Student-t distribution.
    Uses each asset's estimated nu (degrees of freedom).
    Average nu across assets for the shared chi-squared draw.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(params)
    dt = 1.0 / 252

    if S0 is None:
        S0 = [1.0] * n_assets

    avg_nu = float(np.mean([p.nu for p in params]))
    z = _cholesky_t_shocks(n_simulations, n_days, n_assets, corr, avg_nu, rng)

    paths = np.empty((n_simulations, n_days + 1, n_assets))
    paths[:, 0, :] = S0

    for a, p in enumerate(params):
        daily_mu = p.mu / 252
        daily_sigma = p.sigma / np.sqrt(252)
        # Scale t-shocks to match target volatility
        t_scale = daily_sigma * np.sqrt(dt) * np.sqrt((avg_nu - 2) / avg_nu)
        drift = (daily_mu - 0.5 * daily_sigma ** 2) * dt
        log_increments = drift + t_scale * z[:, :, a]
        paths[:, 1:, a] = S0[a] * np.exp(np.cumsum(log_increments, axis=1))

    return paths


# ── Regime-switching ──────────────────────────────────────────────────────────

@dataclass
class RegimeParams:
    """Bull / bear regime parameters for a single asset."""
    mu_bull: float
    sigma_bull: float
    mu_bear: float
    sigma_bear: float
    p_bull_to_bear: float = 0.02   # daily transition probability
    p_bear_to_bull: float = 0.10


def estimate_regime_params(prices: pd.Series) -> RegimeParams:
    """
    Simple threshold-based regime estimation.
    Bull: rolling 60-day return > 0. Bear: otherwise.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    rolling = log_ret.rolling(60).mean().dropna()
    bull_mask = rolling > 0
    bull_ret = log_ret[bull_mask.index][bull_mask]
    bear_ret = log_ret[bull_mask.index][~bull_mask]

    annualise = 252
    return RegimeParams(
        mu_bull=float(bull_ret.mean() * annualise),
        sigma_bull=float(bull_ret.std() * np.sqrt(annualise)) if len(bull_ret) > 1 else 0.15,
        mu_bear=float(bear_ret.mean() * annualise),
        sigma_bear=float(bear_ret.std() * np.sqrt(annualise)) if len(bear_ret) > 1 else 0.30,
    )


def simulate_regime(
    regime_params: list[RegimeParams],
    corr: np.ndarray,
    n_simulations: int = 10_000,
    n_days: int = 252,
    S0: list[float] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Multi-asset regime-switching simulation with shared regime states
    (assets switch regimes together — captures crisis correlation).
    """
    rng = np.random.default_rng(seed)
    n_assets = len(regime_params)
    dt = 1.0 / 252

    if S0 is None:
        S0 = [1.0] * n_assets

    z = _cholesky_shocks(n_simulations, n_days, n_assets, corr, rng)
    regime_draws = rng.random((n_simulations, n_days))

    paths = np.empty((n_simulations, n_days + 1, n_assets))
    paths[:, 0, :] = S0

    # Regime state: 0 = bull, 1 = bear
    state = np.zeros(n_simulations, dtype=int)

    for t in range(n_days):
        # Regime transitions
        in_bull = state == 0
        in_bear = state == 1
        p_switch_bull = np.array([regime_params[0].p_bull_to_bear] * n_simulations)
        p_switch_bear = np.array([regime_params[0].p_bear_to_bull] * n_simulations)
        state = np.where(in_bull & (regime_draws[:, t] < p_switch_bull), 1, state)
        state = np.where(in_bear & (regime_draws[:, t] < p_switch_bear), 0, state)

        for a, rp in enumerate(regime_params):
            mu_t = np.where(state == 0, rp.mu_bull / 252, rp.mu_bear / 252)
            sig_t = np.where(state == 0, rp.sigma_bull / np.sqrt(252), rp.sigma_bear / np.sqrt(252))
            drift = (mu_t - 0.5 * sig_t ** 2) * dt
            shock = sig_t * np.sqrt(dt) * z[:, t, a]
            paths[:, t + 1, a] = paths[:, t, a] * np.exp(drift + shock)

    return paths
