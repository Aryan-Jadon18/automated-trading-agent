"""Tests for Monte Carlo simulation and risk analytics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.trading.montecarlo.analysis import (
    MCResult,
    cvar,
    drawdown_distribution,
    path_max_drawdown,
    return_distribution,
    run_analysis,
    terminal_returns,
    var,
)
from src.trading.montecarlo.simulation import (
    AssetParams,
    estimate_params,
    estimate_correlation,
    simulate_fat_tail,
    simulate_gbm,
    simulate_regime,
    RegimeParams,
    estimate_regime_params,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

N_SIMS = 500      # small for fast tests
N_DAYS = 63       # one quarter


@pytest.fixture
def single_asset_params() -> list[AssetParams]:
    return [AssetParams(symbol="SPY", mu=0.08, sigma=0.16)]


@pytest.fixture
def two_asset_params() -> list[AssetParams]:
    return [
        AssetParams(symbol="SPY", mu=0.08, sigma=0.16),
        AssetParams(symbol="QQQ", mu=0.10, sigma=0.20),
    ]


@pytest.fixture
def identity_corr() -> np.ndarray:
    return np.eye(1)


@pytest.fixture
def two_asset_corr() -> np.ndarray:
    return np.array([[1.0, 0.75], [0.75, 1.0]])


@pytest.fixture
def gbm_paths(single_asset_params, identity_corr) -> np.ndarray:
    return simulate_gbm(
        single_asset_params, identity_corr,
        n_simulations=N_SIMS, n_days=N_DAYS, seed=42,
    )


@pytest.fixture
def multi_paths(two_asset_params, two_asset_corr) -> np.ndarray:
    return simulate_gbm(
        two_asset_params, two_asset_corr,
        n_simulations=N_SIMS, n_days=N_DAYS, seed=42,
    )


# ── Parameter estimation ──────────────────────────────────────────────────────

def test_estimate_params_returns_asset_params():
    np.random.seed(0)
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(500) * 0.01)), name="TEST")
    p = estimate_params(prices)
    assert isinstance(p, AssetParams)
    assert p.sigma > 0
    assert p.nu > 2


def test_estimate_correlation_shape():
    np.random.seed(1)
    prices = pd.DataFrame({
        "A": 100 * np.exp(np.cumsum(np.random.randn(300) * 0.01)),
        "B": 100 * np.exp(np.cumsum(np.random.randn(300) * 0.01)),
    })
    corr = estimate_correlation(prices)
    assert corr.shape == (2, 2)
    assert corr[0, 0] == pytest.approx(1.0)
    assert -1.0 <= corr[0, 1] <= 1.0


# ── GBM paths shape and properties ───────────────────────────────────────────

def test_gbm_paths_shape(gbm_paths):
    assert gbm_paths.shape == (N_SIMS, N_DAYS + 1, 1)


def test_gbm_paths_start_at_s0(gbm_paths):
    assert (gbm_paths[:, 0, 0] == pytest.approx(1.0)).all()


def test_gbm_paths_positive(gbm_paths):
    assert (gbm_paths > 0).all()


def test_gbm_multi_asset_shape(multi_paths):
    assert multi_paths.shape == (N_SIMS, N_DAYS + 1, 2)


def test_gbm_deterministic_with_seed(single_asset_params, identity_corr):
    p1 = simulate_gbm(single_asset_params, identity_corr, N_SIMS, N_DAYS, seed=7)
    p2 = simulate_gbm(single_asset_params, identity_corr, N_SIMS, N_DAYS, seed=7)
    np.testing.assert_array_equal(p1, p2)


def test_gbm_different_seeds_differ(single_asset_params, identity_corr):
    p1 = simulate_gbm(single_asset_params, identity_corr, N_SIMS, N_DAYS, seed=1)
    p2 = simulate_gbm(single_asset_params, identity_corr, N_SIMS, N_DAYS, seed=2)
    assert not np.allclose(p1, p2)


# ── Fat-tail paths ────────────────────────────────────────────────────────────

def test_fat_tail_shape(single_asset_params, identity_corr):
    paths = simulate_fat_tail(single_asset_params, identity_corr, N_SIMS, N_DAYS, seed=42)
    assert paths.shape == (N_SIMS, N_DAYS + 1, 1)


def test_fat_tail_paths_positive(single_asset_params, identity_corr):
    paths = simulate_fat_tail(single_asset_params, identity_corr, N_SIMS, N_DAYS, seed=42)
    assert (paths > 0).all()


def test_fat_tail_fatter_than_gbm(single_asset_params, identity_corr):
    """Fat-tail should produce more extreme terminal returns than GBM."""
    gbm = simulate_gbm(single_asset_params, identity_corr, 2000, N_DAYS, seed=99)
    ft = simulate_fat_tail(single_asset_params, identity_corr, 2000, N_DAYS, seed=99)
    gbm_rets = terminal_returns(gbm)
    ft_rets = terminal_returns(ft)
    # Fat-tail kurtosis should exceed GBM kurtosis
    assert pd.Series(ft_rets).kurtosis() > pd.Series(gbm_rets).kurtosis()


# ── Regime switching ──────────────────────────────────────────────────────────

def test_regime_shape():
    rp = [RegimeParams(mu_bull=0.12, sigma_bull=0.14, mu_bear=-0.05, sigma_bear=0.30)]
    corr = np.eye(1)
    paths = simulate_regime(rp, corr, n_simulations=N_SIMS, n_days=N_DAYS, seed=42)
    assert paths.shape == (N_SIMS, N_DAYS + 1, 1)


def test_regime_paths_positive():
    rp = [RegimeParams(mu_bull=0.12, sigma_bull=0.14, mu_bear=-0.05, sigma_bear=0.30)]
    corr = np.eye(1)
    paths = simulate_regime(rp, corr, n_simulations=N_SIMS, n_days=N_DAYS, seed=42)
    assert (paths > 0).all()


def test_estimate_regime_params():
    np.random.seed(5)
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(600) * 0.01)))
    rp = estimate_regime_params(prices)
    assert rp.sigma_bull > 0
    assert rp.sigma_bear > 0


# ── VaR / CVaR ────────────────────────────────────────────────────────────────

def test_var_positive(gbm_paths):
    v = var(gbm_paths, confidence=0.95)
    assert v >= 0


def test_var_99_ge_95(gbm_paths):
    v95 = var(gbm_paths, confidence=0.95)
    v99 = var(gbm_paths, confidence=0.99)
    assert v99 >= v95


def test_cvar_ge_var(gbm_paths):
    v = var(gbm_paths, confidence=0.95)
    cv = cvar(gbm_paths, confidence=0.95)
    assert cv >= v


def test_var_horizon(gbm_paths):
    v_full = var(gbm_paths, confidence=0.95)
    v_1d = var(gbm_paths, confidence=0.95, horizon_days=1)
    assert v_full >= v_1d     # longer horizon → larger VaR


# ── Drawdown ──────────────────────────────────────────────────────────────────

def test_mdd_non_positive(gbm_paths):
    mdd = path_max_drawdown(gbm_paths)
    assert (mdd <= 0).all()


def test_mdd_shape(gbm_paths):
    mdd = path_max_drawdown(gbm_paths)
    assert mdd.shape == (N_SIMS,)


def test_drawdown_distribution_keys(gbm_paths):
    dd = drawdown_distribution(gbm_paths, [0.95, 0.99])
    assert "mean_mdd" in dd
    assert "worst_mdd" in dd
    assert "mdd_at_95pct" in dd
    assert "mdd_at_99pct" in dd


# ── Return distribution ───────────────────────────────────────────────────────

def test_return_distribution_prob_profit_in_range(gbm_paths):
    rd = return_distribution(gbm_paths)
    assert 0.0 <= rd["prob_profit"] <= 1.0


def test_return_distribution_percentiles_ordered(gbm_paths):
    rd = return_distribution(gbm_paths)
    assert rd["p5"] <= rd["p25"] <= rd["p50"] <= rd["p75"] <= rd["p95"]


# ── run_analysis integration ──────────────────────────────────────────────────

def test_run_analysis_returns_mcresult(multi_paths):
    result = run_analysis(multi_paths, ["SPY", "QQQ"], model="gbm")
    assert isinstance(result, MCResult)
    assert len(result.var_95) == 2
    assert len(result.drawdown_stats) == 2


def test_run_analysis_summary_shape(multi_paths):
    result = run_analysis(multi_paths, ["SPY", "QQQ"], model="gbm")
    df = result.summary()
    assert df.shape == (2, 8)
    assert set(df.index) == {"SPY", "QQQ"}


def test_run_analysis_fat_tail_vs_gbm_cvar(single_asset_params, identity_corr):
    """Fat-tail CVaR should be >= GBM CVaR at same confidence."""
    gbm = simulate_gbm(single_asset_params, identity_corr, 2000, N_DAYS, seed=10)
    ft = simulate_fat_tail(single_asset_params, identity_corr, 2000, N_DAYS, seed=10)
    r_gbm = run_analysis(gbm, ["SPY"], model="gbm")
    r_ft = run_analysis(ft, ["SPY"], model="fat_tail")
    assert r_ft.cvar_95[0] >= r_gbm.cvar_95[0]
