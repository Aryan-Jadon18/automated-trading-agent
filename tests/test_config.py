"""Unit tests for config/settings."""
from config.settings import Settings


def test_settings_loads_defaults():
    s = Settings.from_yaml()
    assert s.data.provider == "yfinance"
    assert s.backtest.initial_capital == 100_000.0
    assert s.montecarlo.n_simulations == 10_000
    assert s.risk.kelly_fraction == 0.25
    assert s.execution.mode == "paper"


def test_settings_symbols_non_empty():
    s = Settings.from_yaml()
    assert len(s.data.symbols) > 0


def test_settings_confidence_levels():
    s = Settings.from_yaml()
    assert 0.95 in s.montecarlo.confidence_levels
