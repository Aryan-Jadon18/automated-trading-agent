from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_CONFIG_DIR = Path(__file__).parent


class DataConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DATA_")
    provider: Literal["yfinance", "alpaca"] = "yfinance"
    symbols: list[str] = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"]
    interval: str = "1d"
    lookback_days: int = 1825


class StrategyConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STRATEGY_")
    name: Literal["technical", "ml", "rl"] = "technical"
    short_window: int = 20
    long_window: int = 50
    rsi_period: int = 14


class BacktestConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BACKTEST_")
    initial_capital: float = 100_000.0
    commission: float = 0.001
    slippage: float = 0.0005


class MonteCarloConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MC_")
    n_simulations: int = 10_000
    n_days: int = 252
    model: Literal["gbm", "fat_tail", "regime"] = "gbm"
    confidence_levels: list[float] = [0.95, 0.99]


class RiskConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RISK_")
    max_position_pct: float = 0.20
    max_drawdown_pct: float = 0.15
    kelly_fraction: float = 0.25


class ModelsConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MODELS_")
    xgb_n_estimators: int = 300
    xgb_learning_rate: float = 0.05
    lstm_hidden: int = 128
    lstm_layers: int = 2
    lstm_seq_len: int = 60
    rl_timesteps: int = 500_000


class ExecutionConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EXEC_")
    mode: Literal["paper", "live"] = "paper"
    broker: str = "alpaca"
    rebalance_freq: Literal["daily", "weekly", "signal"] = "daily"
    # Set via env: ALPACA_API_KEY, ALPACA_SECRET_KEY
    alpaca_api_key: str = Field(default="", alias="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(default="", alias="ALPACA_SECRET_KEY")


class Settings(BaseSettings):
    """Root config — loads from default.yaml, overridden by env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data: DataConfig = Field(default_factory=DataConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    montecarlo: MonteCarloConfig = Field(default_factory=MonteCarloConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @classmethod
    def from_yaml(cls, path: Path = _CONFIG_DIR / "default.yaml") -> "Settings":
        raw = yaml.safe_load(path.read_text())
        return cls(
            data=DataConfig(**raw.get("data", {})),
            strategy=StrategyConfig(**raw.get("strategy", {})),
            backtest=BacktestConfig(**raw.get("backtest", {})),
            montecarlo=MonteCarloConfig(**raw.get("montecarlo", {})),
            risk=RiskConfig(**raw.get("risk", {})),
            models=ModelsConfig(**raw.get("models", {})),
            execution=ExecutionConfig(**raw.get("execution", {})),
        )


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_yaml()
