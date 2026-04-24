# CLAUDE.md — Automated Trading Bot

## Project Goal
Fully automated Monte Carlo + AI-based trading bot in Python.

## Stack
- Python 3.11+, pyproject.toml (hatchling)
- Data: yfinance, pandas, numpy, ta-lib / pandas-ta
- ML: scikit-learn, xgboost, torch (LSTM/Transformer)
- RL: stable-baselines3
- Broker: alpaca-trade-api (paper + live)
- Config: pydantic-settings + YAML
- Tests: pytest

## Directory Layout
```
src/trading/
  data/         ingestion.py, features.py
  strategy/     base.py, technical.py
  backtest/     engine.py
  montecarlo/   simulation.py
  models/       base.py, lstm.py, xgb.py, rl_agent.py
  risk/         manager.py
  execution/    broker.py, orders.py
  utils/        logging.py, metrics.py
config/         settings.py, default.yaml
tests/
data/           raw/, processed/, cache/  [git-ignored]
models/         checkpoints/, saved/      [git-ignored]
```

## Progress Tracker

| Tier | Area | Status |
|------|------|--------|
| 1 | Project scaffold (pyproject, config, layout) | ✅ Done |
| 1 | Config system (pydantic-settings + YAML) | ✅ Done |
| 2 | Data ingestion (yfinance, OHLCV) | ✅ Done |
| 2 | Feature engineering pipeline | ✅ Done |
| 3 | Strategy base class + technical strategies | ✅ Done |
| 3 | Backtesting engine (event-driven, no look-ahead) | ✅ Done |
| 4 | Monte Carlo simulation (GBM + fat-tail + regime) | ✅ Done |
| 4 | VaR / CVaR / drawdown analysis | ✅ Done |
| 5 | ML models (XGBoost signal predictor) | ⬜ Pending |
| 5 | LSTM / Transformer price model | ⬜ Pending |
| 5 | RL agent (PPO execution) | ⬜ Pending |
| 6 | Risk manager (Kelly, drawdown limits) | ⬜ Pending |
| 7 | Execution layer (Alpaca paper trading) | ⬜ Pending |
| 8 | Live orchestration + monitoring | ⬜ Pending |

## Key Design Decisions
- Event-driven backtest engine (not vectorized) for realistic simulation
- Monte Carlo uses GBM with Cholesky correlation for multi-asset
- RL agent trained on backtest environment, deployed for execution
- All secrets via env vars, never committed
- Paper trading first; live trading gated behind config flag

## Backtest Architecture (Tier 3)
Event queue flow (no look-ahead bias):
`MarketEvent → Strategy.on_bar() → SignalEvent → Portfolio.on_signal() → OrderEvent → SimulatedBroker → FillEvent → Portfolio.on_fill()`
Orders fill at **next bar's open** + slippage. Strategies: SMACrossover, MeanReversion.

## Monte Carlo Architecture (Tier 4)
Three models: GBM (normal shocks), fat_tail (Student-t, auto-fit ν), regime (2-state Markov bull/bear).
Multi-asset via Cholesky decomposition. Analysis: VaR, CVaR, MDD distribution, return percentiles, strategy overlay.
Entry point: `simulate_gbm / simulate_fat_tail / simulate_regime` → `run_analysis` → `MCResult.summary()`.

## Next Step
**Tier 5:** ML models — XGBoost signal predictor, LSTM price model, PPO RL execution agent.
