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
| 5 | ML models (XGBoost signal predictor) | ✅ Done |
| 5 | LSTM price model | ✅ Done |
| 5 | RL agent (PPO execution) | ✅ Done |
| 6 | Risk manager (Kelly, drawdown, exposure, correlation) | ✅ Done |
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

## ML Layer (Tier 5)
- `models/base.py`: `BaseModel` ABC (fit/predict/save/load) + `make_labels()` (forward-return labelling)
- `models/xgb.py`: `XGBSignalModel` — XGBoost + StandardScaler, threshold-gated signals, walk-forward CV
- `models/lstm.py`: `LSTMPriceModel` — PyTorch LSTM seq-to-one, early stopping, GPU-aware
- `models/rl_agent.py`: `TradingEnv` (Gymnasium) + `PPORLAgent` (SB3 PPO) — position-based reward

## Risk Layer (Tier 6)
`risk/manager.py` — every order passes `RiskManager.evaluate()` before execution.
- `kelly_fraction(win_rate, avg_win, avg_loss)` / `kelly_from_returns(series)` — fractional Kelly
- `DrawdownGuard` — halts on breach, hysteresis resume (won't re-enter until recovered)
- `ExposureLimits` — per-position / gross / net / max-open-positions caps
- `CorrelationLimiter` — caps combined exposure to clusters where |corr| >= threshold
- Returns `RiskDecision(approved, approved_quantity, reasons)`; orders are **shrunk, not just rejected**
- **Closing/reducing a position is never blocked**, even while halted
- Wired into `Portfolio(risk_manager=...)` and `run_backtest(risk_manager=...)`

## Testing
119 tests passing (`python3 -m pytest tests/ -q`). Heavy deps needed for the ML tier:
`pip install numpy pandas pytest pyyaml pydantic-settings scikit-learn structlog xgboost torch gymnasium stable-baselines3`
Note: PyTorch's own CDN is blocked by the proxy — install `torch` from default PyPI.

## Next Step
**Tier 7:** Execution layer — Alpaca paper-trading broker, order management, reconciliation.
