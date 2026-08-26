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
  live/         orchestrator.py, state.py, monitor.py
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
| 7 | Execution layer (orders, brokers, Alpaca) | ✅ Done |
| 8 | Live orchestration + monitoring | ✅ Done |

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

## Execution Layer (Tier 7)
`execution/orders.py` — real orders, distinct from the backtest's internal events.
- `Order` with an enforced status state machine; illegal transitions raise
  `InvalidOrderTransition` rather than silently corrupting state
- Partial fills accumulate a volume-weighted `avg_fill_price`
- `OrderManager` tracks by client + broker id, and `reconcile()` reports drift
  between locally-tracked fills and the broker's actual positions

`execution/broker.py` — `BrokerBase` ABC over two implementations.
- `MockBroker`: in-memory, no network. Market orders fill at once; limit orders
  rest until `set_price()` moves the market through them. Used by all tests.
- `AlpacaBroker`: alpaca-py, imported lazily so the module loads without it.
  Maps Alpaca's 15 order states onto local statuses; broker is authoritative on drift.
- **Live trading needs two independent switches**: `allow_live=True` *and*
  `TRADING_ALLOW_LIVE=1`. Either one alone raises `LiveTradingBlocked`.

## Live Layer (Tier 8)
`live/state.py` — `TradingState`, everything needed to survive a restart.
- `last_bar_time` per symbol is the **idempotency key**: a bar is acted on at most once
- Atomic saves (temp file + `os.replace`), so a kill mid-write keeps the last good state
- A **corrupt state file raises rather than resetting** — silently starting fresh would
  lose the record of which bars were traded and could double-submit orders

`live/monitor.py` — health checks + alert dispatch.
- Failing **CRITICAL** check aborts the cycle before any order is placed
- Standard checks: broker connectivity, position reconciliation, drawdown, data freshness
- A check that raises counts as failed; a broken alert sink never reaches the trading path
- Alerts dedupe per title, but **CRITICAL always fires**

`live/orchestrator.py` — `LiveTrader`, the loop that joins every tier.
`fetch → new-bar check → Strategy.on_bar → size → RiskManager.evaluate → Broker.submit`
- `startup()` **warms the strategy with history** — without this a restarted process is
  blind for its whole SMA window and can miss a crossing entirely (found by running it)
- Startup **halts on position drift**: positions moved outside the session mean local
  sizing would be wrong
- `run_once()` is the testable unit; `run_forever()` schedules it (inject `sleep_fn`)
- `dry_run=True` runs the full decision path without submitting
- SIGINT/SIGTERM finish the cycle, optionally cancel working orders, always persist

## Testing
268 tests passing (`python3 -m pytest tests/ -q`). Heavy deps needed for the ML tier:
`pip install numpy pandas pytest pyyaml pydantic-settings scikit-learn structlog xgboost torch gymnasium stable-baselines3`
Note: PyTorch's own CDN is blocked by the proxy — install `torch` from default PyPI.

## Status
All 8 tiers complete. To run live against Alpaca paper:
1. `export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...`
2. `AlpacaBroker.from_config(settings.execution)` → `LiveTrader(...)` → `run_forever()`
3. Start with `dry_run=True` and confirm the decision log before removing it.

Live trading (real money) additionally needs `allow_live=True` **and**
`TRADING_ALLOW_LIVE=1`; either alone raises `LiveTradingBlocked`.

### Possible next work
- Backfill an integration test against the real Alpaca paper API (needs credentials)
- Multi-strategy allocation across an ensemble (XGB + LSTM + RL voting)
- Walk-forward retraining scheduled inside the live loop
- Web dashboard over `LiveTrader.status()`
