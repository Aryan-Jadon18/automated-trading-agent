"""Tests for XGBoost, LSTM, and PPO RL models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.trading.models.base import BaseModel, make_labels
from src.trading.models.xgb import XGBSignalModel
from src.trading.models.lstm import LSTMPriceModel, _make_sequences
from src.trading.models.rl_agent import TradingEnv, PPORLAgent


# ── Shared fixtures ───────────────────────────────────────────────────────────

N = 400

@pytest.fixture
def price_series() -> pd.Series:
    np.random.seed(42)
    return pd.Series(100 * np.exp(np.cumsum(np.random.randn(N) * 0.01 + 0.0003)), name="close")


@pytest.fixture
def feature_df(price_series) -> pd.DataFrame:
    """Minimal synthetic feature matrix."""
    np.random.seed(0)
    close = price_series
    return pd.DataFrame({
        "ret_1": close.pct_change().fillna(0),
        "ret_5": close.pct_change(5).fillna(0),
        "sma_ratio": (close / close.rolling(20).mean()).fillna(1),
        "vol_20": close.pct_change().rolling(20).std().fillna(0.01),
        "rsi": 50 + np.random.randn(N) * 10,
    }, index=close.index)


@pytest.fixture
def log_returns(price_series) -> pd.Series:
    return np.log(price_series / price_series.shift(1)).fillna(0)


# ── make_labels ───────────────────────────────────────────────────────────────

def test_make_labels_values(price_series):
    labels = make_labels(price_series, forward_days=1)
    assert set(labels.dropna().unique()).issubset({-1, 0, 1})


def test_make_labels_length(price_series):
    labels = make_labels(price_series, forward_days=1)
    assert len(labels) == len(price_series)


# ── XGBoost ───────────────────────────────────────────────────────────────────

def test_xgb_fit_predict(feature_df, price_series):
    labels = make_labels(price_series).dropna()
    X = feature_df.loc[labels.index]
    model = XGBSignalModel(n_estimators=50)
    model.fit(X, labels)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds).issubset({-1, 0, 1})


def test_xgb_predict_signal_returns_series(feature_df, price_series):
    labels = make_labels(price_series).dropna()
    X = feature_df.loc[labels.index]
    model = XGBSignalModel(n_estimators=50)
    model.fit(X, labels)
    sig = model.predict_signal(X)
    assert isinstance(sig, pd.Series)
    assert len(sig) == len(X)


def test_xgb_not_fitted_raises():
    model = XGBSignalModel()
    with pytest.raises(RuntimeError):
        model.predict(pd.DataFrame({"a": [1, 2, 3]}))


def test_xgb_feature_importance(feature_df, price_series):
    labels = make_labels(price_series).dropna()
    X = feature_df.loc[labels.index]
    model = XGBSignalModel(n_estimators=50)
    model.fit(X, labels)
    imp = model.feature_importance(list(X.columns))
    assert len(imp) == X.shape[1]
    assert (imp >= 0).all()


def test_xgb_walk_forward_length(feature_df, price_series):
    model = XGBSignalModel(n_estimators=30)
    sig = model.walk_forward_predict(
        feature_df, price_series,
        train_window=200, step=50, forward_days=1,
    )
    assert len(sig) == len(feature_df)
    assert set(sig.unique()).issubset({-1, 0, 1})


# ── LSTM ──────────────────────────────────────────────────────────────────────

def test_make_sequences_shape():
    X = np.random.randn(100, 5).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    xs, ys = _make_sequences(X, y, seq_len=10)
    assert xs.shape == (90, 10, 5)
    assert ys.shape == (90,)


def test_lstm_fit_predict(feature_df, log_returns):
    model = LSTMPriceModel(seq_len=10, hidden=16, n_layers=1, max_epochs=3, patience=3)
    model.fit(feature_df, log_returns, val_fraction=0.1)
    assert model.is_fitted
    preds = model.predict(feature_df)
    assert preds.shape == (len(feature_df),)
    assert set(preds).issubset({-1, 0, 1})


def test_lstm_predict_returns_shape(feature_df, log_returns):
    model = LSTMPriceModel(seq_len=10, hidden=16, n_layers=1, max_epochs=2, patience=2)
    model.fit(feature_df, log_returns, val_fraction=0.1)
    rets = model.predict_returns(feature_df)
    assert rets.shape == (len(feature_df),)


def test_lstm_not_fitted_raises():
    model = LSTMPriceModel()
    with pytest.raises(RuntimeError):
        model.predict(pd.DataFrame({"a": [1.0, 2.0]}))


def test_lstm_training_loss_recorded(feature_df, log_returns):
    model = LSTMPriceModel(seq_len=10, hidden=16, n_layers=1, max_epochs=5, patience=5)
    model.fit(feature_df, log_returns, val_fraction=0.1)
    assert len(model.train_losses) > 0


# ── TradingEnv ────────────────────────────────────────────────────────────────

def test_trading_env_reset():
    feat = np.random.randn(100, 5).astype(np.float32)
    rets = np.random.randn(100).astype(np.float32) * 0.01
    env = TradingEnv(feat, rets)
    obs, info = env.reset()
    assert obs.shape == (6,)   # 5 features + 1 position
    assert isinstance(info, dict)


def test_trading_env_step():
    feat = np.random.randn(100, 5).astype(np.float32)
    rets = np.random.randn(100).astype(np.float32) * 0.01
    env = TradingEnv(feat, rets)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(1)  # buy
    assert obs.shape == (6,)
    assert isinstance(reward, float)
    assert not terminated


def test_trading_env_terminates():
    feat = np.random.randn(10, 3).astype(np.float32)
    rets = np.zeros(10, dtype=np.float32)
    env = TradingEnv(feat, rets)
    env.reset()
    done = False
    for _ in range(15):
        _, _, done, _, _ = env.step(0)
        if done:
            break
    assert done


def test_trading_env_observation_space():
    feat = np.random.randn(50, 4).astype(np.float32)
    rets = np.zeros(50, dtype=np.float32)
    env = TradingEnv(feat, rets)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)


# ── PPO RL Agent ──────────────────────────────────────────────────────────────

def test_ppo_fit_predict(feature_df, log_returns):
    agent = PPORLAgent(total_timesteps=512, n_steps=64, batch_size=32, verbose=0)
    agent.fit(feature_df, log_returns)
    assert agent.is_fitted
    signals = agent.predict(feature_df)
    assert signals.shape == (len(feature_df),)
    assert set(signals).issubset({-1, 0, 1})


def test_ppo_predict_signal_series(feature_df, log_returns):
    agent = PPORLAgent(total_timesteps=256, n_steps=64, batch_size=32, verbose=0)
    agent.fit(feature_df, log_returns)
    sig = agent.predict_signal(feature_df)
    assert isinstance(sig, pd.Series)
    assert sig.index.equals(feature_df.index)


def test_ppo_not_fitted_raises():
    agent = PPORLAgent()
    with pytest.raises(RuntimeError):
        agent.predict(pd.DataFrame({"a": [1.0, 2.0]}))
