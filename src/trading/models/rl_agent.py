"""
PPO RL execution agent via stable-baselines3.

TradingEnv wraps a feature matrix into a Gymnasium environment:
  - Observation: normalized feature vector at each bar
  - Action:      Discrete(3) — 0=hold, 1=buy, 2=sell
  - Reward:      step log-return × position (PnL), penalised for excessive trading
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.trading.models.base import BaseModel


# ── Gymnasium environment ─────────────────────────────────────────────────────

class TradingEnv(gym.Env):
    """
    Single-asset trading environment for RL training.

    Each episode runs through the entire feature matrix once.
    The agent decides at each step: hold / buy (go long) / sell (go short / exit).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,        # (n_bars, n_features) — already normalised
        log_returns: np.ndarray,     # (n_bars,) — actual bar returns
        trade_cost: float = 0.001,   # round-trip commission per position change
        hold_penalty: float = 0.0,   # optional penalty for idle holding
    ) -> None:
        super().__init__()
        assert len(features) == len(log_returns)
        self.features = features.astype(np.float32)
        self.log_returns = log_returns.astype(np.float32)
        self.trade_cost = trade_cost
        self.hold_penalty = hold_penalty
        self.n_bars = len(features)
        self.n_features = features.shape[1]

        # +1 feature: current position (one-hot encoded as scalar -1/0/1)
        obs_dim = self.n_features + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # 0 = hold, 1 = long, 2 = short
        self.action_space = spaces.Discrete(3)

        self._step = 0
        self._position = 0   # -1 / 0 / +1

    def _obs(self) -> np.ndarray:
        feat = self.features[self._step]
        return np.append(feat, float(self._position)).astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step = 0
        self._position = 0
        return self._obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        prev_pos = self._position

        # Map action → position
        target = {0: 0, 1: 1, 2: -1}[int(action)]

        # Trading cost on position change
        cost = self.trade_cost if target != prev_pos else 0.0
        self._position = target

        # Reward = position × bar return − cost − idle penalty
        bar_ret = float(self.log_returns[self._step])
        reward = float(self._position) * bar_ret - cost
        if self._position == 0:
            reward -= self.hold_penalty

        self._step += 1
        terminated = self._step >= self.n_bars
        if terminated:
            self._step = self.n_bars - 1

        return self._obs(), reward, terminated, False, {
            "position": self._position,
            "bar_return": bar_ret,
        }

    def render(self) -> None:
        pass


# ── PPO wrapper ───────────────────────────────────────────────────────────────

class PPORLAgent(BaseModel):
    """
    Wraps stable-baselines3 PPO with the TradingEnv.

    Usage:
        agent = PPORLAgent(total_timesteps=500_000)
        agent.fit(features, log_returns)
        signals = agent.predict(features)
    """

    def __init__(
        self,
        total_timesteps: int = 500_000,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        ent_coef: float = 0.01,
        trade_cost: float = 0.001,
        seed: int = 42,
        verbose: int = 0,
    ) -> None:
        super().__init__("PPORLAgent")
        self.total_timesteps = total_timesteps
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.trade_cost = trade_cost
        self.seed = seed
        self.verbose = verbose
        self._ppo: PPO | None = None
        self._feature_cols: list[str] = []

    # ── Normalise features ────────────────────────────────────────────────────

    @staticmethod
    def _normalise(X: pd.DataFrame) -> np.ndarray:
        from sklearn.preprocessing import StandardScaler
        return StandardScaler().fit_transform(X.values.astype(np.float32))

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,           # y = log returns aligned to X
        **kwargs: Any,
    ) -> "PPORLAgent":
        """
        Train PPO on the TradingEnv built from X (features) and y (log returns).
        """
        self._feature_cols = list(X.columns)
        feat_norm = self._normalise(X)
        log_ret = y.values.astype(np.float32)

        def make_env():
            return TradingEnv(feat_norm, log_ret, trade_cost=self.trade_cost)

        env = DummyVecEnv([make_env])
        self._ppo = PPO(
            policy="MlpPolicy",
            env=env,
            n_steps=min(self.n_steps, len(X) - 1),
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            ent_coef=self.ent_coef,
            verbose=self.verbose,
            seed=self.seed,
        )
        self._ppo.learn(total_timesteps=self.total_timesteps)
        self.is_fitted = True
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Run greedy policy over X. Returns {-1, 0, +1} signal array.
        """
        self._check_fitted()
        feat_norm = self._normalise(X)
        # Build a temporary env just for inference
        dummy_ret = np.zeros(len(feat_norm), dtype=np.float32)
        env = TradingEnv(feat_norm, dummy_ret, trade_cost=0.0)
        obs, _ = env.reset()
        signals = np.zeros(len(X), dtype=int)
        action_map = {0: 0, 1: 1, 2: -1}

        for i in range(len(X)):
            action, _ = self._ppo.predict(obs, deterministic=True)
            signals[i] = action_map[int(action)]
            obs, _, terminated, _, _ = env.step(int(action))
            if terminated:
                break

        return signals

    def predict_signal(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.predict(X), index=X.index, name="signal")

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        self._check_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._ppo.save(str(path / "ppo_agent"))

    def load(self, path: Path) -> "PPORLAgent":
        path = Path(path)
        self._ppo = PPO.load(str(path / "ppo_agent"))
        self.is_fitted = True
        return self
