"""Abstract base class for all ML models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Common interface for XGBoost, LSTM, and RL models.
    All models accept a feature DataFrame (rows = bars, cols = features)
    and return a signal array aligned to the same index.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> "BaseModel":
        """Train the model. Returns self for chaining."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return a signal array aligned to X's index.
        Convention: +1 = long, 0 = hold, -1 = short.
        """
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw probability / score (optional; default raises)."""
        raise NotImplementedError(f"{self.name} does not support predict_proba")

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model weights to disk."""
        ...

    @abstractmethod
    def load(self, path: Path) -> "BaseModel":
        """Load model weights from disk. Returns self."""
        ...

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} has not been fitted yet.")


def make_labels(
    close: pd.Series,
    forward_days: int = 1,
    threshold: float = 0.0,
) -> pd.Series:
    """
    Create directional labels from future returns.
    +1 if forward return > threshold, -1 if < -threshold, 0 otherwise.
    The last `forward_days` rows are NaN (no future available).
    """
    fwd_ret = close.shift(-forward_days) / close - 1
    labels = pd.Series(0, index=close.index, dtype=int)
    labels[fwd_ret > threshold] = 1
    labels[fwd_ret < -threshold] = -1
    labels[fwd_ret.isna()] = 0
    return labels
