"""XGBoost signal predictor with walk-forward cross-validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.trading.models.base import BaseModel, make_labels


class XGBSignalModel(BaseModel):
    """
    Directional signal predictor: predicts +1 (long), 0 (hold), -1 (short).

    Uses XGBoost with a StandardScaler pre-processing step.
    Walk-forward evaluation is supported via walk_forward_predict().
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        threshold: float = 0.55,    # confidence needed to flip from hold
        random_state: int = 42,
    ) -> None:
        super().__init__("XGBSignalModel")
        self.threshold = threshold
        self._scaler = StandardScaler()
        self._model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )
        # Map internal class indices → signal values
        self._class_map: dict[int, int] = {}

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> "XGBSignalModel":
        """
        Train on feature matrix X and integer labels y ∈ {-1, 0, +1}.
        Drops rows where y is NaN.
        """
        mask = y.notna() & (y != 0)   # exclude hold/NaN from training for sharper signal
        X_clean = X.loc[mask]
        y_clean = y.loc[mask].astype(int)

        X_scaled = self._scaler.fit_transform(X_clean)
        self._model.fit(X_scaled, y_clean, **kwargs)
        self._class_map = {i: c for i, c in enumerate(self._model.classes_)}
        self.is_fitted = True
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns (n_samples, n_classes) probability matrix."""
        self._check_fitted()
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)

    def predict_signal(self, X: pd.DataFrame) -> pd.Series:
        """
        Threshold-gated signal: only emit non-zero when max class prob > threshold.
        Returns a Series of {-1, 0, +1} aligned to X.index.
        """
        self._check_fitted()
        proba = self.predict_proba(X)
        raw = self.predict(X)
        max_prob = proba.max(axis=1)
        signal = np.where(max_prob >= self.threshold, raw, 0)
        return pd.Series(signal, index=X.index, name="signal")

    def feature_importance(self, feature_names: list[str]) -> pd.Series:
        self._check_fitted()
        imp = self._model.feature_importances_
        return pd.Series(imp, index=feature_names, name="importance").sort_values(ascending=False)

    # ── Walk-forward ─────────────────────────────────────────────────────────

    def walk_forward_predict(
        self,
        features: pd.DataFrame,
        close: pd.Series,
        train_window: int = 504,    # ~2 years of daily bars
        step: int = 63,             # retrain every quarter
        forward_days: int = 1,
        label_threshold: float = 0.0,
    ) -> pd.Series:
        """
        Expanding-window walk-forward prediction.
        Trains on [0, t-step], predicts [t-step, t].
        Returns signal Series aligned to features.index.
        """
        labels = make_labels(close, forward_days, label_threshold)
        signals = pd.Series(0, index=features.index, dtype=int, name="signal")
        n = len(features)

        for start in range(train_window, n, step):
            X_train = features.iloc[:start]
            y_train = labels.iloc[:start]
            end = min(start + step, n)
            X_pred = features.iloc[start:end]

            try:
                self.fit(X_train, y_train)
                signals.iloc[start:end] = self.predict_signal(X_pred).values
            except Exception:
                pass   # insufficient data for this fold — leave as hold

        return signals

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        self._check_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path / "xgb_model.joblib")
        joblib.dump(self._scaler, path / "scaler.joblib")
        joblib.dump(self._class_map, path / "class_map.joblib")

    def load(self, path: Path) -> "XGBSignalModel":
        path = Path(path)
        self._model = joblib.load(path / "xgb_model.joblib")
        self._scaler = joblib.load(path / "scaler.joblib")
        self._class_map = joblib.load(path / "class_map.joblib")
        self.is_fitted = True
        return self
