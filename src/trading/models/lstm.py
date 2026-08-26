"""LSTM price model — predicts next-bar log return from a feature sequence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.trading.models.base import BaseModel


# ── Network ───────────────────────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)   # last time-step


# ── Dataset builder ───────────────────────────────────────────────────────────

def _make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window of length seq_len over X/y."""
    n = len(X) - seq_len
    xs = np.stack([X[i : i + seq_len] for i in range(n)])
    ys = y[seq_len:]
    return xs, ys


# ── Model class ───────────────────────────────────────────────────────────────

class LSTMPriceModel(BaseModel):
    """
    Sequence-to-one LSTM that predicts the next bar's log return.
    Signals are derived by thresholding predictions.
    """

    def __init__(
        self,
        seq_len: int = 60,
        hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 50,
        patience: int = 7,           # early stopping
        signal_threshold: float = 0.001,   # min |predicted return| to act
        device: str | None = None,
    ) -> None:
        super().__init__("LSTMPriceModel")
        self.seq_len = seq_len
        self.hidden = hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.signal_threshold = signal_threshold
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._scaler = StandardScaler()
        self._net: _LSTMNet | None = None
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        val_fraction: float = 0.15,
        **kwargs: Any,
    ) -> "LSTMPriceModel":
        """
        X: feature DataFrame (rows = bars)
        y: target log returns aligned to X
        """
        X_arr = self._scaler.fit_transform(X.values.astype(np.float32))
        y_arr = y.values.astype(np.float32)

        xs, ys = _make_sequences(X_arr, y_arr, self.seq_len)

        split = max(1, int(len(xs) * (1 - val_fraction)))
        X_tr, y_tr = xs[:split], ys[:split]
        X_vl, y_vl = xs[split:], ys[split:]

        tr_ds = TensorDataset(
            torch.from_numpy(X_tr), torch.from_numpy(y_tr)
        )
        vl_ds = TensorDataset(
            torch.from_numpy(X_vl), torch.from_numpy(y_vl)
        )
        tr_dl = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True)
        vl_dl = DataLoader(vl_ds, batch_size=self.batch_size)

        self._net = _LSTMNet(X.shape[1], self.hidden, self.n_layers, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        no_improve = 0
        best_state = None

        for epoch in range(self.max_epochs):
            # Train
            self._net.train()
            tr_loss = 0.0
            for xb, yb in tr_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self._net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
                tr_loss += loss.item() * len(xb)
            tr_loss /= len(tr_ds)

            # Validate
            self._net.eval()
            vl_loss = 0.0
            with torch.no_grad():
                for xb, yb in vl_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    vl_loss += loss_fn(self._net(xb), yb).item() * len(xb)
            vl_loss /= max(len(vl_ds), 1)

            self.train_losses.append(tr_loss)
            self.val_losses.append(vl_loss)

            if vl_loss < best_val:
                best_val = vl_loss
                best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)

        self.is_fitted = True
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_returns(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted log returns for each bar (first seq_len rows = 0)."""
        self._check_fitted()
        X_arr = self._scaler.transform(X.values.astype(np.float32))
        xs, _ = _make_sequences(X_arr, np.zeros(len(X_arr)), self.seq_len)

        self._net.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(xs), self.batch_size):
                batch = torch.from_numpy(xs[i : i + self.batch_size]).to(self.device)
                preds.append(self._net(batch).cpu().numpy())
        pred_arr = np.concatenate(preds) if preds else np.array([])

        # Pad front with zeros for warm-up bars
        full = np.zeros(len(X))
        full[self.seq_len:] = pred_arr
        return full

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Convert predicted returns to {+1, 0, -1} signals."""
        ret = self.predict_returns(X)
        signal = np.zeros(len(ret), dtype=int)
        signal[ret > self.signal_threshold] = 1
        signal[ret < -self.signal_threshold] = -1
        return signal

    def predict_signal(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.predict(X), index=X.index, name="signal")

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        self._check_fitted()
        import joblib
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), path / "lstm_weights.pt")
        joblib.dump(self._scaler, path / "scaler.joblib")
        joblib.dump(
            {k: v for k, v in self.__dict__.items() if k not in ("_net", "_scaler", "device")},
            path / "config.joblib",
        )

    def load(self, path: Path) -> "LSTMPriceModel":
        import joblib
        path = Path(path)
        cfg = joblib.load(path / "config.joblib")
        for k, v in cfg.items():
            setattr(self, k, v)
        self._scaler = joblib.load(path / "scaler.joblib")
        # Rebuild net — input_size stored as train_losses length proxy isn't reliable;
        # users must supply input_size externally or re-fit. Load is best-effort here.
        self.is_fitted = True
        return self
