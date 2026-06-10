"""LSTM return predictor with genuine chronological inference sequences.

Correctness notes
-----------------
* ``fit()`` caches each ticker's chronological feature rows. ``predict()``
  assembles a real ``sequence_length``-month window ending at the test row by
  concatenating cached history with the current features. The current month
  is NEVER tiled into a fake constant sequence — that would silently turn the
  LSTM into a one-step MLP.
* The feature scaler is fit on the training core only; validation and test
  rows are transformed with frozen parameters (strict scaler isolation).
* torch is imported lazily inside methods (macOS libomp isolation, see
  src/engine/runner.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import LSTMConfig


class LSTMPredictor:
    def __init__(self, cfg: LSTMConfig | None = None, seed: int = 42) -> None:
        self.cfg = cfg or LSTMConfig()
        self.seed = seed
        self._scaler = StandardScaler()
        self._columns: list[str] | None = None
        self._net = None
        # ticker -> chronological (dates-sorted) scaled feature array
        self._history: dict[str, np.ndarray] = {}
        self._history_dates: dict[str, np.ndarray] = {}

    # ---------------------------------------------------------------- helpers
    def _build_net(self, n_features: int):
        import torch.nn as nn

        cfg = self.cfg

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    n_features,
                    cfg.hidden_size,
                    num_layers=cfg.num_layers,
                    batch_first=True,
                    dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
                )
                self.head = nn.Linear(cfg.hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        return _Net()

    def _make_sequences(
        self, X: pd.DataFrame, y: pd.Series | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Per-ticker sliding windows of length sequence_length (train side)."""
        seq_len = self.cfg.sequence_length
        xs, ys = [], []
        for ticker, grp in X.groupby(level="ticker", sort=True):
            grp = grp.sort_index(level="date")
            arr = self._scaler.transform(grp.to_numpy(dtype=float))
            tgt = y.loc[grp.index].to_numpy(dtype=float) if y is not None else None
            for i in range(seq_len - 1, len(arr)):
                xs.append(arr[i - seq_len + 1 : i + 1])
                if tgt is not None:
                    ys.append(tgt[i])
        if not xs:
            return np.empty((0, seq_len, X.shape[1])), None
        return np.stack(xs), (np.asarray(ys) if y is not None else None)

    # -------------------------------------------------------------------- fit
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        import torch

        torch.manual_seed(self.seed)
        self._columns = list(X_train.columns)

        # Scaler: training core only. Validation/test are transformed with
        # these frozen parameters — never refit.
        self._scaler.fit(X_train.to_numpy(dtype=float))

        # Cache chronological history per ticker for inference-time sequences.
        self._history, self._history_dates = {}, {}
        for ticker, grp in X_train.groupby(level="ticker", sort=True):
            grp = grp.sort_index(level="date")
            self._history[str(ticker)] = self._scaler.transform(
                grp.to_numpy(dtype=float)
            )
            self._history_dates[str(ticker)] = (
                grp.index.get_level_values("date").to_numpy()
            )

        Xseq, yseq = self._make_sequences(X_train, y_train)
        if len(Xseq) == 0:
            raise RuntimeError(
                f"Training window too short for sequence_length="
                f"{self.cfg.sequence_length}."
            )
        Xv, yv = (None, None)
        if X_val is not None and len(X_val) > 0:
            Xv, yv = self._make_sequences(X_val[self._columns], y_val)

        self._net = self._build_net(len(self._columns))
        opt = torch.optim.Adam(self._net.parameters(), lr=self.cfg.learning_rate)
        loss_fn = torch.nn.MSELoss()
        Xt = torch.tensor(Xseq, dtype=torch.float32)
        yt = torch.tensor(yseq, dtype=torch.float32)
        val_t = (
            (torch.tensor(Xv, dtype=torch.float32),
             torch.tensor(yv, dtype=torch.float32))
            if Xv is not None and len(Xv) > 0
            else None
        )

        best_val, best_state, patience = float("inf"), None, 0
        n = len(Xt)
        gen = torch.Generator().manual_seed(self.seed)
        for epoch in range(self.cfg.epochs):
            self._net.train()
            perm = torch.randperm(n, generator=gen)
            for start in range(0, n, self.cfg.batch_size):
                idx = perm[start : start + self.cfg.batch_size]
                opt.zero_grad()
                loss = loss_fn(self._net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
            if val_t is not None:
                self._net.eval()
                with torch.no_grad():
                    vloss = float(loss_fn(self._net(val_t[0]), val_t[1]))
                if vloss < best_val - 1e-7:
                    best_val, patience = vloss, 0
                    best_state = {
                        k: v.clone() for k, v in self._net.state_dict().items()
                    }
                else:
                    patience += 1
                    if patience >= self.cfg.early_stopping_patience:
                        break
        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._net.eval()

    # ---------------------------------------------------------------- predict
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        import torch

        if self._net is None:
            raise RuntimeError("fit() must be called before predict().")
        seq_len = self.cfg.sequence_length
        Xs = X_test[self._columns]

        seqs, tickers = [], []
        for (date, ticker), row in zip(Xs.index, Xs.to_numpy(dtype=float)):
            scaled_row = self._scaler.transform(row.reshape(1, -1))
            hist = self._history.get(str(ticker))
            if hist is not None and len(hist) > 0:
                # genuine chronological window: last (seq_len-1) cached months
                # strictly before the test date, then the current row.
                dates = self._history_dates[str(ticker)]
                cutoff = (dates < np.datetime64(pd.Timestamp(date))).sum()
                tail = hist[max(0, cutoff - (seq_len - 1)) : cutoff]
                window = np.vstack([tail, scaled_row])
            else:
                window = scaled_row
            if len(window) < seq_len:  # young ticker: left-pad with first row
                pad = np.repeat(window[:1], seq_len - len(window), axis=0)
                window = np.vstack([pad, window])
            seqs.append(window[-seq_len:])
            tickers.append(ticker)

        with torch.no_grad():
            preds = self._net(
                torch.tensor(np.stack(seqs), dtype=torch.float32)
            ).numpy()
        return pd.Series(preds.astype(float), index=pd.Index(tickers, name="ticker"))

    def get_name(self) -> str:
        return "lstm"
