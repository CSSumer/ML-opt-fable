"""End-to-end allocator (RQ4): a network that outputs portfolio weights.

Instead of predict-then-optimise, a small MLP scores each ticker from its
features; a date-wise softmax converts scores to long-only weights summing to
one, and the training loss is the *negative Sharpe ratio* of the resulting
portfolio return series over the training dates — a differentiable portfolio
objective, so gradients flow from the financial criterion straight into the
feature weights.

Implements the ``WeightPredictor`` protocol: the engine detects
``predict_weights`` and bypasses the covariance/optimiser stages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import EndToEndConfig


class EndToEndAllocator:
    def __init__(self, cfg: EndToEndConfig | None = None, seed: int = 42) -> None:
        self.cfg = cfg or EndToEndConfig()
        self.seed = seed
        self._scaler = StandardScaler()
        self._columns: list[str] | None = None
        self._net = None

    def _build_net(self, n_features: int):
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(n_features, self.cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_size, 1),
        )

    @staticmethod
    def _negative_sharpe(port_returns):
        import torch

        mean = port_returns.mean()
        std = port_returns.std() + 1e-8
        return -(mean / std) * torch.sqrt(torch.tensor(12.0))

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
        self._scaler.fit(X_train.to_numpy(dtype=float))  # train-only scaler

        def to_batches(X: pd.DataFrame, y: pd.Series):
            batches = []
            for date, grp in X.groupby(level="date", sort=True):
                feats = self._scaler.transform(grp.to_numpy(dtype=float))
                rets = y.loc[grp.index].to_numpy(dtype=float)
                batches.append(
                    (torch.tensor(feats, dtype=torch.float32),
                     torch.tensor(rets, dtype=torch.float32))
                )
            return batches

        train_batches = to_batches(X_train, y_train)
        val_batches = (
            to_batches(X_val[self._columns], y_val)
            if X_val is not None and len(X_val) > 0
            else None
        )

        self._net = self._build_net(len(self._columns))
        opt = torch.optim.Adam(self._net.parameters(), lr=self.cfg.learning_rate)

        def portfolio_returns(batches):
            rets = []
            for feats, realised in batches:
                w = torch.softmax(self._net(feats).squeeze(-1), dim=0)
                rets.append((w * realised).sum())
            return torch.stack(rets)

        best_val, best_state, patience = float("inf"), None, 0
        for epoch in range(self.cfg.epochs):
            self._net.train()
            opt.zero_grad()
            loss = self._negative_sharpe(portfolio_returns(train_batches))
            loss.backward()
            opt.step()
            if val_batches:
                self._net.eval()
                with torch.no_grad():
                    vloss = float(
                        self._negative_sharpe(portfolio_returns(val_batches))
                    )
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

    def predict_weights(self, X_test: pd.DataFrame) -> pd.Series:
        import torch

        if self._net is None:
            raise RuntimeError("fit() must be called before predict_weights().")
        feats = self._scaler.transform(
            X_test[self._columns].to_numpy(dtype=float)
        )
        with torch.no_grad():
            w = torch.softmax(
                self._net(torch.tensor(feats, dtype=torch.float32)).squeeze(-1),
                dim=0,
            ).numpy()
        tickers = X_test.index.get_level_values("ticker")
        return pd.Series(np.asarray(w, dtype=float), index=tickers)

    def get_name(self) -> str:
        return "e2e"
