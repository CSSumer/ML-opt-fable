"""Abstract component interfaces.

Three small protocols let predictors, covariance estimators and optimisers mix
freely. Components are duck-typed (``runtime_checkable`` Protocols) so adding
a new model never requires touching the engine.

Data contract: feature matrices carry a ``(date, ticker)`` MultiIndex;
predictions and weights are ``pd.Series`` indexed by ticker.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class ReturnPredictor(Protocol):
    """Predicts next-period cross-sectional returns."""

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train on features/labels with a (date, ticker) MultiIndex."""
        ...

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Return expected returns indexed by ticker for one rebalance date."""
        ...

    def get_name(self) -> str:
        ...


@runtime_checkable
class WeightPredictor(Protocol):
    """End-to-end models that output portfolio weights directly (RQ4).

    The engine detects this protocol and bypasses the covariance/optimiser
    stages entirely.
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        ...

    def predict_weights(self, X_test: pd.DataFrame) -> pd.Series:
        """Return portfolio weights indexed by ticker (sum to 1, long-only)."""
        ...

    def get_name(self) -> str:
        ...


@runtime_checkable
class PortfolioOptimiser(Protocol):
    def optimise(
        self,
        expected_returns: pd.Series,
        cov_matrix: np.ndarray,
        tickers: list[str],
    ) -> pd.Series:
        """Return portfolio weights indexed by ticker."""
        ...

    def get_name(self) -> str:
        ...


@runtime_checkable
class CovarianceEstimator(Protocol):
    def estimate(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate an (N x N) covariance matrix from a returns panel."""
        ...

    def get_name(self) -> str:
        ...
