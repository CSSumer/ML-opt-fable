"""Naive baselines (RQ1 control group)."""

from __future__ import annotations

import pandas as pd


class ZeroPredictor:
    """Predicts zero expected return for every ticker.

    With a mean-variance optimiser this collapses to minimum-variance; it
    isolates how much value the *return forecast* adds at all.
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        pass

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        tickers = X_test.index.get_level_values("ticker")
        return pd.Series(0.0, index=tickers)

    def get_name(self) -> str:
        return "zero"


class HistoricalMeanPredictor:
    """Predicts each ticker's expanding historical mean return."""

    def __init__(self) -> None:
        self._means: pd.Series | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self._means = y_train.groupby(level="ticker").mean()

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        if self._means is None:
            raise RuntimeError("fit() must be called before predict().")
        tickers = X_test.index.get_level_values("ticker")
        return self._means.reindex(tickers).fillna(self._means.mean())

    def get_name(self) -> str:
        return "histmean"
