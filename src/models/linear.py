"""Traditional econometric baselines: OLS and a Fama-French 3-factor model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class OLSPredictor:
    """Pooled OLS of forward returns on all features.

    The scaler is fit on training data only and reused at predict time —
    strict scaler isolation, no refit on test distributions.
    """

    def __init__(self) -> None:
        self._model = LinearRegression()
        self._scaler = StandardScaler()
        self._columns: list[str] | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self._columns = list(X_train.columns)
        Xs = self._scaler.fit_transform(X_train.to_numpy(dtype=float))
        self._model.fit(Xs, y_train.to_numpy(dtype=float))

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        Xs = self._scaler.transform(X_test[self._columns].to_numpy(dtype=float))
        preds = self._model.predict(Xs)
        tickers = X_test.index.get_level_values("ticker")
        return pd.Series(preds, index=tickers)

    def get_name(self) -> str:
        return "ols"


class FamaFrench3Predictor:
    """Fama-French 3-factor style cross-sectional model.

    Without CRSP/Compustat we proxy the three factors from the feature set
    itself, per date: market (cross-sectional mean 1m momentum), size proxy
    (long-window vol, low vol ~ large cap) and value/reversal proxy (negative
    12m momentum). Factor loadings are estimated per ticker on the training
    window via time-series OLS; expected returns combine estimated loadings
    with expanding factor-premium means. This is a transparent, documented
    proxy — the thesis text should flag it as such.
    """

    MARKET_COL = "mom_1m"
    SIZE_COL_CANDIDATES = ("vol_6m", "vol_3m")
    VALUE_COL = "mom_12m"

    def __init__(self) -> None:
        self._betas: pd.DataFrame | None = None
        self._premia: pd.Series | None = None
        self._fallback_mean: float = 0.0

    def _factor_returns(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        df = X.copy()
        df["_y"] = y
        size_col = next(
            (c for c in self.SIZE_COL_CANDIDATES if c in X.columns), None
        )
        rows = {}
        for date, grp in df.groupby(level="date"):
            yv = grp["_y"]
            mkt = yv.mean()
            smb = (
                yv[grp[size_col] >= grp[size_col].median()].mean()
                - yv[grp[size_col] < grp[size_col].median()].mean()
                if size_col is not None
                else 0.0
            )
            val = (
                yv[grp[self.VALUE_COL] <= grp[self.VALUE_COL].median()].mean()
                - yv[grp[self.VALUE_COL] > grp[self.VALUE_COL].median()].mean()
                if self.VALUE_COL in grp.columns
                else 0.0
            )
            rows[date] = (mkt, smb, val)
        return pd.DataFrame.from_dict(
            rows, orient="index", columns=["MKT", "SMB", "HML"]
        ).sort_index()

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        factors = self._factor_returns(X_train, y_train).fillna(0.0)
        self._premia = factors.mean()
        self._fallback_mean = float(y_train.mean())

        F = np.column_stack([np.ones(len(factors)), factors.to_numpy()])
        betas = {}
        y_wide = y_train.unstack("ticker").reindex(factors.index)
        for ticker in y_wide.columns:
            yv = y_wide[ticker].to_numpy(dtype=float)
            ok = ~np.isnan(yv)
            if ok.sum() < 12:  # too little history for a stable loading
                continue
            coef, *_ = np.linalg.lstsq(F[ok], yv[ok], rcond=None)
            betas[ticker] = coef[1:]  # drop intercept
        self._betas = pd.DataFrame(betas, index=["MKT", "SMB", "HML"]).T

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        if self._betas is None or self._premia is None:
            raise RuntimeError("fit() must be called before predict().")
        tickers = X_test.index.get_level_values("ticker")
        exp = (self._betas @ self._premia).reindex(tickers)
        return exp.fillna(self._fallback_mean)

    def get_name(self) -> str:
        return "ff3"
