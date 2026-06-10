"""XGBoost return predictor.

xgboost is imported lazily inside methods: on macOS, loading xgboost and
torch into the same process triggers libomp segfaults, so combinations run in
isolated subprocesses and this module must not import xgboost at module load.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import XGBoostConfig


class XGBoostPredictor:
    def __init__(self, cfg: XGBoostConfig | None = None, seed: int = 42) -> None:
        self.cfg = cfg or XGBoostConfig()
        self.seed = seed
        self._model = None
        self._columns: list[str] | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        import xgboost as xgb  # lazy (libomp isolation)

        self._columns = list(X_train.columns)
        params = dict(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            learning_rate=self.cfg.learning_rate,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            random_state=self.seed,
            n_jobs=1,  # deterministic; parallelism lives at the combo level
            objective="reg:squarederror",
        )
        eval_set = None
        if X_val is not None and len(X_val) > 0:
            params["early_stopping_rounds"] = self.cfg.early_stopping_rounds
            eval_set = [(X_val[self._columns].to_numpy(dtype=float),
                         y_val.to_numpy(dtype=float))]
        self._model = xgb.XGBRegressor(**params)
        self._model.fit(
            X_train.to_numpy(dtype=float),
            y_train.to_numpy(dtype=float),
            eval_set=eval_set,
            verbose=False,
        )

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("fit() must be called before predict().")
        preds = self._model.predict(X_test[self._columns].to_numpy(dtype=float))
        tickers = X_test.index.get_level_values("ticker")
        return pd.Series(np.asarray(preds, dtype=float), index=tickers)

    def get_name(self) -> str:
        return "xgboost"
