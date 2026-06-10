"""ML model tests (skipped automatically when torch/xgboost are absent)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import EndToEndConfig, LSTMConfig, XGBoostConfig


def _split(X, y, n_test_dates=1):
    dates = X.index.get_level_values("date").unique().sort_values()
    train_dates, test_date = dates[:-1 - n_test_dates], dates[-1]
    idx = X.index.get_level_values("date")
    X_tr = X[idx.isin(train_dates)]
    return X_tr, y.loc[X_tr.index], X[idx == test_date]


class TestXGBoost:
    def test_fit_predict_nondegenerate(self, dataset):
        pytest.importorskip("xgboost")
        from src.models.xgb import XGBoostPredictor

        X, y, _ = dataset
        X_tr, y_tr, X_te = _split(X, y)
        model = XGBoostPredictor(XGBoostConfig(n_estimators=30), seed=0)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert len(preds) == len(X_te)
        assert preds.std() > 0

    def test_deterministic_given_seed(self, dataset):
        pytest.importorskip("xgboost")
        from src.models.xgb import XGBoostPredictor

        X, y, _ = dataset
        X_tr, y_tr, X_te = _split(X, y)
        outs = []
        for _ in range(2):
            m = XGBoostPredictor(XGBoostConfig(n_estimators=20), seed=0)
            m.fit(X_tr, y_tr)
            outs.append(m.predict(X_te).to_numpy())
        np.testing.assert_allclose(outs[0], outs[1])


class TestLSTM:
    def test_inference_uses_real_history_not_tiling(self, dataset):
        """The prediction must depend on past months, not just the current
        row — i.e. the inference sequence is genuine history, not the current
        month tiled sequence_length times."""
        pytest.importorskip("torch")
        from src.models.lstm import LSTMPredictor

        X, y, _ = dataset
        X_tr, y_tr, X_te = _split(X, y)
        cfg = LSTMConfig(sequence_length=6, epochs=3, hidden_size=8)
        model = LSTMPredictor(cfg, seed=0)
        model.fit(X_tr, y_tr)
        base = model.predict(X_te)
        assert base.std() > 0  # non-degenerate across tickers

        # Perturb one ticker's cached *history* (not the current features):
        # the prediction for that ticker must change.
        ticker = str(X_te.index.get_level_values("ticker")[0])
        model._history[ticker] = model._history[ticker] + 5.0
        perturbed = model.predict(X_te)
        assert abs(perturbed.loc[ticker] - base.loc[ticker]) > 1e-8, (
            "LSTM ignored cached history — it is tiling the current month"
        )

    def test_scaler_fit_on_train_only(self, dataset):
        pytest.importorskip("torch")
        from src.models.lstm import LSTMPredictor

        X, y, _ = dataset
        X_tr, y_tr, X_te = _split(X, y)
        model = LSTMPredictor(LSTMConfig(sequence_length=4, epochs=1,
                                         hidden_size=4), seed=0)
        model.fit(X_tr, y_tr)
        frozen_mean = model._scaler.mean_.copy()
        model.predict(X_te)  # must not refit
        np.testing.assert_array_equal(frozen_mean, model._scaler.mean_)


class TestEndToEnd:
    def test_weights_valid_and_nondegenerate(self, dataset):
        pytest.importorskip("torch")
        from src.models.end_to_end import EndToEndAllocator

        X, y, _ = dataset
        X_tr, y_tr, X_te = _split(X, y)
        model = EndToEndAllocator(EndToEndConfig(epochs=5, hidden_size=16),
                                  seed=0)
        model.fit(X_tr, y_tr)
        w = model.predict_weights(X_te)
        assert len(w) == len(X_te)
        assert abs(w.sum() - 1.0) < 1e-5
        assert (w >= 0).all()
        assert w.std() > 0
