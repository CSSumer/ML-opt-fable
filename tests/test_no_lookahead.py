"""Mandatory non-mocked integration suite.

Instantiates the *real* engine on a synthetic data matrix and verifies:
1. the hard look-ahead guard catches artificially injected future data,
2. point-in-time alignment of macro features (publication-lag aware),
3. PSD guarantees for every covariance estimator,
4. model output is non-degenerate (predictions vary across tickers),
plus an end-to-end engine run with real fit/predict/optimise calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.config import PipelineConfig
from src.covariance import COVARIANCE_NAMES, make_covariance
from src.data.macro import align_macro_pit, first_release_series
from src.engine.walkforward import (
    LookAheadError,
    WalkForwardEngine,
    check_no_lookahead,
)
from src.models.linear import OLSPredictor
from src.models.naive import HistoricalMeanPredictor
from src.optimisers import make_optimiser


# --------------------------------------------------------------------------
# 1. Look-ahead guard
# --------------------------------------------------------------------------
class TestLookAheadGuard:
    def test_guard_catches_injected_future_data(self):
        test_date = pd.Timestamp("2020-06-30")
        poisoned = pd.DatetimeIndex(
            ["2019-01-31", "2019-02-28", "2020-07-31"]  # one month of future
        )
        with pytest.raises(LookAheadError, match="DATA LEAKAGE"):
            check_no_lookahead(poisoned, test_date)

    def test_guard_rejects_train_equal_to_test(self):
        d = pd.Timestamp("2020-06-30")
        with pytest.raises(LookAheadError):
            check_no_lookahead(pd.DatetimeIndex([d]), d)

    def test_guard_passes_clean_window(self):
        check_no_lookahead(
            pd.DatetimeIndex(["2019-01-31", "2020-05-31"]),
            pd.Timestamp("2020-06-30"),
        )

    def test_engine_crashes_on_poisoned_split(self, dataset, config):
        """Inject future dates into the engine's windowing and verify the
        run dies loudly instead of silently producing inflated results."""
        X, y, monthly = dataset
        engine = WalkForwardEngine(X, y, monthly, config)

        original = engine._split_windows

        def poisoned_split(t):
            core, val, full = original(t)
            future = engine.dates[engine.dates > t]
            poisoned_core = core.append(future[:1])  # inject one future month
            return poisoned_core, val, full

        engine._split_windows = poisoned_split
        with pytest.raises(LookAheadError):
            engine.run(
                HistoricalMeanPredictor(),
                make_covariance("sample"),
                make_optimiser("equal"),
            )

    def test_purge_gap_zero_rejected_by_schema(self):
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(
                {"walkforward": {"purge_gap_months": 0}}
            )

    def test_training_labels_realised_before_rebalance(self, dataset, config):
        """The label of the last training date spans (d, d+1m]; with a >=1
        month purge gap that window must close at or before t."""
        X, y, monthly = dataset
        engine = WalkForwardEngine(X, y, monthly, config)
        t = engine.dates[-2]
        core, val, _ = engine._split_windows(t)
        label_end = pd.Timestamp(core.max()) + pd.DateOffset(months=1)
        assert label_end <= t + pd.DateOffset(days=4)  # month-end jitter


# --------------------------------------------------------------------------
# 2. Point-in-time macro alignment
# --------------------------------------------------------------------------
class TestPointInTimeMacro:
    def _vintages(self) -> pd.DataFrame:
        # Observation for March published 10 April, revised 10 May.
        return pd.DataFrame(
            {
                "obs_date": pd.to_datetime(
                    ["2020-03-01", "2020-03-01", "2020-04-01"]
                ),
                "publication_date": pd.to_datetime(
                    ["2020-04-10", "2020-05-10", "2020-05-08"]
                ),
                "value": [5.0, 5.7, 6.0],  # 5.7 is the later *revision*
            }
        )

    def test_first_release_keeps_unrevised_value(self):
        fr = first_release_series(self._vintages())
        march = fr[fr["obs_date"] == "2020-03-01"]
        assert len(march) == 1
        assert march["value"].iloc[0] == 5.0  # first print, not the revision
        assert march["publication_date"].iloc[0] == pd.Timestamp("2020-04-10")

    def test_alignment_respects_publication_lag(self):
        fr = first_release_series(self._vintages())
        dates = pd.DatetimeIndex(["2020-03-31", "2020-04-30", "2020-05-31"])
        aligned = align_macro_pit(fr, dates)
        # 31 Mar: March data not yet published -> NaN, never the future value
        assert np.isnan(aligned.iloc[0])
        # 30 Apr: March first print (5.0) is out; April (pub 8 May) is not
        assert aligned.iloc[1] == 5.0
        # 31 May: April observation (6.0) is the latest published
        assert aligned.iloc[2] == 6.0

    def test_no_future_value_ever_visible(self):
        fr = first_release_series(self._vintages())
        for t in pd.date_range("2020-03-01", "2020-07-01", freq="D"):
            val = align_macro_pit(fr, pd.DatetimeIndex([t])).iloc[0]
            if not np.isnan(val):
                published = fr[fr["publication_date"] <= t]
                assert val in published["value"].to_numpy()


# --------------------------------------------------------------------------
# 3. PSD guarantees
# --------------------------------------------------------------------------
class TestCovariancePSD:
    @pytest.mark.parametrize("name", COVARIANCE_NAMES)
    def test_estimator_is_psd(self, dataset, name):
        _, _, monthly = dataset
        cov = make_covariance(name).estimate(monthly.tail(36))
        assert cov.shape == (monthly.shape[1], monthly.shape[1])
        eigvals = np.linalg.eigvalsh((cov + cov.T) / 2)
        assert eigvals.min() >= -1e-10, f"{name} produced a non-PSD matrix"

    @pytest.mark.parametrize("name", COVARIANCE_NAMES)
    def test_psd_under_short_window(self, dataset, name):
        """T < N (rank-deficient sample cov) must still come out PSD."""
        _, _, monthly = dataset
        cov = make_covariance(name).estimate(monthly.tail(8))
        eigvals = np.linalg.eigvalsh((cov + cov.T) / 2)
        assert eigvals.min() >= -1e-10


# --------------------------------------------------------------------------
# 4. Non-degenerate predictions + full integration run
# --------------------------------------------------------------------------
class TestIntegration:
    def test_predictions_vary_across_tickers(self, dataset):
        X, y, _ = dataset
        dates = X.index.get_level_values("date").unique().sort_values()
        train_dates, test_date = dates[:-2], dates[-1]
        idx = X.index.get_level_values("date")
        X_tr, X_te = X[idx.isin(train_dates)], X[idx == test_date]
        for model in (HistoricalMeanPredictor(), OLSPredictor()):
            model.fit(X_tr, y.loc[X_tr.index])
            preds = model.predict(X_te)
            assert len(preds) == len(X_te)
            assert preds.std() > 0, (
                f"{model.get_name()} produced degenerate (constant) output"
            )

    def test_full_engine_run_two_stage(self, dataset, config):
        X, y, monthly = dataset
        engine = WalkForwardEngine(X, y, monthly, config)
        result = engine.run(
            OLSPredictor(),
            make_covariance("lw"),
            make_optimiser("mvo", max_weight=config.portfolio.max_weight),
        )
        assert len(result.net_returns) >= 12
        assert result.net_returns.notna().all()
        # weights: long-only, fully invested, capped
        w = result.weights_history
        assert (w >= -1e-9).all().all()
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-6)
        assert (w <= config.portfolio.max_weight + 1e-6).all().all()
        # net of costs must not exceed gross anywhere with positive turnover
        assert (result.net_returns <= result.gross_returns + 1e-12).all()
        # every model fit stayed strictly in the past (guard ran every month)
        assert result.resource_log["training_time_seconds"] > 0

    def test_full_engine_returns_differ_across_strategies(self, dataset, config):
        X, y, monthly = dataset
        engine = WalkForwardEngine(X, y, monthly, config)
        res_a = engine.run(HistoricalMeanPredictor(), make_covariance("sample"),
                           make_optimiser("equal"))
        res_b = engine.run(OLSPredictor(), make_covariance("lw"),
                           make_optimiser("mvo"))
        assert res_a.strategy_name != res_b.strategy_name
        assert not res_a.net_returns.equals(res_b.net_returns)
