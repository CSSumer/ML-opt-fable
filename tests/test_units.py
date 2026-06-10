"""Unit tests: projection, metrics, statistics, config, runner ordering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig, load_config
from src.engine.metrics import (
    annualized_return,
    apply_transaction_costs,
    max_drawdown,
    sharpe_ratio,
    turnover_series,
)
from src.engine.runner import Combination, experiment_combinations
from src.engine.stats import (
    block_bootstrap_sharpe_ci,
    holm_bonferroni,
    jobson_korkie_memmel,
)
from src.optimisers import project_weights


class TestProjection:
    def test_caps_and_sums_to_one(self):
        w = project_weights(np.array([0.9, 0.05, 0.03, 0.02]), max_weight=0.4)
        assert w.max() <= 0.4 + 1e-9
        assert abs(w.sum() - 1.0) < 1e-9
        assert (w >= 0).all()

    def test_negative_weights_clipped(self):
        w = project_weights(np.array([-0.5, 0.8, 0.7]), max_weight=0.6)
        assert (w >= 0).all() and abs(w.sum() - 1.0) < 1e-9

    def test_infeasible_falls_back_to_equal_weight(self):
        # 3 assets * 0.2 cap = 0.6 < 1 -> mathematically infeasible
        w = project_weights(np.array([1.0, 0.0, 0.0]), max_weight=0.2)
        np.testing.assert_allclose(w, np.full(3, 1 / 3))

    def test_all_zero_input_becomes_equal_weight(self):
        w = project_weights(np.zeros(5), max_weight=0.5)
        np.testing.assert_allclose(w, np.full(5, 0.2))

    def test_water_filling_cascade(self):
        # capping the giant pushes mass into the next names, which then cap too
        w = project_weights(np.array([0.97, 0.01, 0.01, 0.01]), max_weight=0.3)
        assert w.max() <= 0.3 + 1e-9
        assert abs(w.sum() - 1.0) < 1e-9


class TestMetrics:
    def test_annualisation_uses_calendar_time_not_row_count(self):
        # 1% monthly over 24 calendar months, but with rows missing: the
        # elapsed-time annualisation must still see ~2 years, not 18 rows.
        full_idx = pd.date_range("2020-01-31", periods=24, freq="ME")
        sparse_idx = full_idx.delete([3, 7, 11, 15, 19, 23][:6])
        rets = pd.Series(0.01, index=sparse_idx)
        ar = annualized_return(rets)
        naive_ar = (1.01 ** 12) - 1  # what a row-count annualisation would say
        assert ar < naive_ar * 0.9  # materially lower than the naive figure

    def test_sharpe_uses_nonzero_risk_free(self):
        rets = pd.Series(0.005, index=pd.date_range("2020-01-31", periods=36,
                                                    freq="ME"))
        rets = rets + np.random.default_rng(0).normal(0, 0.002, 36)
        assert sharpe_ratio(rets, 0.0) > sharpe_ratio(rets, 0.04)

    def test_max_drawdown_sign_and_magnitude(self):
        rets = pd.Series([0.10, -0.50, 0.10],
                         index=pd.date_range("2020-01-31", periods=3, freq="ME"))
        assert max_drawdown(rets) == pytest.approx(-0.5)

    def test_turnover_and_costs(self):
        w = pd.DataFrame(
            [[0.5, 0.5], [1.0, 0.0]],
            index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
            columns=["A", "B"],
        )
        to = turnover_series(w)
        assert to.iloc[1] == pytest.approx(0.5)  # half the book traded
        gross = pd.Series([0.01, 0.01], index=w.index)
        net = apply_transaction_costs(gross, to, cost_bps=100.0)  # 1% per unit
        assert net.iloc[1] == pytest.approx(0.01 - 0.005)


class TestStats:
    def test_bootstrap_ci_brackets_point_estimate(self):
        rng = np.random.default_rng(1)
        rets = pd.Series(rng.normal(0.008, 0.04, 120),
                         index=pd.date_range("2010-01-31", periods=120, freq="ME"))
        point, lo, hi = block_bootstrap_sharpe_ci(rets, n_samples=300, seed=1)
        assert lo < point < hi

    def test_jkm_identical_series_not_significant(self):
        rng = np.random.default_rng(2)
        rets = pd.Series(rng.normal(0.005, 0.03, 120),
                         index=pd.date_range("2010-01-31", periods=120, freq="ME"))
        z, p = jobson_korkie_memmel(rets, rets.copy())
        assert p > 0.99 or np.isnan(z) or abs(z) < 1e-6

    def test_jkm_detects_clear_difference(self):
        idx = pd.date_range("2000-01-31", periods=300, freq="ME")
        rng = np.random.default_rng(3)
        base = rng.normal(0.0, 0.03, 300)
        good = pd.Series(base + 0.015, index=idx)
        bad = pd.Series(base - 0.005, index=idx)
        _, p = jobson_korkie_memmel(good, bad)
        assert p < 0.01

    def test_holm_more_conservative_than_raw(self):
        ps = {"a": 0.01, "b": 0.02, "c": 0.04}
        out = holm_bonferroni(ps, alpha=0.05).set_index("comparison")
        assert (out["p_adjusted"] >= pd.Series(ps)).all()
        assert out["p_adjusted"]["a"] == pytest.approx(0.03)  # 3 * 0.01


class TestConfigAndRunner:
    def test_default_yaml_validates(self):
        cfg = load_config()
        assert cfg.walkforward.purge_gap_months >= 1
        assert cfg.portfolio.risk_free_rate_annual > 0  # realistic, non-zero

    def test_defaults_without_yaml(self):
        assert PipelineConfig().portfolio.max_weight == 0.10

    def test_combination_order_is_deterministic(self):
        a = experiment_combinations("all")
        b = experiment_combinations("all")
        assert a == b
        assert len(a) == len(dict.fromkeys(a))  # deduped, order preserved

    def test_strategy_name_format(self):
        c = Combination("xgboost", "lw", "mvo")
        assert c.name == "xgboost_lw_mvo"

    def test_all_covers_every_rq(self):
        names = {c.name for c in experiment_combinations("all")}
        assert "zero_lw_mvo" in names          # rq1 + rq2 overlap deduped
        assert "xgboost_lw_riskparity" in names
        assert "e2e_direct_direct" in names
