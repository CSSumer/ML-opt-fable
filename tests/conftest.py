"""Shared synthetic-data fixtures. No network, fully deterministic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PipelineConfig  # noqa: E402
from src.data.prices import to_monthly_returns  # noqa: E402
from src.features.engineering import build_feature_matrix  # noqa: E402

N_TICKERS = 12
N_YEARS = 9
SEED = 7


@pytest.fixture(scope="session")
def synthetic_prices() -> pd.DataFrame:
    """Daily prices with a market factor + per-ticker drift so that the
    cross-section is genuinely heterogeneous (predictions must differ)."""
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("2012-01-02", periods=252 * N_YEARS)
    market = rng.normal(0.0003, 0.01, len(dates))
    betas = rng.uniform(0.4, 1.6, N_TICKERS)
    drifts = rng.uniform(-0.0002, 0.0006, N_TICKERS)
    idio = rng.normal(0.0, 0.012, (len(dates), N_TICKERS))
    rets = drifts[None, :] + market[:, None] * betas[None, :] + idio
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(
        prices, index=dates, columns=[f"T{i:02d}" for i in range(N_TICKERS)]
    )


@pytest.fixture(scope="session")
def dataset(synthetic_prices):
    X, y = build_feature_matrix(synthetic_prices, rank_normalise=True)
    monthly = to_monthly_returns(synthetic_prices)
    return X, y, monthly


@pytest.fixture()
def config() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.walkforward.train_window_months = 36
    cfg.walkforward.purge_gap_months = 1
    cfg.evaluation.bootstrap_samples = 200
    cfg.runner.use_subprocess = False  # in-process for tests
    return cfg
