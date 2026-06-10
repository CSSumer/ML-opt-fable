"""Aggregate backtest results into thesis-ready tables.

Produces, per strategy: financial metrics, bootstrap Sharpe CIs, resource
logs; plus the family of pairwise Sharpe tests against the configured
benchmark with multiple-testing correction.
"""

from __future__ import annotations

import pandas as pd

from src.config import PipelineConfig
from src.engine.metrics import compute_all_metrics
from src.engine.stats import block_bootstrap_sharpe_ci, pairwise_sharpe_tests
from src.engine.walkforward import BacktestResult


def summarise(
    results: dict[str, BacktestResult], config: PipelineConfig
) -> dict[str, pd.DataFrame]:
    rf = config.portfolio.risk_free_rate_annual
    rows = []
    for name, res in results.items():
        m = compute_all_metrics(res.net_returns, res.weights_history, rf)
        sharpe, lo, hi = block_bootstrap_sharpe_ci(
            res.net_returns,
            n_samples=config.evaluation.bootstrap_samples,
            confidence=config.evaluation.confidence_level,
            risk_free_annual=rf,
            seed=config.models.seed,
        )
        rows.append(
            {
                "strategy": name,
                **m,
                "sharpe_ci_low": lo,
                "sharpe_ci_high": hi,
                **res.resource_log,
                "n_skipped_dates": len(res.skipped_dates),
            }
        )
    metrics_df = pd.DataFrame(rows).set_index("strategy").sort_values(
        "sharpe_ratio", ascending=False
    )

    returns_map = {n: r.net_returns for n, r in results.items()}
    benchmark = config.evaluation.benchmark
    if benchmark not in returns_map and returns_map:
        # fall back to the first strategy, deterministically
        benchmark = next(iter(returns_map))
    tests_df = pairwise_sharpe_tests(
        returns_map,
        benchmark=benchmark,
        risk_free_annual=rf,
        method=config.evaluation.multiple_testing,
    )
    return {"metrics": metrics_df, "tests": tests_df, "benchmark": benchmark}
