#!/usr/bin/env python3
"""Unified backtest CLI with three operational modes.

1. Single combination:
       python scripts/run_backtest.py --predictor xgboost --cov lw --optimiser mvo
2. Experiment group (maps to a research question):
       python scripts/run_backtest.py --experiment rq1
3. Everything:
       python scripts/run_backtest.py --all

Add ``--synthetic`` to run on a deterministic synthetic market (no network,
no API keys) — useful for smoke tests and CI.

Results land in ``results/<run_id>/`` as pickles + CSV tables, ready for
``streamlit run scripts/dashboard.py``.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.data.prices import to_monthly_returns  # noqa: E402
from src.engine.evaluate import summarise  # noqa: E402
from src.engine.runner import Combination, experiment_combinations, run_matrix  # noqa: E402
from src.features.engineering import build_feature_matrix  # noqa: E402
from src.seeding import set_global_seeds  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def make_synthetic_market(
    n_tickers: int = 20, n_years: int = 10, seed: int = 42
) -> pd.DataFrame:
    """Deterministic synthetic daily prices with factor structure."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=252 * n_years)
    market = rng.normal(0.0003, 0.01, len(dates))
    betas = rng.uniform(0.5, 1.5, n_tickers)
    idio = rng.normal(0.0, 0.012, (len(dates), n_tickers))
    rets = market[:, None] * betas[None, :] + idio
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    cols = [f"SYN{i:03d}" for i in range(n_tickers)]
    return pd.DataFrame(prices, index=dates, columns=cols)


def load_dataset(cfg, synthetic: bool):
    if synthetic:
        logger.info("Using deterministic synthetic market (no network).")
        daily = make_synthetic_market(seed=cfg.models.seed)
        macro = None
    else:
        cache = Path(cfg.data.price_cache_dir)
        prices_file = cache / "prices.parquet"
        if not prices_file.exists():
            raise SystemExit(
                f"No cached prices at {prices_file}. Run "
                "`python scripts/download_data.py` first (or pass --synthetic)."
            )
        daily = pd.read_parquet(prices_file)
        macro_file = cache / "macro_pit.parquet"
        macro = pd.read_parquet(macro_file) if macro_file.exists() else None
        if macro is None:
            logger.warning("No PIT macro panel found; running technicals-only.")

    fc = cfg.features
    X, y = build_feature_matrix(
        daily,
        macro_panel=macro,
        momentum_windows=fc.momentum_windows,
        volatility_window=fc.volatility_window,
        rsi_window=fc.rsi_window,
        ma_windows=fc.ma_windows,
        rank_normalise=fc.cross_sectional_rank,
    )
    monthly_returns = to_monthly_returns(daily)
    return X, y, monthly_returns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--predictor")
    parser.add_argument("--cov")
    parser.add_argument("--optimiser")
    parser.add_argument("--experiment", choices=["rq1", "rq2", "rq3", "rq4"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--synthetic", action="store_true",
                        help="run on synthetic data (no network needed)")
    parser.add_argument("--no-subprocess", action="store_true",
                        help="run combinations in-process (debugging)")
    args = parser.parse_args()

    modes = sum([bool(args.predictor), bool(args.experiment), args.all])
    if modes != 1:
        parser.error(
            "Choose exactly one mode: --predictor/--cov/--optimiser, "
            "--experiment rqN, or --all"
        )

    cfg = load_config(args.config)
    if args.no_subprocess:
        cfg.runner.use_subprocess = False
    set_global_seeds(cfg.models.seed)

    if args.predictor:
        if args.predictor == "e2e":
            combos = [Combination("e2e", "direct", "direct")]
        else:
            if not (args.cov and args.optimiser):
                parser.error("--predictor mode needs --cov and --optimiser too")
            combos = [Combination(args.predictor, args.cov, args.optimiser)]
    elif args.experiment:
        combos = experiment_combinations(args.experiment)
    else:
        combos = experiment_combinations("all")

    X, y, monthly_returns = load_dataset(cfg, args.synthetic)
    logger.info("Dataset: %d feature rows, %d months, %d tickers",
                len(X),
                X.index.get_level_values("date").nunique(),
                X.index.get_level_values("ticker").nunique())

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.runner.results_dir) / run_id
    summary = run_matrix(X, y, monthly_returns, cfg, combos, run_dir)

    if summary["results"]:
        tables = summarise(summary["results"], cfg)
        tables["metrics"].to_csv(run_dir / "metrics.csv")
        tables["tests"].to_csv(run_dir / "sharpe_tests.csv", index=False)
        with open(run_dir / "evaluation.pkl", "wb") as f:
            pickle.dump(tables, f)
        print("\n=== Strategy metrics (net of costs) ===")
        cols = ["annualized_return", "annualized_volatility", "sharpe_ratio",
                "sortino_ratio", "max_drawdown", "calmar_ratio"]
        print(tables["metrics"][cols].round(3).to_string())
        print(f"\n=== Sharpe tests vs benchmark '{tables['benchmark']}' "
              f"({cfg.evaluation.multiple_testing} corrected) ===")
        print(tables["tests"].round(4).to_string(index=False))
    if summary["skipped"]:
        print("\n=== Skipped combinations (see run.log for tracebacks) ===")
        for name in summary["skipped"]:
            print(f"  SKIPPED: {name}")
    print(f"\nArtifacts: {run_dir}/  →  streamlit run scripts/dashboard.py")


if __name__ == "__main__":
    main()
