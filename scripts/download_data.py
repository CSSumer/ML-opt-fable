#!/usr/bin/env python3
"""Download and cache all input data (prices + PIT macro).

Usage:
    python scripts/download_data.py [--config config/default.yaml]

Requires network access and (for macro) a free ALFRED/FRED API key in the
FRED_API_KEY environment variable. Everything is cached under
``data.price_cache_dir`` so the backtest itself runs offline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.data.macro import build_macro_panel  # noqa: E402
from src.data.prices import download_prices, get_sp500_tickers  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    tickers = get_sp500_tickers(cfg.data.max_tickers)
    logger.info("Universe: %d tickers (current S&P 500 constituents — "
                "survivorship bias documented in README)", len(tickers))
    prices = download_prices(
        tickers, cfg.data.start_date, cfg.data.end_date, cfg.data.price_cache_dir
    )
    logger.info("Prices: %s rows x %s tickers", *prices.shape)

    month_ends = prices.resample("ME").last().index
    macro = build_macro_panel(
        cfg.data.macro_series,
        month_ends,
        api_key_env=cfg.data.alfred_api_key_env,
        start=cfg.data.start_date,
    )
    macro_path = Path(cfg.data.price_cache_dir) / "macro_pit.parquet"
    macro.to_parquet(macro_path)
    logger.info("PIT macro panel saved to %s", macro_path)


if __name__ == "__main__":
    main()
