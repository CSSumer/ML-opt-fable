"""Price download and caching.

SURVIVORSHIP BIAS — explicit limitation
---------------------------------------
The universe is the *current* S&P 500 constituent list scraped from Wikipedia.
Firms that were delisted, acquired, or dropped from the index before today are
absent, which mechanically inflates absolute backtest returns. Because every
strategy in the comparison shares the same biased universe, *relative*
comparisons (ML vs. baselines, RQ1–RQ4) remain informative, but absolute
performance numbers must not be quoted as investable returns. This is
documented in the README and must be restated in any thesis write-up.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp500_tickers(max_tickers: int | None = None) -> list[str]:
    """Scrape the current S&P 500 constituents (see survivorship note above)."""
    tables = pd.read_html(SP500_WIKI_URL)
    tickers = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False)
    # dict.fromkeys: dedupe while preserving order (deterministic runs).
    ordered = list(dict.fromkeys(tickers.tolist()))
    return ordered[:max_tickers] if max_tickers else ordered


def download_prices(
    tickers: list[str],
    start: str,
    end: str | None,
    cache_dir: str | Path = "data_cache",
) -> pd.DataFrame:
    """Download adjusted close prices via yfinance, with a parquet cache.

    Returns a (date x ticker) DataFrame of adjusted closes.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "prices.parquet"
    if cache_file.exists():
        cached = pd.read_parquet(cache_file)
        missing = [t for t in tickers if t not in cached.columns]
        if not missing:
            logger.info("Loaded %d tickers from cache %s", len(tickers), cache_file)
            return cached[tickers]

    import yfinance as yf  # lazy: not needed for offline/synthetic runs

    logger.info("Downloading %d tickers from yfinance ...", len(tickers))
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = prices.dropna(axis=1, how="all")
    prices.to_parquet(cache_file)
    return prices


def to_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end simple returns from (daily) adjusted closes."""
    monthly_px = prices.resample("ME").last()
    return monthly_px.pct_change(fill_method=None).iloc[1:]
