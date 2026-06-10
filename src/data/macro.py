"""Point-in-time macro features from ALFRED (ArchivaL FRED).

Why ALFRED and not FRED
-----------------------
Standard FRED returns today's *revised* history. Macro series like CPI and
industrial production are revised for months after first publication, so a
model trained on revised values sees data that physically did not exist at
decision time. ALFRED exposes every vintage: for each observation we keep the
value *as first published* and stamp it with its publication (realtime_start)
date. Downstream alignment then only reveals an observation once its
publication date has passed.

The network call lives in :func:`fetch_alfred_vintages`; everything after it
is pure and unit-tested offline (``tests/test_no_lookahead.py`` checks the
PIT alignment directly).
"""

from __future__ import annotations

import logging
import os

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ALFRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_alfred_vintages(
    series_id: str,
    api_key: str | None = None,
    api_key_env: str = "FRED_API_KEY",
    start: str = "2000-01-01",
) -> pd.DataFrame:
    """Fetch all vintages of a series from ALFRED.

    Returns a DataFrame with columns ``[obs_date, publication_date, value]``
    where ``publication_date`` is the ALFRED ``realtime_start`` — the first
    day this exact value was publicly available.
    """
    key = api_key or os.environ.get(api_key_env)
    if not key:
        raise RuntimeError(
            f"No ALFRED API key: set the {api_key_env} environment variable "
            "(free at https://fred.stlouisfed.org/docs/api/api_key.html)."
        )
    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "observation_start": start,
        # Full realtime range = every vintage ever published.
        "realtime_start": "1776-07-04",
        "realtime_end": "9999-12-31",
    }
    resp = requests.get(ALFRED_URL, params=params, timeout=60)
    resp.raise_for_status()
    rows = resp.json()["observations"]
    df = pd.DataFrame(rows)
    df = df[df["value"] != "."]
    return pd.DataFrame(
        {
            "obs_date": pd.to_datetime(df["date"]),
            "publication_date": pd.to_datetime(df["realtime_start"]),
            "value": df["value"].astype(float),
        }
    )


def first_release_series(vintages: pd.DataFrame) -> pd.DataFrame:
    """Reduce a vintage table to the first-published value per observation.

    Pure function. Output columns: ``[obs_date, publication_date, value]``
    with exactly one row per obs_date (the earliest publication).
    """
    idx = vintages.groupby("obs_date")["publication_date"].idxmin()
    out = vintages.loc[idx, ["obs_date", "publication_date", "value"]]
    return out.sort_values("obs_date").reset_index(drop=True)


def align_macro_pit(first_release: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """For each backtest date t, return the latest value *published* <= t.

    This is the point-in-time guarantee: an observation is invisible until its
    publication date has passed, regardless of the period it refers to. Pure
    function; tested with synthetic publication lags.
    """
    pub_sorted = first_release.sort_values("publication_date")
    pub_dates = pub_sorted["publication_date"].to_numpy()
    values = pub_sorted["value"].to_numpy()
    out = []
    for t in dates:
        mask_count = (pub_dates <= pd.Timestamp(t).to_datetime64()).sum()
        out.append(values[mask_count - 1] if mask_count > 0 else float("nan"))
    return pd.Series(out, index=dates, dtype=float)


def build_macro_panel(
    series_ids: list[str],
    dates: pd.DatetimeIndex,
    api_key: str | None = None,
    api_key_env: str = "FRED_API_KEY",
    start: str = "2000-01-01",
) -> pd.DataFrame:
    """Fetch + PIT-align several ALFRED series onto the backtest date grid."""
    panel = {}
    for sid in dict.fromkeys(series_ids):  # ordered dedupe, deterministic
        vintages = fetch_alfred_vintages(sid, api_key, api_key_env, start)
        panel[f"macro_{sid.lower()}"] = align_macro_pit(
            first_release_series(vintages), dates
        )
        logger.info("PIT-aligned ALFRED series %s", sid)
    return pd.DataFrame(panel, index=dates)
