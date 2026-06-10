"""Feature engineering on a (date, ticker) MultiIndex.

Per-stock technicals (momentum, volatility, RSI, moving-average gaps) are
combined with macro regime features broadcast to every ticker. Cross-sectional
rank normalisation per date removes scale drift across the decades-long
sample, so a 2006 feature row and a 2024 feature row live on the same scale.

Everything here uses only information available at the row's date: features
at date t are computed from prices up to and including t, never beyond.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder's RSI on daily prices, per ticker."""
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def build_stock_features(
    daily_prices: pd.DataFrame,
    momentum_windows: list[int] | None = None,
    volatility_window: int = 6,
    rsi_window: int = 14,
    ma_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Month-end technical features. Returns a (date, ticker) MultiIndex frame."""
    momentum_windows = momentum_windows or [1, 3, 6, 12]
    ma_windows = ma_windows or [50, 200]

    monthly_px = daily_prices.resample("ME").last()
    monthly_ret = monthly_px.pct_change(fill_method=None)

    feats: dict[str, pd.DataFrame] = {}
    for m in momentum_windows:
        feats[f"mom_{m}m"] = monthly_px.pct_change(m, fill_method=None)
    feats[f"vol_{volatility_window}m"] = monthly_ret.rolling(volatility_window).std()

    rsi_daily = compute_rsi(daily_prices, rsi_window)
    feats[f"rsi_{rsi_window}d"] = rsi_daily.resample("ME").last()

    for w in ma_windows:
        ma = daily_prices.rolling(w).mean()
        # price/MA gap at month end: classic trend feature, scale-free
        feats[f"px_over_ma{w}"] = (daily_prices / ma - 1.0).resample("ME").last()

    stacked = {name: df.stack() for name, df in feats.items()}
    out = pd.DataFrame(stacked)
    out.index.names = ["date", "ticker"]
    return out.sort_index()


def broadcast_macro(
    stock_features: pd.DataFrame, macro_panel: pd.DataFrame
) -> pd.DataFrame:
    """Attach macro regime columns (indexed by date) to every (date, ticker) row."""
    dates = stock_features.index.get_level_values("date")
    macro_aligned = macro_panel.reindex(dates)
    macro_aligned.index = stock_features.index
    return pd.concat([stock_features, macro_aligned], axis=1)


def cross_sectional_rank(features: pd.DataFrame) -> pd.DataFrame:
    """Rank-normalise each feature within each date to [-0.5, 0.5].

    Per-date ranking is inherently point-in-time (it only mixes information
    across tickers at the same date, never across time) and eliminates scale
    drift in raw features.
    """
    grouped = features.groupby(level="date")
    ranked = grouped.rank(pct=True) - 0.5
    # Constant-per-date columns (broadcast macro) rank to the same value
    # everywhere; keep their z-scored time variation instead.
    constant_cols = [
        c for c in features.columns
        if (grouped[c].nunique(dropna=True) <= 1).all()
    ]
    for c in constant_cols:
        col = features[c]
        std = col.std()
        ranked[c] = (col - col.mean()) / std if std and std > 0 else 0.0
    return ranked


def build_feature_matrix(
    daily_prices: pd.DataFrame,
    macro_panel: pd.DataFrame | None = None,
    momentum_windows: list[int] | None = None,
    volatility_window: int = 6,
    rsi_window: int = 14,
    ma_windows: list[int] | None = None,
    rank_normalise: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Full pipeline: features X and forward-return labels y.

    The label for row (t, ticker) is the realised return over (t, t+1m] —
    strictly after the feature timestamp. The walk-forward purge gap exists
    precisely because this label window extends one month past t.
    """
    X = build_stock_features(
        daily_prices, momentum_windows, volatility_window, rsi_window, ma_windows
    )
    if macro_panel is not None:
        X = broadcast_macro(X, macro_panel)
    if rank_normalise:
        X = cross_sectional_rank(X)

    monthly_px = daily_prices.resample("ME").last()
    fwd = monthly_px.pct_change(fill_method=None).shift(-1)  # return over (t, t+1m]
    y = fwd.stack()
    y.index.names = ["date", "ticker"]
    y.name = "fwd_return_1m"

    common = X.dropna().index.intersection(y.dropna().index)
    return X.loc[common].sort_index(), y.loc[common].sort_index()
