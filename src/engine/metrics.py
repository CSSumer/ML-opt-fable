"""Financial performance metrics.

Annualisation uses *elapsed calendar time*, not a count of return rows: a
sparse series spanning ten years annualises over ten years even if some
months are missing, so sparse and dense strategies stay comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12.0


def elapsed_years(returns: pd.Series) -> float:
    if len(returns) < 2:
        return max(len(returns) / MONTHS_PER_YEAR, 1e-9)
    start, end = returns.index[0], returns.index[-1]
    # include the first month: index dates mark month-ends
    days = (end - start).days + 30.44
    return max(days / 365.25, 1e-9)


def annualized_return(returns: pd.Series) -> float:
    total = float((1.0 + returns).prod())
    if total <= 0:
        return -1.0
    return total ** (1.0 / elapsed_years(returns)) - 1.0


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1)) * np.sqrt(MONTHS_PER_YEAR)


def sharpe_ratio(returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    rf_monthly = (1.0 + risk_free_annual) ** (1.0 / MONTHS_PER_YEAR) - 1.0
    excess = returns - rf_monthly
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd) * np.sqrt(MONTHS_PER_YEAR)


def sortino_ratio(returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    rf_monthly = (1.0 + risk_free_annual) ** (1.0 / MONTHS_PER_YEAR) - 1.0
    excess = returns - rf_monthly
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    dd = np.sqrt(float((downside**2).mean()))
    if dd == 0:
        return float("inf")
    return float(excess.mean() / dd) * np.sqrt(MONTHS_PER_YEAR)


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns).cumprod()
    peak = wealth.cummax()
    return float((wealth / peak - 1.0).min())


def calmar_ratio(returns: pd.Series) -> float:
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return float("inf")
    return annualized_return(returns) / mdd


def turnover_series(weights_history: pd.DataFrame) -> pd.Series:
    """One-way turnover per rebalance: 0.5 * sum |w_t - w_{t-1}|."""
    aligned = weights_history.fillna(0.0)
    diffs = aligned.diff()
    to = 0.5 * diffs.abs().sum(axis=1)
    to.iloc[0] = 0.5 * aligned.iloc[0].abs().sum()  # initial buy-in
    return to


def apply_transaction_costs(
    gross_returns: pd.Series, turnover: pd.Series, cost_bps: float
) -> pd.Series:
    cost = turnover.reindex(gross_returns.index).fillna(0.0) * cost_bps / 1e4
    return gross_returns - cost


def compute_all_metrics(
    returns: pd.Series,
    weights_history: pd.DataFrame | None = None,
    risk_free_annual: float = 0.0,
) -> dict[str, float]:
    out = {
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_annual),
        "sortino_ratio": sortino_ratio(returns, risk_free_annual),
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns),
        "n_months": float(len(returns)),
        "elapsed_years": elapsed_years(returns),
    }
    if weights_history is not None and len(weights_history) > 0:
        to = turnover_series(weights_history)
        out["avg_monthly_turnover"] = float(to.iloc[1:].mean()) if len(to) > 1 else 0.0
    return out
