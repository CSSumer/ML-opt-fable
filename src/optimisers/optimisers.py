"""Portfolio optimisers (RQ3) and the constraint-projection safety net.

Every optimiser's raw solution passes through :func:`project_weights`, an
iterative water-filling projection onto the feasible set
{w >= 0, sum w = 1, w <= max_weight}. If the set is mathematically infeasible
(N * max_weight < 1) we fall back to equal weight rather than crash.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def project_weights(
    weights: np.ndarray, max_weight: float = 1.0, tol: float = 1e-12
) -> np.ndarray:
    """Iterative water-filling projection onto the long-only capped simplex.

    Clip negatives, then repeatedly: renormalise the uncapped mass to fill
    the budget left by capped names, cap any new violators, repeat. Each pass
    caps at least one new asset, so it terminates in <= N iterations.
    Falls back to equal weight when constraints are infeasible.
    """
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    n = len(w)
    if n == 0:
        return w
    if n * max_weight < 1.0 - tol:  # infeasible: cannot reach sum=1 under cap
        logger.warning(
            "Infeasible constraints (N=%d, max_weight=%.4f): equal-weight fallback.",
            n, max_weight,
        )
        return np.full(n, 1.0 / n)
    if w.sum() <= tol:
        w = np.full(n, 1.0 / n)

    w = w / w.sum()
    capped = np.zeros(n, dtype=bool)
    for _ in range(n):
        over = (w > max_weight + tol) & ~capped
        if not over.any():
            break
        capped |= over
        w[capped] = max_weight
        residual = 1.0 - capped.sum() * max_weight
        free = ~capped
        free_sum = w[free].sum()
        if free_sum <= tol:
            # spread residual uniformly over free names (or done if none)
            if free.any():
                w[free] = residual / free.sum()
            break
        w[free] = w[free] * (residual / free_sum)
    return w / w.sum()


class EqualWeightOptimiser:
    def __init__(self, max_weight: float = 1.0) -> None:
        self.max_weight = max_weight

    def optimise(
        self, expected_returns: pd.Series, cov_matrix: np.ndarray, tickers: list[str]
    ) -> pd.Series:
        n = len(tickers)
        return pd.Series(project_weights(np.full(n, 1.0 / n), self.max_weight),
                         index=tickers)

    def get_name(self) -> str:
        return "equal"


class MeanVarianceOptimiser:
    """Max-Sharpe-style mean-variance: maximise mu'w - (gamma/2) w'Sigma w."""

    def __init__(self, max_weight: float = 0.10, risk_aversion: float = 10.0) -> None:
        self.max_weight = max_weight
        self.risk_aversion = risk_aversion

    def optimise(
        self, expected_returns: pd.Series, cov_matrix: np.ndarray, tickers: list[str]
    ) -> pd.Series:
        n = len(tickers)
        mu = expected_returns.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
        sigma = np.asarray(cov_matrix, dtype=float)

        def neg_utility(w: np.ndarray) -> float:
            return -(mu @ w) + 0.5 * self.risk_aversion * (w @ sigma @ w)

        def grad(w: np.ndarray) -> np.ndarray:
            return -mu + self.risk_aversion * (sigma @ w)

        w0 = np.full(n, 1.0 / n)
        res = minimize(
            neg_utility, w0, jac=grad, method="SLSQP",
            bounds=[(0.0, self.max_weight)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"maxiter": 500, "ftol": 1e-10},
        )
        w = res.x if res.success else w0
        if not res.success:
            logger.warning("MVO solver failed (%s); using equal weight.", res.message)
        return pd.Series(project_weights(w, self.max_weight), index=tickers)

    def get_name(self) -> str:
        return "mvo"


class RiskParityOptimiser:
    """Equal risk contribution via Newton-style multiplicative iteration."""

    def __init__(self, max_weight: float = 1.0, n_iter: int = 200) -> None:
        self.max_weight = max_weight
        self.n_iter = n_iter

    def optimise(
        self, expected_returns: pd.Series, cov_matrix: np.ndarray, tickers: list[str]
    ) -> pd.Series:
        n = len(tickers)
        sigma = np.asarray(cov_matrix, dtype=float) + 1e-10 * np.eye(n)
        w = np.full(n, 1.0 / n)
        for _ in range(self.n_iter):
            marginal = sigma @ w
            # fixed point of equal risk contribution: w_i ∝ 1 / (Sigma w)_i
            w_new = (1.0 / np.clip(marginal, 1e-12, None))
            w_new /= w_new.sum()
            if np.max(np.abs(w_new - w)) < 1e-10:
                w = w_new
                break
            w = 0.5 * w + 0.5 * w_new
        return pd.Series(project_weights(w, self.max_weight), index=tickers)

    def get_name(self) -> str:
        return "riskparity"


class CardinalityConstrainedOptimiser:
    """Hold only the top-k assets by expected return, then MVO inside them.

    Exact cardinality-constrained MVO is NP-hard; this screen-then-optimise
    heuristic is the standard tractable relaxation and is documented as such.
    """

    def __init__(self, k: int = 15, max_weight: float = 0.10) -> None:
        self.k = k
        self.max_weight = max_weight

    def optimise(
        self, expected_returns: pd.Series, cov_matrix: np.ndarray, tickers: list[str]
    ) -> pd.Series:
        mu = expected_returns.reindex(tickers).fillna(0.0)
        k = min(self.k, len(tickers))
        top = list(mu.sort_values(ascending=False).index[:k])
        pos = [tickers.index(t) for t in top]
        sub_cov = np.asarray(cov_matrix)[np.ix_(pos, pos)]
        inner = MeanVarianceOptimiser(max_weight=self.max_weight)
        w_top = inner.optimise(mu.loc[top], sub_cov, top)
        return w_top.reindex(tickers).fillna(0.0)

    def get_name(self) -> str:
        return "cardinality"


def make_optimiser(name: str, max_weight: float = 0.10, cardinality_k: int = 15):
    name = name.lower()
    if name == "equal":
        return EqualWeightOptimiser(max_weight=1.0)  # EW baseline is uncapped
    if name == "mvo":
        return MeanVarianceOptimiser(max_weight=max_weight)
    if name == "riskparity":
        return RiskParityOptimiser(max_weight=max_weight)
    if name == "cardinality":
        return CardinalityConstrainedOptimiser(k=cardinality_k, max_weight=max_weight)
    raise ValueError(f"Unknown optimiser: {name!r}")


OPTIMISER_NAMES = list(dict.fromkeys(["equal", "mvo", "riskparity", "cardinality"]))
