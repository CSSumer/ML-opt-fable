"""Statistical rigour layer.

* Bootstrap confidence intervals for Sharpe ratios (circular block bootstrap
  to respect serial dependence in monthly returns).
* Pairwise Sharpe-difference tests: Jobson-Korkie (1981) with the Memmel
  (2003) variance correction.
* Family-wise multiple-testing control via Holm-Bonferroni across the
  strategy family, so 12+ simultaneous comparisons don't manufacture
  significance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine.metrics import sharpe_ratio


def block_bootstrap_sharpe_ci(
    returns: pd.Series,
    n_samples: int = 2000,
    confidence: float = 0.95,
    block_size: int = 6,
    risk_free_annual: float = 0.0,
    seed: int = 42,
) -> tuple[float, float, float]:
    """(point, lo, hi) Sharpe with a circular block bootstrap CI."""
    rng = np.random.default_rng(seed)
    arr = returns.to_numpy(dtype=float)
    n = len(arr)
    point = sharpe_ratio(returns, risk_free_annual)
    if n < block_size + 1:
        return point, float("nan"), float("nan")
    n_blocks = int(np.ceil(n / block_size))
    sharpes = np.empty(n_samples)
    for b in range(n_samples):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_size)[None, :]).ravel() % n
        sample = pd.Series(arr[idx[:n]], index=returns.index)
        sharpes[b] = sharpe_ratio(sample, risk_free_annual)
    alpha = 1.0 - confidence
    lo, hi = np.percentile(sharpes, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def jobson_korkie_memmel(
    returns_a: pd.Series, returns_b: pd.Series, risk_free_annual: float = 0.0
) -> tuple[float, float]:
    """Test H0: Sharpe(a) == Sharpe(b). Returns (z_stat, p_value)."""
    from scipy.stats import norm

    df = pd.concat([returns_a, returns_b], axis=1, keys=["a", "b"]).dropna()
    if len(df) < 12:
        return float("nan"), float("nan")
    rf_m = (1.0 + risk_free_annual) ** (1.0 / 12.0) - 1.0
    a = df["a"].to_numpy(dtype=float) - rf_m
    b = df["b"].to_numpy(dtype=float) - rf_m
    t = len(a)
    mu_a, mu_b = a.mean(), b.mean()
    s_a, s_b = a.std(ddof=1), b.std(ddof=1)
    if s_a == 0 or s_b == 0:
        return float("nan"), float("nan")
    sh_a, sh_b = mu_a / s_a, mu_b / s_b
    rho = float(np.corrcoef(a, b)[0, 1])
    # Memmel (2003) corrected asymptotic variance of the Sharpe difference
    var = (1.0 / t) * (
        2.0 * (1.0 - rho)
        + 0.5 * (sh_a**2 + sh_b**2 - 2.0 * sh_a * sh_b * rho**2)
    )
    if var <= 0:
        return float("nan"), float("nan")
    z = (sh_a - sh_b) / np.sqrt(var)
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    return float(z), float(p)


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> pd.DataFrame:
    """Holm step-down correction. Returns adjusted p-values + reject flags."""
    items = [(k, v) for k, v in p_values.items() if not np.isnan(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    rows, running_max, rejecting = [], 0.0, True
    for rank, (name, p) in enumerate(items):
        adj = min((m - rank) * p, 1.0)
        running_max = max(running_max, adj)  # enforce monotonicity
        reject = rejecting and (running_max < alpha)
        if not reject:
            rejecting = False  # step-down: once we fail, all later fail
        rows.append({"comparison": name, "p_raw": p,
                     "p_adjusted": running_max, "reject_h0": reject})
    for name, p in p_values.items():
        if np.isnan(p):
            rows.append({"comparison": name, "p_raw": p,
                         "p_adjusted": float("nan"), "reject_h0": False})
    return pd.DataFrame(rows)


def pairwise_sharpe_tests(
    strategy_returns: dict[str, pd.Series],
    benchmark: str,
    risk_free_annual: float = 0.0,
    method: str = "holm",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Each strategy vs the benchmark, corrected across the family."""
    raw: dict[str, float] = {}
    zs: dict[str, float] = {}
    bench = strategy_returns.get(benchmark)
    for name, rets in strategy_returns.items():
        if name == benchmark or bench is None:
            continue
        z, p = jobson_korkie_memmel(rets, bench, risk_free_annual)
        raw[f"{name}_vs_{benchmark}"] = p
        zs[f"{name}_vs_{benchmark}"] = z
    if not raw:  # single strategy or missing benchmark: nothing to test
        return pd.DataFrame(
            columns=["comparison", "p_raw", "p_adjusted", "reject_h0", "z_stat"]
        )
    if method == "none":
        df = pd.DataFrame(
            [{"comparison": k, "p_raw": v, "p_adjusted": v,
              "reject_h0": (v < alpha) if not np.isnan(v) else False}
             for k, v in raw.items()]
        )
    elif method == "bonferroni":
        m = len(raw)
        df = pd.DataFrame(
            [{"comparison": k, "p_raw": v, "p_adjusted": min(v * m, 1.0),
              "reject_h0": (v * m < alpha) if not np.isnan(v) else False}
             for k, v in raw.items()]
        )
    else:  # holm (default)
        df = holm_bonferroni(raw, alpha)
    df["z_stat"] = df["comparison"].map(zs)
    return df.reset_index(drop=True)
