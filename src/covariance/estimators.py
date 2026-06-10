"""Covariance estimators (RQ2). All outputs are guaranteed PSD.

Estimating an N x N covariance from T monthly observations with T ~ N is
ill-posed; shrinkage (Ledoit-Wolf, OAS) and factor structure (PCA) are the
standard cures. Each estimate passes through a symmetric-eigenvalue-clip
projection so downstream optimisers can rely on PSD inputs unconditionally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import OAS, LedoitWolf


def nearest_psd(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone by clipping eigenvalues."""
    sym = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    if eigvals.min() >= 0.0:
        return sym
    clipped = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(clipped) @ eigvecs.T


def _prep(returns: pd.DataFrame) -> np.ndarray:
    arr = returns.to_numpy(dtype=float)
    # column-wise demean handled by estimators; just drop rows with any NaN
    return arr[~np.isnan(arr).any(axis=1)]


class SampleCovariance:
    def estimate(self, returns: pd.DataFrame) -> np.ndarray:
        arr = _prep(returns)
        return nearest_psd(np.cov(arr, rowvar=False))

    def get_name(self) -> str:
        return "sample"


class LedoitWolfCovariance:
    def estimate(self, returns: pd.DataFrame) -> np.ndarray:
        return nearest_psd(LedoitWolf().fit(_prep(returns)).covariance_)

    def get_name(self) -> str:
        return "lw"


class OASCovariance:
    def estimate(self, returns: pd.DataFrame) -> np.ndarray:
        return nearest_psd(OAS().fit(_prep(returns)).covariance_)

    def get_name(self) -> str:
        return "oas"


class PCAFactorCovariance:
    """K-factor model: Sigma = B F B' + diag(idiosyncratic variance)."""

    def __init__(self, n_factors: int = 5) -> None:
        self.n_factors = n_factors

    def estimate(self, returns: pd.DataFrame) -> np.ndarray:
        arr = _prep(returns)
        arr = arr - arr.mean(axis=0)
        t, n = arr.shape
        k = min(self.n_factors, n - 1, max(t - 1, 1))
        cov = np.cov(arr, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1][:k]
        loadings = eigvecs[:, order] * np.sqrt(np.clip(eigvals[order], 0, None))
        factor_part = loadings @ loadings.T
        idio = np.clip(np.diag(cov) - np.diag(factor_part), 1e-10, None)
        return nearest_psd(factor_part + np.diag(idio))

    def get_name(self) -> str:
        return "pca"


def make_covariance(name: str):
    name = name.lower()
    mapping = {
        "sample": SampleCovariance,
        "lw": LedoitWolfCovariance,
        "oas": OASCovariance,
        "pca": PCAFactorCovariance,
    }
    if name not in mapping:
        raise ValueError(f"Unknown covariance estimator: {name!r}")
    return mapping[name]()


COVARIANCE_NAMES = list(dict.fromkeys(["sample", "lw", "oas", "pca"]))
