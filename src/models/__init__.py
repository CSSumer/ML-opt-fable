"""Return predictors.

torch / xgboost are imported lazily *inside* the model modules — importing
both into one process can segfault on macOS (conflicting libomp runtimes), so
each backtest combination runs in its own subprocess (src/engine/runner.py)
and only imports what it needs.
"""

from __future__ import annotations

from typing import Any


def make_predictor(name: str, config: Any) -> Any:
    """Factory: predictor name -> instance. Keeps heavy imports lazy."""
    name = name.lower()
    if name in ("zero", "histmean"):
        from src.models.naive import HistoricalMeanPredictor, ZeroPredictor

        return ZeroPredictor() if name == "zero" else HistoricalMeanPredictor()
    if name == "ols":
        from src.models.linear import OLSPredictor

        return OLSPredictor()
    if name == "ff3":
        from src.models.linear import FamaFrench3Predictor

        return FamaFrench3Predictor()
    if name == "xgboost":
        from src.models.xgb import XGBoostPredictor

        return XGBoostPredictor(config.models.xgboost, seed=config.models.seed)
    if name == "lstm":
        from src.models.lstm import LSTMPredictor

        return LSTMPredictor(config.models.lstm, seed=config.models.seed)
    if name == "e2e":
        from src.models.end_to_end import EndToEndAllocator

        return EndToEndAllocator(config.models.end_to_end, seed=config.models.seed)
    raise ValueError(f"Unknown predictor: {name!r}")


PREDICTOR_NAMES = list(
    dict.fromkeys(["zero", "histmean", "ols", "ff3", "xgboost", "lstm", "e2e"])
)
