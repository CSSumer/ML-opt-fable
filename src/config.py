"""Pydantic-validated configuration schema.

A single ``config/default.yaml`` drives the whole pipeline. Every field has a
sensible default so a partial YAML (or none at all) still yields a valid
config. Validation failures raise immediately at startup — never mid-backtest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    universe: str = "sp500"
    max_tickers: Optional[int] = 50
    start_date: str = "2005-01-01"
    end_date: Optional[str] = None
    price_cache_dir: str = "data_cache"
    macro_series: list[str] = Field(
        default_factory=lambda: ["UNRATE", "CPIAUCSL", "INDPRO", "FEDFUNDS"]
    )
    alfred_api_key_env: str = "FRED_API_KEY"


class FeatureConfig(BaseModel):
    momentum_windows: list[int] = Field(default_factory=lambda: [1, 3, 6, 12])
    volatility_window: int = 6
    rsi_window: int = 14
    ma_windows: list[int] = Field(default_factory=lambda: [50, 200])
    cross_sectional_rank: bool = True


class WalkForwardConfig(BaseModel):
    train_window_months: int = 60
    purge_gap_months: int = 1
    validation_fraction: float = 0.15
    validation_purge_gap_months: int = 1
    rebalance_frequency: Literal["monthly"] = "monthly"

    @field_validator("purge_gap_months")
    @classmethod
    def _purge_gap_at_least_one(cls, v: int) -> int:
        # Forward-return labels at date t span (t, t+1m]; a gap < 1 month lets
        # the label window overlap the test date — i.e. leakage by design.
        if v < 1:
            raise ValueError(
                "purge_gap_months must be >= 1: forward-return labels overlap "
                "the rebalance date and would leak into training."
            )
        return v

    @field_validator("validation_purge_gap_months")
    @classmethod
    def _val_purge_nonneg(cls, v: int) -> int:
        if v < 1:
            raise ValueError("validation_purge_gap_months must be >= 1.")
        return v

    @field_validator("validation_fraction")
    @classmethod
    def _val_fraction_range(cls, v: float) -> float:
        if not 0.0 <= v < 0.5:
            raise ValueError("validation_fraction must be in [0, 0.5).")
        return v


class XGBoostConfig(BaseModel):
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 25


class LSTMConfig(BaseModel):
    sequence_length: int = 12
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.1
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    early_stopping_patience: int = 5


class EndToEndConfig(BaseModel):
    hidden_size: int = 64
    epochs: int = 60
    learning_rate: float = 1e-3
    early_stopping_patience: int = 8


class ModelConfig(BaseModel):
    seed: int = 42
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    lstm: LSTMConfig = Field(default_factory=LSTMConfig)
    end_to_end: EndToEndConfig = Field(default_factory=EndToEndConfig)


class PortfolioConfig(BaseModel):
    max_weight: float = 0.10
    cardinality_k: int = 15
    risk_free_rate_annual: float = 0.02
    transaction_cost_bps: float = 10.0

    @field_validator("max_weight")
    @classmethod
    def _max_weight_range(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("max_weight must be in (0, 1].")
        return v


class EvaluationConfig(BaseModel):
    bootstrap_samples: int = 2000
    confidence_level: float = 0.95
    multiple_testing: Literal["holm", "bonferroni", "none"] = "holm"
    benchmark: str = "histmean_sample_equal"


class RunnerConfig(BaseModel):
    use_subprocess: bool = True
    subprocess_timeout_seconds: int = 3600
    results_dir: str = "results"


class PipelineConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    walkforward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load and validate the pipeline config.

    ``path=None`` falls back to ``config/default.yaml`` if present, otherwise
    pure schema defaults.
    """
    if path is None:
        candidate = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
        path = candidate if candidate.exists() else None
    if path is None:
        return PipelineConfig()
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return PipelineConfig.model_validate(raw)
