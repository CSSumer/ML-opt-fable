# ML-Enhanced Long-Term Portfolio Optimisation

A reproducible research pipeline that compares machine-learning return
predictors against traditional econometric and naive baselines inside a
walk-forward portfolio backtest. Built for an academic thesis:
methodological correctness, computational transparency, and reproducibility
take priority over raw performance.

Non-technical readers: start with **[BEGINNER_GUIDE.md](BEGINNER_GUIDE.md)**.

## Research questions → experiment groups

| RQ | Question | Varies | Held fixed | Run with |
|----|----------|--------|-----------|----------|
| RQ1 | Do ML predictors (XGBoost, LSTM) beat naive (zero, historical mean) and econometric (OLS, Fama-French 3-factor) baselines? | predictor | Ledoit-Wolf cov, MVO | `--experiment rq1` |
| RQ2 | Which covariance estimator minimises realised volatility? (Sample, Ledoit-Wolf, OAS, PCA-factor) | covariance | zero forecast ⇒ pure min-variance | `--experiment rq2` |
| RQ3 | Which optimiser gives the best risk-adjusted returns? (Equal-Weight, MVO, Risk-Parity, Cardinality) | optimiser | XGBoost forecast, Ledoit-Wolf cov | `--experiment rq3` |
| RQ4 | Two-stage (predict-then-optimise) vs end-to-end (network outputs weights via a differentiable negative-Sharpe loss)? | paradigm | — | `--experiment rq4` |

Strategy names auto-generate as `{predictor}_{cov}_{optimiser}`
(e.g. `xgboost_lw_mvo`); the end-to-end model is `e2e_direct_direct`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 0) Sanity check everything offline (no network, no API keys):
python -m pytest tests/ -q
python scripts/run_backtest.py --experiment rq1 --synthetic

# 1) Download real data (needs internet + a free FRED API key for macro):
export FRED_API_KEY=your_key_here
python scripts/download_data.py

# 2) Run backtests — three CLI modes:
python scripts/run_backtest.py --predictor xgboost --cov lw --optimiser mvo
python scripts/run_backtest.py --experiment rq1     # rq1 | rq2 | rq3 | rq4
python scripts/run_backtest.py --all

# 3) Explore results in the non-quant dashboard:
streamlit run scripts/dashboard.py
```

Results land in `results/<run_id>/`: `results.pkl` (full per-strategy
returns/weights), `metrics.csv`, `sharpe_tests.csv`, and `run.log` (including
full tracebacks for any skipped combination).

## Architecture

Three small duck-typed interfaces (`src/interfaces.py`) let components mix
freely; the engine never special-cases a model (except detecting the
`WeightPredictor` protocol for end-to-end allocators):

- `ReturnPredictor`: `fit(X_train, y_train)`, `predict(X_test) -> Series[ticker]`, `get_name()`
- `PortfolioOptimiser`: `optimise(expected_returns, cov_matrix, tickers) -> Series[ticker]`, `get_name()`
- `CovarianceEstimator`: `estimate(returns) -> ndarray`, `get_name()`

All features carry a `(date, ticker)` MultiIndex. One
`config/default.yaml`, validated by the Pydantic schema in `src/config.py`,
controls everything.

```
config/      default.yaml + Pydantic schema (src/config.py)
src/
  data/      prices (yfinance, cached) · macro (ALFRED vintages, PIT-aligned)
  features/  technicals + broadcast macro + cross-sectional rank normalisation
  models/    zero · histmean · ols · ff3 · xgboost · lstm · e2e
  covariance/ sample · ledoit-wolf · oas · pca-factor (all PSD-guaranteed)
  optimisers/ equal · mvo · riskparity · cardinality (+ water-filling projection)
  engine/    walk-forward engine · subprocess runner · metrics · statistics
scripts/     download_data.py · run_backtest.py · dashboard.py
tests/       test_no_lookahead.py (mandatory integration) + unit tests
```

## Methodological guarantees

**Point-in-time macro (ALFRED, not FRED).** Macro series are pulled vintage
by vintage from ALFRED; for each observation we keep the value *as first
published* and only reveal it once its publication date has passed. Models
never see revised history or unpublished prints
(`src/data/macro.py`, tested offline in `tests/test_no_lookahead.py`).

**Purged walk-forward windows.** The label at date *d* is the return over
(*d*, *d*+1m], so training ends at `t − purge_gap_months` with
`purge_gap ≥ 1` enforced by the config schema. Models needing early stopping
get a validation set carved from the tail of the training window, separated
from the training core by its own purge gap. The test window is never read.

**Hard look-ahead guard.** Before any model call the engine asserts
`train_max_date < test_min_date`; a violation logs an explicit data-leakage
message and crashes the run (`LookAheadError`). The integration test injects
future data and verifies the crash.

**Strict scaler isolation.** All feature scalers fit on training data only;
validation/test rows are transformed with frozen parameters.

**Genuine LSTM inference sequences.** Each ticker's chronological feature
history is cached at `fit()`; `predict()` assembles a real
`sequence_length`-month window ending at the test row. The current month is
never tiled into a fake constant sequence (regression-tested).

**Constraint projection.** Every optimiser's solution is projected onto
{w ≥ 0, Σw = 1, w ≤ max_weight} by an iterative water-filling algorithm,
with an equal-weight fallback when `N × max_weight < 1` is infeasible.

**Subprocess isolation (macOS libomp).** torch and xgboost bundle
conflicting OpenMP runtimes and segfault when sharing a process on macOS.
Both are imported lazily inside model code, and each combination runs in an
isolated `spawn` subprocess. A failing combination writes its full traceback
to `run.log`, is recorded as a skip, and the rest of the matrix continues.

**Determinism.** Global seeds for numpy/torch/xgboost
(`src/seeding.py`); combination parsing preserves order via
`dict.fromkeys()` — never an unordered `set`.

## Evaluation

- **Financial:** annualised return & volatility, Sharpe (non-zero risk-free
  rate from the YAML), Sortino, max drawdown, Calmar, turnover with
  transaction costs deducted at `transaction_cost_bps` per unit of turnover.
  Annualisation uses **elapsed calendar time**, never a count of sparse rows.
- **Statistical:** circular block-bootstrap confidence intervals for Sharpe
  ratios; pairwise Sharpe-difference tests (Jobson-Korkie with Memmel's
  correction); Holm-Bonferroni (or Bonferroni) correction across the
  strategy family to control error rates over 12+ simultaneous comparisons.
- **Resources:** `training_time_seconds`, mean `inference_latency_ms`, and
  peak memory per strategy.

## Known limitations (state these in the thesis)

1. **Survivorship bias.** The universe is the *current* S&P 500 constituent
   list; delisted firms are absent. Absolute returns are inflated; relative
   comparisons between strategies (the object of RQ1–RQ4) remain valid since
   all strategies share the same universe. (`src/data/prices.py`)
2. **Fama-French proxy.** Without CRSP/Compustat, FF3 factors are proxied
   from the feature set (documented in `src/models/linear.py`). Swap in the
   official Ken French library factors for publication-grade results.
3. **Cardinality optimiser** uses a screen-then-optimise heuristic; exact
   cardinality-constrained MVO is NP-hard.
4. Transaction costs are linear in turnover; no market impact model.

## Testing

```bash
python -m pytest tests/ -q
```

`tests/test_no_lookahead.py` is a non-mocked integration suite that runs the
real engine on synthetic data and verifies: the look-ahead guard catches
injected future data, PIT macro alignment respects publication lags, every
covariance estimator returns a PSD matrix (including T < N), and model
output is non-degenerate. torch/xgboost tests skip automatically when those
libraries are absent.
