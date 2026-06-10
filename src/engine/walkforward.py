"""Walk-forward backtest engine with hard look-ahead protection.

At each monthly rebalance date t:

1. Training window = the ``train_window_months`` of feature dates ending at
   ``t - purge_gap_months``. The purge gap (>= 1, enforced by the config
   schema) exists because the label at date d is the return over (d, d+1m] —
   the label of the last permissible training date must be fully realised
   before t.
2. Models that support early stopping get a validation split carved from the
   tail of the training window, separated from the training core by a
   validation purge gap. The test window is never touched.
3. HARD LOOK-AHEAD GUARD: before any model call we assert
   ``train_max_date < test_min_date`` and crash loudly on violation.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.engine.resources import ResourceTracker

logger = logging.getLogger(__name__)


class LookAheadError(RuntimeError):
    """Raised when training data would include the test period. Fatal."""


def check_no_lookahead(
    train_dates: pd.DatetimeIndex | pd.Index,
    test_min_date: pd.Timestamp,
    context: str = "",
) -> None:
    """Hard assertion that all training dates strictly precede the test date.

    This is the last line of defence: if windowing logic ever regresses, the
    run dies here with an explicit data-leakage log instead of silently
    producing inflated results.
    """
    if len(train_dates) == 0:
        raise LookAheadError(f"Empty training window at {test_min_date} {context}")
    train_max = pd.Timestamp(pd.DatetimeIndex(train_dates).max())
    test_min = pd.Timestamp(test_min_date)
    if not train_max < test_min:
        msg = (
            "DATA LEAKAGE DETECTED — train_max_date "
            f"{train_max} >= test_min_date {test_min} {context}. "
            "Aborting run: results would be invalid."
        )
        logger.critical(msg)
        raise LookAheadError(msg)


@dataclass
class BacktestResult:
    strategy_name: str
    gross_returns: pd.Series
    net_returns: pd.Series
    weights_history: pd.DataFrame
    turnover: pd.Series
    resource_log: dict = field(default_factory=dict)
    skipped_dates: list = field(default_factory=list)


def _fit_accepts_validation(predictor) -> bool:
    try:
        return "X_val" in inspect.signature(predictor.fit).parameters
    except (TypeError, ValueError):
        return False


class WalkForwardEngine:
    """Runs one (predictor, covariance, optimiser) combination."""

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        monthly_returns: pd.DataFrame,
        config: PipelineConfig,
    ) -> None:
        if list(X.index.names) != ["date", "ticker"]:
            raise ValueError("X must carry a (date, ticker) MultiIndex.")
        self.X = X.sort_index()
        self.y = y.sort_index()
        self.monthly_returns = monthly_returns.sort_index()
        self.cfg = config
        self.dates = self.X.index.get_level_values("date").unique().sort_values()

    # ------------------------------------------------------------- windowing
    def _split_windows(self, t: pd.Timestamp):
        wf = self.cfg.walkforward
        train_end = t - pd.DateOffset(months=wf.purge_gap_months)
        train_start = train_end - pd.DateOffset(months=wf.train_window_months)
        train_dates = self.dates[(self.dates > train_start) & (self.dates <= train_end)]

        val_dates = train_dates[:0]
        core_dates = train_dates
        if wf.validation_fraction > 0 and len(train_dates) >= 12:
            n_val = max(1, int(round(len(train_dates) * wf.validation_fraction)))
            val_dates = train_dates[-n_val:]
            # validation purge gap between training core and validation tail
            core_cutoff = val_dates[0] - pd.DateOffset(
                months=wf.validation_purge_gap_months
            )
            core_dates = train_dates[train_dates <= core_cutoff]
        return core_dates, val_dates, train_dates

    # ------------------------------------------------------------------- run
    def run(self, predictor, cov_estimator=None, optimiser=None) -> BacktestResult:
        wf = self.cfg.walkforward
        is_e2e = hasattr(predictor, "predict_weights")
        if not is_e2e and (cov_estimator is None or optimiser is None):
            raise ValueError("Two-stage strategies need both cov and optimiser.")

        name = (
            f"{predictor.get_name()}_direct_direct"
            if is_e2e
            else f"{predictor.get_name()}_{cov_estimator.get_name()}_"
            f"{optimiser.get_name()}"
        )
        tracker = ResourceTracker()
        tracker.start_memory()

        # First rebalance once at least half the training window (and a year
        # minimum) of usable core dates exists.
        min_core = max(12, wf.train_window_months // 2)
        first_idx = None
        for i, t in enumerate(self.dates):
            core, _, _ = self._split_windows(t)
            if len(core) >= min_core:
                first_idx = i
                break
        if first_idx is None:
            raise RuntimeError(
                f"Not enough history: need >= {min_core} core training months "
                "before the first rebalance."
            )

        port_returns, weights_rows, skipped = {}, {}, []
        ret_dates = self.monthly_returns.index

        for t in self.dates[first_idx:]:
            future = ret_dates[ret_dates > t]
            if len(future) == 0:
                break  # last date has no realised next-month return
            next_date = future[0]

            core_dates, val_dates, _ = self._split_windows(t)

            # ---- HARD LOOK-AHEAD GUARD (crash loudly on violation) --------
            check_no_lookahead(core_dates, t, context=f"[{name}]")
            if len(val_dates) > 0:
                check_no_lookahead(val_dates, t, context=f"[{name}/validation]")

            idx = self.X.index.get_level_values("date")
            X_tr = self.X[idx.isin(core_dates)]
            y_tr = self.y.loc[X_tr.index]
            X_te = self.X[idx == t]
            if len(X_tr) == 0 or len(X_te) == 0:
                skipped.append((t, "empty train or test slice"))
                continue

            try:
                with tracker.training():
                    if _fit_accepts_validation(predictor) and len(val_dates) > 0:
                        X_val = self.X[idx.isin(val_dates)]
                        predictor.fit(X_tr, y_tr, X_val=X_val,
                                      y_val=self.y.loc[X_val.index])
                    else:
                        predictor.fit(X_tr, y_tr)

                tickers = [
                    str(tk) for tk in X_te.index.get_level_values("ticker")
                ]
                with tracker.inference():
                    if is_e2e:
                        weights = predictor.predict_weights(X_te)
                        weights = weights.groupby(level=0).first() \
                            if weights.index.has_duplicates else weights
                    else:
                        mu = predictor.predict(X_te)
                        mu.index = pd.Index(tickers, name="ticker")
                        # Covariance uses *realised* returns up to t — known
                        # at decision time, no purge needed.
                        hist = self.monthly_returns.loc[
                            self.monthly_returns.index <= t, tickers
                        ].tail(wf.train_window_months)
                        cov = cov_estimator.estimate(hist)
                        weights = optimiser.optimise(mu, cov, tickers)
            except LookAheadError:
                raise  # never swallow the leakage guard
            except Exception as exc:  # noqa: BLE001 — log + skip this date
                logger.exception("Rebalance %s failed for %s: %s", t, name, exc)
                skipped.append((t, repr(exc)))
                continue

            weights = weights / weights.sum() if weights.sum() > 0 else weights
            weights_rows[t] = weights

            # Realised next-month portfolio return (evaluation only — the
            # future never feeds back into any model input).
            realised = self.monthly_returns.loc[next_date].reindex(
                weights.index
            ).fillna(0.0)
            port_returns[next_date] = float((weights * realised).sum())

        tracker.stop_memory()

        gross = pd.Series(port_returns, dtype=float).sort_index()
        gross.name = name
        weights_history = pd.DataFrame.from_dict(
            weights_rows, orient="index"
        ).sort_index().fillna(0.0)

        from src.engine.metrics import apply_transaction_costs, turnover_series

        if len(weights_history) > 0:
            # Rebalance dates map 1:1 (in order) onto realised-return dates:
            # the cost of trading at t is paid out of the (t, t+1m] return.
            to = turnover_series(weights_history).set_axis(gross.index)
            net = apply_transaction_costs(
                gross, to, self.cfg.portfolio.transaction_cost_bps
            )
        else:
            to, net = pd.Series(dtype=float), gross

        return BacktestResult(
            strategy_name=name,
            gross_returns=gross,
            net_returns=net,
            weights_history=weights_history,
            turnover=to,
            resource_log=tracker.log.to_dict(),
            skipped_dates=skipped,
        )
