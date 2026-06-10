"""Strategy-matrix runner with subprocess isolation.

Why subprocesses
----------------
On macOS, torch and xgboost each bundle their own libomp; loading both into
one process space segfaults. Each (predictor, cov, optimiser) combination
therefore runs in an isolated ``spawn`` subprocess that lazily imports only
the libraries it needs. A crash or exception in one combination is logged
with a full traceback to the run log, recorded as a skip in the summary, and
the rest of the matrix continues seamlessly.

Combination order is preserved everywhere with ``dict.fromkeys`` — never an
unordered ``set`` — so runs are reproducible byte-for-byte.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import pickle
import traceback
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import PipelineConfig
from src.seeding import set_global_seeds

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Combination:
    predictor: str
    cov: str
    optimiser: str

    @property
    def name(self) -> str:
        return f"{self.predictor}_{self.cov}_{self.optimiser}"


def experiment_combinations(experiment: str) -> list[Combination]:
    """Combination matrix per research question. Order is deterministic."""
    exp = experiment.lower()
    if exp == "rq1":  # predictor edge, risk model & allocator held fixed
        preds = list(dict.fromkeys(
            ["zero", "histmean", "ols", "ff3", "xgboost", "lstm"]
        ))
        return [Combination(p, "lw", "mvo") for p in preds]
    if exp == "rq2":  # risk architecture: zero forecast => pure min-variance
        covs = list(dict.fromkeys(["sample", "lw", "oas", "pca"]))
        return [Combination("zero", c, "mvo") for c in covs]
    if exp == "rq3":  # allocation mechanics under a fixed ML forecast
        opts = list(dict.fromkeys(["equal", "mvo", "riskparity", "cardinality"]))
        return [Combination("xgboost", "lw", o) for o in opts]
    if exp == "rq4":  # two-stage vs end-to-end
        return [
            Combination("xgboost", "lw", "mvo"),
            Combination("lstm", "lw", "mvo"),
            Combination("e2e", "direct", "direct"),
        ]
    if exp == "all":
        combos: list[Combination] = []
        for rq in ("rq1", "rq2", "rq3", "rq4"):
            combos.extend(experiment_combinations(rq))
        # dict.fromkeys dedupes across RQs while preserving first-seen order
        return list(dict.fromkeys(combos))
    raise ValueError(f"Unknown experiment: {experiment!r} (rq1..rq4 or all)")


# --------------------------------------------------------------------- worker
def _run_combination_worker(
    data_path: str, combo: Combination, config_dict: dict, out_path: str
) -> None:
    """Subprocess entry point: run one combination, write result or error."""
    out = Path(out_path)
    try:
        config = PipelineConfig.model_validate(config_dict)
        set_global_seeds(config.models.seed)

        with open(data_path, "rb") as fh:
            data = pickle.load(fh)

        from src.engine.walkforward import WalkForwardEngine
        from src.models import make_predictor

        engine = WalkForwardEngine(
            data["X"], data["y"], data["monthly_returns"], config
        )
        predictor = make_predictor(combo.predictor, config)
        if combo.predictor == "e2e":
            result = engine.run(predictor)
        else:
            from src.covariance import make_covariance
            from src.optimisers import make_optimiser

            result = engine.run(
                predictor,
                make_covariance(combo.cov),
                make_optimiser(
                    combo.optimiser,
                    max_weight=config.portfolio.max_weight,
                    cardinality_k=config.portfolio.cardinality_k,
                ),
            )
        with open(out, "wb") as fh:
            pickle.dump({"status": "ok", "result": result}, fh)
    except BaseException:  # noqa: BLE001 — full traceback must reach the parent
        with open(out, "wb") as fh:
            pickle.dump(
                {"status": "error", "traceback": traceback.format_exc()}, fh
            )
        raise


def run_matrix(
    X: pd.DataFrame,
    y: pd.Series,
    monthly_returns: pd.DataFrame,
    config: PipelineConfig,
    combinations: list[Combination],
    run_dir: str | Path,
) -> dict:
    """Run the full matrix; one isolated subprocess per combination.

    Returns ``{"results": {name: BacktestResult}, "skipped": {name: reason}}``
    and persists everything (plus a run log) under ``run_dir``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "run.log"
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)

    combinations = list(dict.fromkeys(combinations))  # ordered dedupe
    data_path = run_dir / "dataset.pkl"
    with open(data_path, "wb") as f:
        pickle.dump({"X": X, "y": y, "monthly_returns": monthly_returns}, f)

    results: dict[str, object] = {}
    skipped: dict[str, str] = {}
    ctx = mp.get_context("spawn")  # spawn: clean import state per combo

    for combo in combinations:
        out_path = run_dir / f"combo_{combo.name}.pkl"
        logger.info("Running combination %s ...", combo.name)
        try:
            if config.runner.use_subprocess:
                proc = ctx.Process(
                    target=_run_combination_worker,
                    args=(str(data_path), combo, config.model_dump(),
                          str(out_path)),
                )
                proc.start()
                proc.join(timeout=config.runner.subprocess_timeout_seconds)
                if proc.is_alive():
                    proc.terminate()
                    proc.join()
                    raise TimeoutError(
                        f"Combination {combo.name} exceeded "
                        f"{config.runner.subprocess_timeout_seconds}s"
                    )
                if not out_path.exists():
                    raise RuntimeError(
                        f"Worker died without output (exit code "
                        f"{proc.exitcode}) — see {log_file}"
                    )
            else:
                _run_combination_worker(
                    str(data_path), combo, config.model_dump(), str(out_path)
                )

            with open(out_path, "rb") as f:
                payload = pickle.load(f)
            if payload["status"] == "ok":
                results[combo.name] = payload["result"]
                logger.info("Combination %s finished.", combo.name)
            else:
                skipped[combo.name] = payload["traceback"]
                logger.error(
                    "Combination %s FAILED — recorded as skip.\n%s",
                    combo.name, payload["traceback"],
                )
        except Exception as exc:  # noqa: BLE001 — matrix must keep going
            skipped[combo.name] = f"{exc}\n{traceback.format_exc()}"
            logger.exception("Combination %s crashed; continuing matrix.",
                             combo.name)

    summary = {"results": results, "skipped": skipped}
    with open(run_dir / "results.pkl", "wb") as f:
        pickle.dump(summary, f)
    logging.getLogger().removeHandler(fh)
    return summary
