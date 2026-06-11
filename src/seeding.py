"""Deterministic execution: one place that seeds every RNG we use.

torch and xgboost are imported lazily — both for determinism hygiene and
because importing them together in a shared process segfaults on macOS
(conflicting libomp runtimes). See src/engine/runner.py.
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Only seed torch if some model has ALREADY imported it in this process.
    # Importing it here unconditionally would (a) drag a multi-second torch
    # import into every worker, and (b) load torch alongside xgboost in the
    # same process — the exact macOS libomp clash the subprocess isolation
    # exists to prevent. Torch models also call torch.manual_seed in fit().
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    # xgboost takes its seed via the model constructor (random_state); nothing
    # global to set here.
