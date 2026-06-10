"""Deterministic execution: one place that seeds every RNG we use.

torch and xgboost are imported lazily — both for determinism hygiene and
because importing them together in a shared process segfaults on macOS
(conflicting libomp runtimes). See src/engine/runner.py.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch only if already importable in this worker
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
    # xgboost takes its seed via the model constructor (random_state); nothing
    # global to set here.
