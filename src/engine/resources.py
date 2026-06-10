"""Resource-aware efficiency tracking: time and peak memory per strategy."""

from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class ResourceLog:
    training_time_seconds: float = 0.0
    inference_latency_ms_total: float = 0.0
    n_inferences: int = 0
    peak_memory_mb: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def inference_latency_ms(self) -> float:
        """Mean per-rebalance inference latency."""
        if self.n_inferences == 0:
            return 0.0
        return self.inference_latency_ms_total / self.n_inferences

    def to_dict(self) -> dict:
        return {
            "training_time_seconds": self.training_time_seconds,
            "inference_latency_ms": self.inference_latency_ms,
            "peak_memory_mb": self.peak_memory_mb,
        }


class ResourceTracker:
    def __init__(self) -> None:
        self.log = ResourceLog()
        self._mem_started = False

    def start_memory(self) -> None:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._mem_started = True

    def snapshot_memory(self) -> None:
        if tracemalloc.is_tracing():
            _, peak = tracemalloc.get_traced_memory()
            self.log.peak_memory_mb = max(self.log.peak_memory_mb, peak / 1e6)

    def stop_memory(self) -> None:
        self.snapshot_memory()
        if self._mem_started and tracemalloc.is_tracing():
            tracemalloc.stop()

    @contextmanager
    def training(self):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.log.training_time_seconds += time.perf_counter() - t0
            self.snapshot_memory()

    @contextmanager
    def inference(self):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.log.inference_latency_ms_total += (
                time.perf_counter() - t0
            ) * 1e3
            self.log.n_inferences += 1
