"""Configuration + result dataclasses shared across the toolkit.

These are deliberately framework-agnostic (pure stdlib) so they can be imported,
serialized, and unit-tested without torch / CUDA present.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BenchConfig:
    """Knobs that define *how* a measurement is taken.

    The defaults encode the measurement discipline this project argues for:
    enough warmup to JIT/autotune-stabilize, enough timed iters for stable
    percentiles, and an explicit device synchronization between phases.
    """

    warmup_iters: int = 25
    timed_iters: int = 100
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8)
    device: str = "cuda"
    # "fp32" | "fp16"; backends may override (e.g. TensorRT FP16 engine).
    dtype: str = "fp16"
    # Fail loudly instead of silently OOM-skipping unless explicitly allowed.
    skip_on_oom: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.warmup_iters < 0 or self.timed_iters <= 0:
            raise ValueError("warmup_iters must be >=0 and timed_iters must be >0")
        if not self.batch_sizes:
            raise ValueError("batch_sizes must be non-empty")


@dataclass
class LatencyResult:
    """Outcome of timing one (backend, workload, batch_size) point.

    Latencies are per-iteration wall times of a single forward call, in
    milliseconds, measured with CUDA events. `oom` / `error` let a sweep
    record a missing point instead of silently dropping it.
    """

    backend: str
    workload: str
    batch_size: int
    dtype: str
    # Raw per-iter latencies in ms (kept so percentiles can be recomputed/audited).
    latencies_ms: list[float] = field(default_factory=list)
    mean_ms: float = float("nan")
    p50_ms: float = float("nan")
    p90_ms: float = float("nan")
    p99_ms: float = float("nan")
    std_ms: float = float("nan")
    # Throughput in samples/sec computed from the median latency.
    throughput_samples_s: float = float("nan")
    peak_vram_mb: float = float("nan")
    oom: bool = False
    error: str | None = None

    def to_row(self) -> dict[str, Any]:
        """Compact dict for tabulation (drops the raw latency array)."""
        d = asdict(self)
        d.pop("latencies_ms", None)
        return d


def save_results(results: list[LatencyResult], path: str) -> None:
    """Persist a sweep as JSON (raw latencies included for reproducibility)."""
    payload = [asdict(r) for r in results]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_results(path: str) -> list[LatencyResult]:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return [LatencyResult(**row) for row in payload]
