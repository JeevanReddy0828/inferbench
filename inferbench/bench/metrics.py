"""Latency/throughput reductions.

Pure-Python + numpy, intentionally free of any torch/CUDA imports so the
measurement math can be unit-tested on any machine (this is the part that
encodes the *methodology*, so it is the part that must be provably correct).
"""

from __future__ import annotations

import math

import numpy as np


def percentile(latencies_ms: list[float], q: float) -> float:
    """q-th percentile (q in [0, 100]) using linear interpolation.

    Uses numpy's default ("linear") method so results match the wider
    benchmarking ecosystem (trtexec, torch.utils.benchmark report similarly).
    """
    if not latencies_ms:
        return float("nan")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"percentile q must be in [0, 100], got {q}")
    return float(np.percentile(np.asarray(latencies_ms, dtype=np.float64), q))


def summarize(latencies_ms: list[float], batch_size: int) -> dict[str, float]:
    """Reduce raw per-iter latencies to the headline metrics.

    Throughput is derived from the **median** latency rather than the mean:
    the median is robust to the occasional scheduler/clock-spike outlier that
    laptop GPUs (thermal throttling on a 3050 Ti) produce, which would
    otherwise deflate a mean-based throughput.
    """
    if not latencies_ms:
        return {
            "mean_ms": float("nan"),
            "p50_ms": float("nan"),
            "p90_ms": float("nan"),
            "p99_ms": float("nan"),
            "std_ms": float("nan"),
            "throughput_samples_s": float("nan"),
        }

    arr = np.asarray(latencies_ms, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    throughput = (batch_size * 1000.0 / p50) if p50 > 0 else float("nan")
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": p50,
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        # Sample std (ddof=1); falls back to 0.0 for a single sample.
        "std_ms": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "throughput_samples_s": throughput,
    }


def speedup(baseline_ms: float, candidate_ms: float) -> float:
    """Speedup factor of candidate vs baseline (>1 means candidate is faster)."""
    if candidate_ms <= 0 or math.isnan(candidate_ms) or math.isnan(baseline_ms):
        return float("nan")
    return baseline_ms / candidate_ms


def coefficient_of_variation(latencies_ms: list[float]) -> float:
    """std/mean — a measurement-stability check.

    A high CoV (>~0.05 on a steady-state forward pass) is a signal the run was
    noisy (throttling, background load, insufficient warmup) and the numbers
    should not be trusted. The harness logs this so bad runs are visible.
    """
    if not latencies_ms:
        return float("nan")
    arr = np.asarray(latencies_ms, dtype=np.float64)
    mean = arr.mean()
    if mean == 0:
        return float("nan")
    std = arr.std(ddof=1) if arr.size > 1 else 0.0
    return float(std / mean)
