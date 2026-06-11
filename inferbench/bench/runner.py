"""GPU timing harness.

Why CUDA events and not ``time.perf_counter()``: GPU kernel launches are
asynchronous. Wall-clock timing around a launch measures *queue* time, not
*execution* time, unless you synchronize — and a naive ``synchronize()`` per
iter serializes the pipeline and inflates latency. ``torch.cuda.Event`` records
timestamps *on the CUDA stream*, giving true device-side per-iter latency with a
single synchronize at the end of the timed window.

This module is the project's core methodological claim, so the discipline is
explicit and commented:

  1. Fixed seed + deterministic input (timing must not depend on data values).
  2. Warmup iters to absorb lazy init, cuDNN/cuBLAS autotuning, torch.compile
     graph capture, and CUDA context/clock spin-up.
  3. ``reset_peak_memory_stats`` so peak-VRAM is for the timed region only.
  4. Per-iter CUDA-event pairs; one synchronize after the loop.
  5. A coefficient-of-variation check that flags noisy/throttled runs.
"""

from __future__ import annotations

from typing import Callable

from ..config import BenchConfig, LatencyResult
from . import metrics


def time_callable(
    fn: Callable[[], object],
    *,
    backend: str,
    workload: str,
    batch_size: int,
    dtype: str,
    config: BenchConfig,
) -> LatencyResult:
    """Time a zero-arg callable that performs one forward pass.

    ``fn`` must enqueue the work (and may return its output); it must not
    synchronize internally — the harness owns synchronization. Returns a fully
    populated :class:`LatencyResult`, including an OOM/error record rather than
    raising, so a sweep can continue and report the gap.
    """
    import torch  # local import: keep module importable without the DL stack

    device = config.device
    result = LatencyResult(
        backend=backend,
        workload=workload,
        batch_size=batch_size,
        dtype=dtype,
    )

    try:
        torch.manual_seed(config.seed)

        # --- Warmup: stabilize autotuning / compilation / clocks ---
        for _ in range(config.warmup_iters):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # --- Timed window: one event pair per iter, single sync at the end ---
        if device == "cuda":
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(config.timed_iters)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(config.timed_iters)]
            for i in range(config.timed_iters):
                starts[i].record()
                fn()
                ends[i].record()
            torch.cuda.synchronize()
            latencies = [starts[i].elapsed_time(ends[i]) for i in range(config.timed_iters)]
            result.peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:  # CPU fallback (used only for smoke tests / no-GPU CI)
            import time

            latencies = []
            for _ in range(config.timed_iters):
                t0 = time.perf_counter()
                fn()
                latencies.append((time.perf_counter() - t0) * 1000.0)

        result.latencies_ms = latencies
        summary = metrics.summarize(latencies, batch_size)
        result.mean_ms = summary["mean_ms"]
        result.p50_ms = summary["p50_ms"]
        result.p90_ms = summary["p90_ms"]
        result.p99_ms = summary["p99_ms"]
        result.std_ms = summary["std_ms"]
        result.throughput_samples_s = summary["throughput_samples_s"]

        cov = metrics.coefficient_of_variation(latencies)
        if cov == cov and cov > 0.05:  # not-NaN and noisy
            print(
                f"  [warn] {backend}/{workload} bs={batch_size}: "
                f"noisy run (CoV={cov:.3f}>0.05) — possible throttling/background load"
            )

    except RuntimeError as exc:  # noqa: BLE001 — we want to record, not crash the sweep
        msg = str(exc)
        if "out of memory" in msg.lower():
            result.oom = True
            result.error = "CUDA out of memory"
            print(f"  [oom] {backend}/{workload} bs={batch_size}: skipped (OOM)")
            if device == "cuda":
                torch.cuda.empty_cache()
            if not config.skip_on_oom:
                raise
        else:
            result.error = msg
            print(f"  [error] {backend}/{workload} bs={batch_size}: {msg}")
            raise

    return result
