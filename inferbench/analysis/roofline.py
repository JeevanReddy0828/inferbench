"""Roofline analysis: is a kernel memory-bound or compute-bound, and how close
is it to the hardware ceiling?

The roofline model bounds attainable throughput as::

    attainable_flops(AI) = min(peak_flops, peak_bandwidth * AI)

where AI (arithmetic intensity) = FLOPs / bytes-moved. The crossover ("ridge
point") at AI = peak_flops / peak_bandwidth separates the memory-bound regime
(left, throughput limited by HBM) from the compute-bound regime (right, limited
by the math units). Plotting a measured kernel against this tells you *which
ceiling to optimize against* — e.g. RMSNorm/elementwise ops live far left and
are won by reducing memory traffic (fusion), not by faster math.

Pure stdlib + numpy/matplotlib so it runs without torch/CUDA; device peaks are
parameters (the RTX 3050 Ti defaults are nominal — confirm with a microbenchmark
before quoting % -of-peak).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class DevicePeaks:
    """Hardware ceilings. Defaults are *nominal* RTX 3050 Ti Laptop (Ampere, sm_86).

    These are spec-sheet estimates, not measured; override with values from a
    microbenchmark (e.g. a large GEMM for FLOPs, a stream copy for bandwidth)
    before reporting percent-of-peak.
    """

    name: str = "RTX 3050 Ti Laptop (sm_86)"
    mem_bandwidth_gb_s: float = 192.0  # GDDR6, 128-bit @ 12 Gbps
    peak_tflops: dict[str, float] = field(
        default_factory=lambda: {
            "fp32": 7.6,        # 2 * 2560 CUDA cores * ~1.485 GHz
            "fp16_tensor": 36.0,  # gen-3 tensor cores, fp16/fp32-accumulate (approx)
        }
    )

    def ridge_ai(self, dtype: str = "fp16_tensor") -> float:
        """Arithmetic intensity (FLOP/byte) where the roofline bends."""
        flops = self.peak_tflops[dtype] * 1e12
        bw = self.mem_bandwidth_gb_s * 1e9
        return flops / bw


@dataclass
class RooflinePoint:
    name: str
    flops: float          # total FLOPs of the op
    bytes_moved: float    # total bytes read+written from/to HBM
    measured_tflops: float | None = None  # flops / latency, if measured

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.bytes_moved if self.bytes_moved else float("nan")

    @classmethod
    def from_latency(cls, name: str, flops: float, bytes_moved: float, latency_ms: float) -> "RooflinePoint":
        tflops = flops / (latency_ms * 1e-3) / 1e12 if latency_ms > 0 else None
        return cls(name=name, flops=flops, bytes_moved=bytes_moved, measured_tflops=tflops)


def attainable_tflops(ai: float, peaks: DevicePeaks, dtype: str = "fp16_tensor") -> float:
    """Roofline ceiling (TFLOP/s) at a given arithmetic intensity."""
    peak = peaks.peak_tflops[dtype]
    mem_bound = (peaks.mem_bandwidth_gb_s * ai) / 1e3  # GB/s * FLOP/byte -> GFLOP/s -> /1e3 TFLOP/s
    return min(peak, mem_bound)


def classify(point: RooflinePoint, peaks: DevicePeaks, dtype: str = "fp16_tensor") -> str:
    """Return 'memory-bound' or 'compute-bound' for the op."""
    return "memory-bound" if point.arithmetic_intensity < peaks.ridge_ai(dtype) else "compute-bound"


def utilization(point: RooflinePoint, peaks: DevicePeaks, dtype: str = "fp16_tensor") -> float:
    """Measured TFLOP/s as a fraction of the roofline ceiling at this op's AI."""
    if point.measured_tflops is None:
        return float("nan")
    ceiling = attainable_tflops(point.arithmetic_intensity, peaks, dtype)
    return point.measured_tflops / ceiling if ceiling > 0 else float("nan")


def plot_roofline(
    points: list[RooflinePoint],
    out_path: str,
    peaks: DevicePeaks | None = None,
    dtype: str = "fp16_tensor",
) -> None:
    """Log-log roofline with the memory & compute ceilings and measured points."""
    import numpy as np

    peaks = peaks or DevicePeaks()
    ais = np.logspace(-2, 3, 200)
    ceil = [attainable_tflops(ai, peaks, dtype) for ai in ais]

    plt.figure(figsize=(7, 5))
    plt.loglog(ais, ceil, "k-", linewidth=2, label=f"roofline ({dtype})")
    plt.axvline(
        peaks.ridge_ai(dtype), color="gray", linestyle="--", alpha=0.6,
        label=f"ridge AI={peaks.ridge_ai(dtype):.1f}",
    )
    for p in points:
        y = p.measured_tflops if p.measured_tflops is not None else attainable_tflops(
            p.arithmetic_intensity, peaks, dtype
        )
        plt.loglog(p.arithmetic_intensity, y, "o", markersize=9)
        plt.annotate(p.name, (p.arithmetic_intensity, y), textcoords="offset points", xytext=(8, 4))

    plt.xlabel("arithmetic intensity (FLOP / byte)")
    plt.ylabel("throughput (TFLOP/s)")
    plt.title(f"Roofline — {peaks.name}")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
