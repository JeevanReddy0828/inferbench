"""Turn a results JSON sweep into tables + plots.

Kept dependency-light (numpy + matplotlib) and torch-free so plots can be
regenerated from committed `results/*.json` on any machine.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402
from tabulate import tabulate  # noqa: E402

from ..config import LatencyResult, load_results


def results_table(results: list[LatencyResult]) -> str:
    """Markdown table of the sweep (one row per backend x batch_size)."""
    rows = []
    for r in sorted(results, key=lambda r: (r.workload, r.batch_size, r.backend)):
        status = "OOM" if r.oom else (r.error or "ok")
        rows.append(
            [
                r.workload,
                r.backend,
                r.batch_size,
                r.dtype,
                _fmt(r.p50_ms),
                _fmt(r.p90_ms),
                _fmt(r.p99_ms),
                _fmt(r.throughput_samples_s, 1),
                _fmt(r.peak_vram_mb, 0),
                status,
            ]
        )
    headers = [
        "workload", "backend", "bs", "dtype",
        "p50 ms", "p90 ms", "p99 ms", "thru smp/s", "VRAM MB", "status",
    ]
    return tabulate(rows, headers=headers, tablefmt="github")


def _fmt(x: float, nd: int = 3) -> str:
    return "N/A" if x != x else f"{x:.{nd}f}"  # x!=x -> NaN


def plot_latency_by_batch(results: list[LatencyResult], out_path: str, workload: str) -> None:
    """p50 latency vs batch size, one line per backend (lower is better)."""
    by_backend: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in results:
        if r.workload != workload or r.oom or r.error:
            continue
        by_backend[r.backend].append((r.batch_size, r.p50_ms))

    plt.figure(figsize=(7, 4.5))
    for backend, pts in sorted(by_backend.items()):
        pts.sort()
        xs, ys = zip(*pts)
        plt.plot(xs, ys, marker="o", label=backend)
    plt.xlabel("batch size")
    plt.ylabel("p50 latency (ms)")
    plt.title(f"{workload}: p50 latency vs batch size (RTX 3050 Ti, 4 GB)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_throughput_by_batch(results: list[LatencyResult], out_path: str, workload: str) -> None:
    """Throughput vs batch size, one line per backend (higher is better)."""
    by_backend: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in results:
        if r.workload != workload or r.oom or r.error:
            continue
        by_backend[r.backend].append((r.batch_size, r.throughput_samples_s))

    plt.figure(figsize=(7, 4.5))
    for backend, pts in sorted(by_backend.items()):
        pts.sort()
        xs, ys = zip(*pts)
        plt.plot(xs, ys, marker="s", label=backend)
    plt.xlabel("batch size")
    plt.ylabel("throughput (samples/s)")
    plt.title(f"{workload}: throughput vs batch size (RTX 3050 Ti, 4 GB)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def regenerate_from_json(json_path: str, out_prefix: str, workload: str = "resnet50") -> str:
    """Convenience: load a results JSON and (re)write its table + both plots."""
    results = load_results(json_path)
    plot_latency_by_batch(results, f"{out_prefix}_latency.png", workload)
    plot_throughput_by_batch(results, f"{out_prefix}_throughput.png", workload)
    return results_table(results)
