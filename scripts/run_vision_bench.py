#!/usr/bin/env python
"""End-to-end vision benchmark sweep.

Runs ResNet50 across {torch-eager, torch-compile, onnxruntime, tensorrt} over a
batch-size sweep, on the local GPU, and writes:
  * results/vision_resnet50.json   (raw + summarized, reproducible)
  * results/vision_resnet50_latency.png
  * results/vision_resnet50_throughput.png
and prints a markdown results table.

Usage (inside the WSL2 / Py3.11 env — see env/SETUP.md):
  python scripts/run_vision_bench.py
  python scripts/run_vision_bench.py --backends eager compile tensorrt --batch-sizes 1 4 8
  python scripts/run_vision_bench.py --dtype fp16 --timed-iters 200
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the repo root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inferbench.analysis import plots  # noqa: E402
from inferbench.backends import get_backend  # noqa: E402
from inferbench.bench.runner import time_callable  # noqa: E402
from inferbench.config import BenchConfig, save_results  # noqa: E402
from inferbench.workloads.vision import VisionWorkload  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ResNet50 inference benchmark sweep")
    p.add_argument("--model", default="resnet50", choices=["resnet50", "vit_b_16"])
    p.add_argument(
        "--backends",
        nargs="+",
        default=["eager", "compile", "onnx", "tensorrt"],
    )
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--warmup-iters", type=int, default=25)
    p.add_argument("--timed-iters", type=int, default=100)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    import torch

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a CUDA GPU.", file=sys.stderr)
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  torch {torch.__version__}")

    cfg = BenchConfig(
        warmup_iters=args.warmup_iters,
        timed_iters=args.timed_iters,
        batch_sizes=tuple(args.batch_sizes),
        dtype=args.dtype,
    )
    workload = VisionWorkload(name=args.model)
    base_model = workload.build_model()
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    all_results = []
    for backend_name in args.backends:
        backend_cls = get_backend(backend_name)
        print(f"\n=== backend: {backend_name} ({backend_cls.__name__}) ===")

        # Build once with the max batch as the export/profile example.
        example = workload.example_input(max(args.batch_sizes))
        backend = backend_cls(dtype=args.dtype)
        try:
            backend.build(base_model, example)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] build failed: {exc}")
            continue

        for bs in args.batch_sizes:
            inp = workload.example_input(bs).to(device="cuda", dtype=torch_dtype)
            res = time_callable(
                lambda inp=inp: backend.infer(inp),
                backend=backend.name,
                workload=args.model,
                batch_size=bs,
                dtype=args.dtype,
                config=cfg,
            )
            all_results.append(res)
            if not res.oom and not res.error:
                print(
                    f"  bs={bs:>2}  p50={res.p50_ms:7.3f}ms  "
                    f"p99={res.p99_ms:7.3f}ms  thru={res.throughput_samples_s:8.1f} smp/s  "
                    f"vram={res.peak_vram_mb:6.0f}MB"
                )
        backend.teardown()
        torch.cuda.empty_cache()

    # Persist + visualize.
    json_path = os.path.join(RESULTS_DIR, f"vision_{args.model}.json")
    save_results(all_results, json_path)
    prefix = os.path.join(RESULTS_DIR, f"vision_{args.model}")
    plots.plot_latency_by_batch(all_results, f"{prefix}_latency.png", args.model)
    plots.plot_throughput_by_batch(all_results, f"{prefix}_throughput.png", args.model)

    print("\n" + plots.results_table(all_results))
    print(f"\nwrote: {json_path}")
    print(f"wrote: {prefix}_latency.png")
    print(f"wrote: {prefix}_throughput.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
