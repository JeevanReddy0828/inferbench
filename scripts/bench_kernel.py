#!/usr/bin/env python
"""Benchmark the fused Triton RMSNorm kernel vs eager PyTorch and torch.compile.

Reports per-shape latency with ``triton.testing.do_bench`` (which handles its own
warmup + percentile windowing) and the achieved HBM bandwidth — RMSNorm is
memory-bound, so bandwidth utilization is the metric that actually explains the
speedup.

Usage (WSL2 / Py3.11 env):
  python scripts/bench_kernel.py
  python scripts/bench_kernel.py --rows 4096 --dims 4096 8192
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Triton RMSNorm microbenchmark")
    p.add_argument("--rows", type=int, default=4096, help="number of rows (tokens)")
    p.add_argument("--dims", nargs="+", type=int, default=[2048, 4096, 8192])
    p.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    import triton

    from inferbench.kernels.triton import rmsnorm_torch_ref, rmsnorm_triton

    if not torch.cuda.is_available():
        print("ERROR: CUDA required.", file=sys.stderr)
        return 1

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  RMSNorm rows={args.rows} dtype={args.dtype}\n")
    print(f"{'dim':>8} {'eager ms':>10} {'compile ms':>12} {'triton ms':>11} "
          f"{'speedup':>8} {'triton GB/s':>12}")

    compiled_ref = torch.compile(rmsnorm_torch_ref)
    for dim in args.dims:
        x = torch.randn(args.rows, dim, device="cuda", dtype=dtype)
        w = torch.randn(dim, device="cuda", dtype=dtype)

        # Correctness gate before timing — never benchmark a wrong kernel.
        y_triton = rmsnorm_triton(x, w)
        y_ref = rmsnorm_torch_ref(x, w)
        torch.testing.assert_close(y_triton, y_ref, atol=1e-2, rtol=1e-2)

        t_eager = triton.testing.do_bench(lambda: rmsnorm_torch_ref(x, w))
        t_compile = triton.testing.do_bench(lambda: compiled_ref(x, w))
        t_triton = triton.testing.do_bench(lambda: rmsnorm_triton(x, w))

        # bytes moved: read x + read w (broadcast, count once per row negligible) + write y
        bytes_moved = 2 * x.numel() * x.element_size()
        gbps = bytes_moved / (t_triton * 1e-3) / 1e9
        speedup = t_eager / t_triton
        print(f"{dim:>8} {t_eager:>10.4f} {t_compile:>12.4f} {t_triton:>11.4f} "
              f"{speedup:>7.2f}x {gbps:>11.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
