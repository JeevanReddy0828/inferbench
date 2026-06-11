"""Fused RMSNorm in Triton.

RMSNorm(x) = x / sqrt(mean(x^2, dim=-1) + eps) * weight

This is *the* normalization used by Llama/Qwen-family transformers, so it's the
natural kernel to hand-write for the LLM track (v3) while also being a clean,
self-contained DSL demo for v1. One Triton program handles one row: it loads the
row, computes the mean-square reduction **in fp32** (numerically important when
inputs are fp16), normalizes, scales by the learned weight, and writes back.

Fusing the reduction + normalization + scale into one kernel avoids the 3 extra
HBM round-trips an eager `x.pow(2).mean(); x*rstd; *w` sequence pays — RMSNorm is
memory-bound, so kernel count, not FLOPs, is what dominates its latency.
"""

from __future__ import annotations

from typing import Any

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in (1024, 2048, 4096, 8192)
        for nw in (4, 8, 16)
    ],
    key=["n_cols"],
)
@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    x_row_stride,
    y_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * x_row_stride
    y_row = y_ptr + row * y_row_stride

    # --- pass 1: sum of squares (fp32 accumulator), tiled over the feature dim ---
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        acc += vals * vals
    mean_sq = tl.sum(acc, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    # --- pass 2: normalize, scale by weight, store ---
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = vals * rstd * w
        tl.store(y_row + cols, out.to(y_row.dtype.element_ty), mask=mask)


def rmsnorm_triton(x: Any, weight: Any, eps: float = 1e-6) -> Any:
    """Fused RMSNorm over the last dim. ``x`` is (..., n_cols); ``weight`` is (n_cols,)."""
    import torch

    assert x.is_cuda and weight.is_cuda, "rmsnorm_triton requires CUDA tensors"
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1]).contiguous()
    n_rows, n_cols = x2d.shape
    y = torch.empty_like(x2d)

    _rmsnorm_fwd_kernel[(n_rows,)](
        x2d,
        weight,
        y,
        x2d.stride(0),
        y.stride(0),
        n_cols,
        eps,
    )
    return y.reshape(orig_shape)


def rmsnorm_torch_ref(x: Any, weight: Any, eps: float = 1e-6) -> Any:
    """Reference RMSNorm in eager PyTorch (fp32 reduction), for correctness checks."""
    import torch

    dtype = x.dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps)
    return (out.to(dtype)) * weight
