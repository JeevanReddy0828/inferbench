"""Correctness tests for the custom CUDA fused bias+GELU op.

Marked `gpu` — requires CUDA + a working nvcc toolchain (the extension
JIT-compiles on first import). Deselect with `-m 'not gpu'`.
"""

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("rows", [1, 32, 512])
@pytest.mark.parametrize("cols", [128, 333, 4096])  # 333 exercises non-aligned tail
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_bias_gelu_matches_reference(rows, cols, dtype):
    from inferbench.kernels.cuda import fused_bias_gelu, fused_bias_gelu_torch_ref

    torch.manual_seed(0)
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    bias = torch.randn(cols, device="cuda", dtype=dtype)

    y = fused_bias_gelu(x, bias)
    ref = fused_bias_gelu_torch_ref(x, bias)

    atol, rtol = (2e-2, 2e-2) if dtype == torch.float16 else (1e-4, 1e-4)
    torch.testing.assert_close(y, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fused_bias_gelu_shape_mismatch_raises():
    from inferbench.kernels.cuda import fused_bias_gelu

    x = torch.randn(4, 128, device="cuda")
    bias = torch.randn(64, device="cuda")  # wrong size
    with pytest.raises(RuntimeError):
        fused_bias_gelu(x, bias)
