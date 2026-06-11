"""Numerical-correctness tests for the Triton RMSNorm kernel.

Marked `gpu` — deselect with `-m 'not gpu'` on a machine without CUDA/triton.
A kernel that is fast but wrong is worthless, so correctness is asserted at
several shapes (including non-power-of-two feature dims that exercise masking)
and both dtypes before any perf claim is made.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from inferbench.kernels.triton import rmsnorm_torch_ref, rmsnorm_triton  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("rows", [1, 17, 512])
@pytest.mark.parametrize("dim", [128, 333, 4096])  # 333 forces the masked tail path
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rmsnorm_matches_reference(rows, dim, dtype):
    torch.manual_seed(0)
    x = torch.randn(rows, dim, device="cuda", dtype=dtype)
    w = torch.randn(dim, device="cuda", dtype=dtype)

    y = rmsnorm_triton(x, w)
    ref = rmsnorm_torch_ref(x, w)

    atol, rtol = (1e-2, 1e-2) if dtype == torch.float16 else (1e-4, 1e-4)
    torch.testing.assert_close(y, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rmsnorm_preserves_shape_3d():
    x = torch.randn(2, 8, 4096, device="cuda", dtype=torch.float16)
    w = torch.randn(4096, device="cuda", dtype=torch.float16)
    y = rmsnorm_triton(x, w)
    assert y.shape == x.shape
