"""Custom CUDA C++ fused bias+GELU op (v2).

The kernel is JIT-compiled on first use via ``torch.utils.cpp_extension.load``
(no manual nvcc flags, and it builds against the *exact* CUDA the torch wheel
was built with — see the version-pinning note in env/SETUP.md). An ahead-of-time
build is also available via ``setup.py`` if you prefer a prebuilt .so.
"""

from __future__ import annotations

import functools
import os
from typing import Any

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@functools.lru_cache(maxsize=1)
def _load_ext():
    """Compile + load the extension once, caching the module."""
    from torch.utils.cpp_extension import load

    return load(
        name="inferbench_fused_bias_gelu",
        sources=[os.path.join(_THIS_DIR, "fused_bias_gelu.cu")],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def fused_bias_gelu(x: Any, bias: Any) -> Any:
    """y = gelu(x + bias) via the fused CUDA kernel. ``bias`` broadcasts over rows."""
    ext = _load_ext()
    return ext.fused_bias_gelu(x, bias)


def fused_bias_gelu_torch_ref(x: Any, bias: Any) -> Any:
    """Reference: unfused bias-add + tanh-approx GELU in eager PyTorch (fp32 math)."""
    import torch
    import torch.nn.functional as F

    dtype = x.dtype
    v = (x.to(torch.float32) + bias.to(torch.float32))
    return F.gelu(v, approximate="tanh").to(dtype)
