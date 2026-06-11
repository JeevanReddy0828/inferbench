"""TorchInductor backend via ``torch.compile``.

This is the "graph compiler" entry in the JD map. Inductor traces the model
(TorchDynamo), fuses pointwise ops, and codegens Triton kernels for the GPU.
The first call triggers compilation, which is why the harness's warmup phase
is essential here — without it the first timed iter would include minutes of
compile time.
"""

from __future__ import annotations

from typing import Any

from .base import Backend
from .torch_eager import _TORCH_DTYPE


class TorchCompileBackend(Backend):
    name = "torch-compile"

    def __init__(self, dtype: str = "fp16", device: str = "cuda", mode: str = "max-autotune") -> None:
        super().__init__(dtype=dtype, device=device)
        self.mode = mode

    def build(self, model: Any, example_inputs: Any) -> "TorchCompileBackend":
        import torch

        dt = getattr(torch, _TORCH_DTYPE[self.dtype])
        base = model.to(device=self.device, dtype=dt).eval()
        # mode="max-autotune" lets Inductor benchmark several Triton/cutlass
        # templates per matmul/conv and pick the fastest — the apples-to-apples
        # "best compiler-found kernel" comparison point against TensorRT.
        self.model = torch.compile(base, mode=self.mode, fullgraph=False)
        self._torch = torch
        return self

    def infer(self, inputs: Any) -> Any:
        torch = self._torch
        with torch.inference_mode():
            return self.model(inputs)
