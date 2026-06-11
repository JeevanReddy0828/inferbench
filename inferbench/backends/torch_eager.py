"""PyTorch eager backend — the baseline every other backend is measured against."""

from __future__ import annotations

from typing import Any

from .base import Backend

_TORCH_DTYPE = {"fp32": "float32", "fp16": "float16"}


class TorchEagerBackend(Backend):
    name = "torch-eager"

    def build(self, model: Any, example_inputs: Any) -> "TorchEagerBackend":
        import torch

        dt = getattr(torch, _TORCH_DTYPE[self.dtype])
        self.model = model.to(device=self.device, dtype=dt).eval()
        self._torch = torch
        self._autocast = self.dtype == "fp16"
        return self

    def infer(self, inputs: Any) -> Any:
        torch = self._torch
        with torch.inference_mode():
            # Weights are already cast; inputs are cast by the caller. autocast
            # would double-handle, so we rely on explicit dtype instead.
            return self.model(inputs)
