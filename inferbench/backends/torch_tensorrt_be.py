"""Torch-TensorRT backend.

A second route to TensorRT that stays inside PyTorch: instead of exporting to
ONNX and hand-driving the TRT builder (``tensorrt_be.py``), Torch-TensorRT
lowers the FX/Dynamo graph and hands convertible subgraphs to TRT while leaving
unsupported ops in Torch. Benchmarking both against each other is informative —
the ONNX path usually yields a fully-fused single engine, while Torch-TensorRT
trades some fusion for op coverage and an all-in-PyTorch workflow.

Registered in v2; the v1 sweep uses the ONNX-based ``tensorrt`` backend.
"""

from __future__ import annotations

from typing import Any

from .base import Backend
from .torch_eager import _TORCH_DTYPE


class TorchTensorRTBackend(Backend):
    name = "torch-tensorrt"

    def build(self, model: Any, example_inputs: Any) -> "TorchTensorRTBackend":
        import torch
        import torch_tensorrt

        dt = getattr(torch, _TORCH_DTYPE[self.dtype])
        model = model.to(device=self.device, dtype=dt).eval()
        example = example_inputs.to(device=self.device, dtype=dt)

        enabled = {torch.float16} if self.dtype == "fp16" else {torch.float32}
        self.model = torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=[example],
            enabled_precisions=enabled,
            # Keep the workspace under the 4 GB card's ceiling.
            workspace_size=3 << 30,
            truncate_double=True,
        )
        self._torch = torch
        return self

    def infer(self, inputs: Any) -> Any:
        torch = self._torch
        with torch.inference_mode():
            return self.model(inputs)
