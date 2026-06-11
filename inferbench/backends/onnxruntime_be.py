"""ONNX Runtime backend (CUDA Execution Provider).

Fair-timing notes:
  * The model is exported once to ONNX in build().
  * Inference uses ORT **IO binding** against the input tensor's device pointer
    so we never pay a host->device copy per iter. Timing a backend that secretly
    does a H2D memcpy each call against one that doesn't is the single most common
    way benchmark tables lie — so we bind on-device explicitly.
  * Output is bound to a preallocated CUDA tensor for the same reason.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from .base import Backend
from .torch_eager import _TORCH_DTYPE


class OnnxRuntimeBackend(Backend):
    name = "onnxruntime"

    def __init__(self, dtype: str = "fp16", device: str = "cuda", onnx_path: str | None = None) -> None:
        super().__init__(dtype=dtype, device=device)
        self.onnx_path = onnx_path

    def build(self, model: Any, example_inputs: Any) -> "OnnxRuntimeBackend":
        import numpy as np
        import onnxruntime as ort
        import torch

        self._torch = torch
        self._np = np
        dt = getattr(torch, _TORCH_DTYPE[self.dtype])
        model = model.to(device=self.device, dtype=dt).eval()
        example = example_inputs.to(device=self.device, dtype=dt)

        if self.onnx_path is None:
            tmpdir = tempfile.mkdtemp(prefix="inferbench_onnx_")
            self.onnx_path = os.path.join(tmpdir, f"{self.name}_{self.dtype}.onnx")

        # Dynamic batch so one engine/session serves the whole batch sweep.
        torch.onnx.export(
            model,
            example,
            self.onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
        )

        providers = [("CUDAExecutionProvider", {"device_id": 0})]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.onnx_path, sess_options=so, providers=providers)
        self._ort = ort
        self._np_dtype = np.float16 if self.dtype == "fp16" else np.float32
        self._out_name = self.session.get_outputs()[0].name
        self._in_name = self.session.get_inputs()[0].name
        return self

    def infer(self, inputs: Any) -> Any:
        torch = self._torch
        ort = self._ort
        inputs = inputs.contiguous()
        binding = self.session.io_binding()
        binding.bind_input(
            name=self._in_name,
            device_type="cuda",
            device_id=0,
            element_type=self._np_dtype,
            shape=tuple(inputs.shape),
            buffer_ptr=inputs.data_ptr(),
        )
        # Let ORT allocate the output on device; we only need it to exist for timing.
        binding.bind_output(self._out_name, device_type="cuda", device_id=0)
        self.session.run_with_iobinding(binding)
        return binding
