"""TensorRT backend (TRT 10.x API).

Pipeline: torch model -> ONNX -> TensorRT engine (FP16) -> execute on-device.

Design choices that matter for a *fair, fast* benchmark:
  * Engine is built once and cached to disk (keyed by model+dtype+shape range);
    rebuilds are the slow part and must never land inside the timed window.
  * A single optimization profile spans the batch sweep (min=1, opt=4, max=8)
    so one engine serves every batch size — matching how you'd actually deploy.
  * Inference binds the input/output **device pointers** (torch tensors) via
    ``set_tensor_address`` and runs ``execute_async_v3`` on a dedicated stream.
    No host<->device copies in the hot path.

INT8 PTQ is added in v2 via ``inferbench.quant.calibrate`` (an IInt8Calibrator
passed into ``build``); v1 ships the FP16 path only.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from typing import Any

from .base import Backend
from .torch_eager import _TORCH_DTYPE


class TensorRTBackend(Backend):
    name = "tensorrt"

    def __init__(
        self,
        dtype: str = "fp16",
        device: str = "cuda",
        min_bs: int = 1,
        opt_bs: int = 4,
        max_bs: int = 8,
        cache_dir: str | None = None,
        calibrator: Any = None,
    ) -> None:
        super().__init__(dtype=dtype, device=device)
        self.min_bs, self.opt_bs, self.max_bs = min_bs, opt_bs, max_bs
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "inferbench_trt")
        self.calibrator = calibrator  # v2: INT8 PTQ

    # ---- engine construction -------------------------------------------------
    def _engine_path(self, example_inputs: Any) -> str:
        os.makedirs(self.cache_dir, exist_ok=True)
        precision = "int8" if self.calibrator is not None else self.dtype
        key = f"{tuple(example_inputs.shape)}-{precision}-{self.min_bs}-{self.opt_bs}-{self.max_bs}"
        h = hashlib.sha1(key.encode()).hexdigest()[:12]
        return os.path.join(self.cache_dir, f"engine_{h}.trt")

    def build(self, model: Any, example_inputs: Any) -> "TensorRTBackend":
        import tensorrt as trt
        import torch

        self._torch = torch
        self._trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)

        dt = getattr(torch, _TORCH_DTYPE[self.dtype])
        model = model.to(device=self.device, dtype=dt).eval()
        example = example_inputs.to(device=self.device, dtype=dt)
        c, h_, w = example.shape[1:]

        engine_path = self._engine_path(example)
        if os.path.exists(engine_path):
            with open(engine_path, "rb") as fh:
                serialized = fh.read()
        else:
            serialized = self._build_serialized(model, example, c, h_, w)
            with open(engine_path, "wb") as fh:
                fh.write(serialized)

        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(serialized)
        self.context = self.engine.create_execution_context()
        self._in_name = self.engine.get_tensor_name(0)
        self._out_name = self.engine.get_tensor_name(1)
        self.stream = torch.cuda.Stream()
        self._out_buffers: dict[int, Any] = {}  # batch_size -> preallocated output tensor
        return self

    def _build_serialized(self, model: Any, example: Any, c: int, h_: int, w: int) -> bytes:
        trt, torch = self._trt, self._torch

        # ONNX export with a dynamic batch dim.
        onnx_path = os.path.join(self.cache_dir, "tmp_model.onnx")
        torch.onnx.export(
            model,
            example,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
        )

        builder = trt.Builder(self.logger)
        network = builder.create_network()  # TRT 10: explicit batch is implicit
        parser = trt.OnnxParser(network, self.logger)
        with open(onnx_path, "rb") as fh:
            if not parser.parse(fh.read()):
                errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                raise RuntimeError(f"TensorRT ONNX parse failed:\n{errs}")

        config = builder.create_builder_config()
        # 3 GB workspace ceiling — leaves headroom on a 4 GB card.
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)
        if self.dtype == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        if self.calibrator is not None:  # v2 INT8 PTQ path
            config.set_flag(trt.BuilderFlag.INT8)
            # FP16 fallback: layers TRT can't profitably run in INT8 stay in FP16
            # rather than dropping to FP32 — best accuracy/latency on Ampere.
            config.set_flag(trt.BuilderFlag.FP16)
            config.int8_calibrator = self.calibrator

        profile = builder.create_optimization_profile()
        profile.set_shape(
            "input",
            (self.min_bs, c, h_, w),
            (self.opt_bs, c, h_, w),
            (self.max_bs, c, h_, w),
        )
        config.add_optimization_profile(profile)
        if self.calibrator is not None:
            # INT8 calibration runs at a fixed shape; calibrate at the opt batch.
            calib_profile = builder.create_optimization_profile()
            calib_profile.set_shape(
                "input",
                (self.opt_bs, c, h_, w),
                (self.opt_bs, c, h_, w),
                (self.opt_bs, c, h_, w),
            )
            config.set_calibration_profile(calib_profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build returned None (see builder log above)")
        return bytes(serialized)

    # ---- inference -----------------------------------------------------------
    def infer(self, inputs: Any) -> Any:
        torch, trt = self._torch, self._trt
        inputs = inputs.contiguous()
        bs = inputs.shape[0]
        self.context.set_input_shape(self._in_name, tuple(inputs.shape))

        out = self._out_buffers.get(bs)
        if out is None:
            out_shape = tuple(self.context.get_tensor_shape(self._out_name))
            out = torch.empty(out_shape, device=self.device, dtype=inputs.dtype)
            self._out_buffers[bs] = out

        self.context.set_tensor_address(self._in_name, inputs.data_ptr())
        self.context.set_tensor_address(self._out_name, out.data_ptr())
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        # The harness synchronizes after the timed loop; we only ensure the
        # caller's default-stream reads see our stream's writes.
        torch.cuda.current_stream().wait_stream(self.stream)
        return out
