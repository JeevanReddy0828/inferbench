"""Inference backends behind a uniform build()/infer() contract.

Import the registry helper, not the modules directly — several backends import
heavy/optional deps (tensorrt, onnxruntime) at construction time, so they are
resolved lazily.
"""

from .base import Backend

__all__ = ["Backend", "get_backend"]


def get_backend(name: str) -> type[Backend]:
    """Resolve a backend class by short name, importing lazily."""
    name = name.lower()
    if name in ("eager", "torch", "torch_eager"):
        from .torch_eager import TorchEagerBackend

        return TorchEagerBackend
    if name in ("compile", "torch_compile", "inductor"):
        from .torch_compile import TorchCompileBackend

        return TorchCompileBackend
    if name in ("onnx", "ort", "onnxruntime"):
        from .onnxruntime_be import OnnxRuntimeBackend

        return OnnxRuntimeBackend
    if name in ("trt", "tensorrt"):
        from .tensorrt_be import TensorRTBackend

        return TensorRTBackend
    if name in ("torch_trt", "torch-tensorrt", "torchtrt"):
        from .torch_tensorrt_be import TorchTensorRTBackend

        return TorchTensorRTBackend
    raise KeyError(f"unknown backend '{name}'")
