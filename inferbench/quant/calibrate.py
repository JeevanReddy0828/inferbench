"""TensorRT INT8 post-training quantization (entropy calibration).

INT8 PTQ maps fp32/fp16 activations to int8 using per-tensor scales chosen to
minimize the information loss of the quantization. TensorRT's
``IInt8EntropyCalibrator2`` derives those scales by observing activation
histograms over a small, *representative* calibration set and picking the
clipping threshold that minimizes KL-divergence between the fp and quantized
distributions.

Two things make or break INT8 accuracy, and both are handled here:
  1. **Representative data.** Calibration on random noise produces garbage
     scales. ``build_calibration_batches`` loads real preprocessed images; a
     random fallback exists only for pipeline smoke-tests and *loudly warns*.
  2. **A cached calibration table.** Calibration is slow; the resulting table is
     cached to disk so rebuilds are fast and reproducible.

Device staging uses a torch CUDA tensor (kept alive for the calibrator's
lifetime) so we avoid a hard pycuda dependency — ``get_batch`` just hands TRT
the tensor's ``data_ptr()``.
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable, Sequence

import numpy as np


def build_calibration_batches(
    data_dir: str | None,
    batch_size: int,
    num_batches: int,
    image_size: int = 224,
) -> list[np.ndarray]:
    """Return ``num_batches`` of shape (batch_size, 3, H, W) fp32 NCHW.

    If ``data_dir`` is a folder of images, real ImageNet-style preprocessing is
    applied (resize/center-crop/normalize). Otherwise a clearly-warned random
    fallback is used so the *pipeline* is testable without a dataset — never
    trust INT8 accuracy numbers produced from the random path.
    """
    if data_dir and os.path.isdir(data_dir):
        return _load_image_batches(data_dir, batch_size, num_batches, image_size)

    warnings.warn(
        "No calibration data_dir provided — using RANDOM calibration data. "
        "INT8 accuracy from this path is meaningless; supply a real image folder "
        "for any reported accuracy number.",
        RuntimeWarning,
        stacklevel=2,
    )
    rng = np.random.default_rng(0)
    return [
        rng.standard_normal((batch_size, 3, image_size, image_size)).astype(np.float32)
        for _ in range(num_batches)
    ]


def _load_image_batches(
    data_dir: str, batch_size: int, num_batches: int, image_size: int
) -> list[np.ndarray]:
    import torch
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    # Standard ImageNet eval preprocessing (matches torchvision ResNet50_Weights).
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = ImageFolder(data_dir, transform=tfm)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)

    batches: list[np.ndarray] = []
    for imgs, _ in loader:
        batches.append(imgs.numpy().astype(np.float32))
        if len(batches) >= num_batches:
            break
    if not batches:
        raise RuntimeError(f"no usable images found under {data_dir}")
    return batches


def EntropyCalibrator(
    batches: Sequence[np.ndarray],
    cache_file: str,
    input_name: str = "input",
):
    """Factory returning a TRT IInt8EntropyCalibrator2 bound to ``batches``.

    Implemented as a factory (not a top-level class) so this module imports
    without ``tensorrt`` present — the class is defined lazily against the live
    TRT/torch runtime inside the WSL2 env.
    """
    import tensorrt as trt
    import torch

    class _EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            super().__init__()
            self.batches: Iterable[np.ndarray] = iter(batches)
            self.cache_file = cache_file
            self.input_name = input_name
            self.batch_size = int(batches[0].shape[0])
            # Persistent device staging buffer (reused every get_batch).
            self.device_input = torch.empty(
                tuple(batches[0].shape), device="cuda", dtype=torch.float32
            )

        def get_batch_size(self) -> int:  # noqa: N802 (TRT API name)
            return self.batch_size

        def get_batch(self, names):  # noqa: N802
            try:
                arr = next(self.batches)
            except StopIteration:
                return None  # signals calibration is done
            host = torch.from_numpy(np.ascontiguousarray(arr))
            self.device_input.copy_(host.to("cuda", non_blocking=True))
            return [int(self.device_input.data_ptr())]

        def read_calibration_cache(self):  # noqa: N802
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as fh:
                    return fh.read()
            return None

        def write_calibration_cache(self, cache):  # noqa: N802
            with open(self.cache_file, "wb") as fh:
                fh.write(cache)

    return _EntropyCalibrator()
