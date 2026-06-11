"""Vision classification workload.

v1: ResNet50 (ImageNet-pretrained). The model + a synthetic-but-correctly-shaped
input is all the *latency/throughput* benchmark needs — pixel values don't affect
kernel timing. Real ImageNet data is only required for the v2 INT8 accuracy
study, so it's an optional path here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class VisionWorkload:
    name: str
    image_size: int = 224
    channels: int = 3

    def build_model(self) -> Any:
        """Return an eval-mode fp32 model on CPU; backends own device/dtype placement."""
        import torchvision.models as tvm

        if self.name == "resnet50":
            weights = tvm.ResNet50_Weights.IMAGENET1K_V2
            return tvm.resnet50(weights=weights).eval()
        if self.name == "vit_b_16":  # available for v3 breadth; harmless to keep
            weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1
            return tvm.vit_b_16(weights=weights).eval()
        raise KeyError(f"unknown vision workload '{self.name}'")

    def example_input(self, batch_size: int = 1) -> Any:
        """A correctly-shaped fp32 CPU tensor (deterministic via fixed seed)."""
        import torch

        g = torch.Generator().manual_seed(0)
        return torch.randn(
            batch_size, self.channels, self.image_size, self.image_size, generator=g
        )


def resnet50() -> VisionWorkload:
    return VisionWorkload(name="resnet50")
