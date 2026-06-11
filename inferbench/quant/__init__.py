"""Post-training quantization utilities (v2).

v2 ships TensorRT INT8 PTQ via an entropy calibrator. Weight-only INT8 for the
LLM track arrives in v3 (`weight_only.py`).
"""

from .calibrate import EntropyCalibrator, build_calibration_batches

__all__ = ["EntropyCalibrator", "build_calibration_batches"]
