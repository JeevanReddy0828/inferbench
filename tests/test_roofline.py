"""Unit tests for the roofline model (pure math, no GPU)."""

import math

import pytest

from inferbench.analysis.roofline import (
    DevicePeaks,
    RooflinePoint,
    attainable_tflops,
    classify,
    utilization,
)


def test_ridge_point():
    # 7600 GFLOP/s / 190 GB/s = 40 FLOP/byte (use round numbers).
    peaks = DevicePeaks(mem_bandwidth_gb_s=190.0, peak_tflops={"x": 7.6})
    assert peaks.ridge_ai("x") == pytest.approx(7.6e12 / 190e9)


def test_arithmetic_intensity():
    p = RooflinePoint("op", flops=2000.0, bytes_moved=1000.0)
    assert p.arithmetic_intensity == pytest.approx(2.0)


def test_attainable_is_min_of_ceilings():
    peaks = DevicePeaks(mem_bandwidth_gb_s=100.0, peak_tflops={"x": 10.0})
    # Low AI -> memory bound: 100 GB/s * 1 FLOP/byte = 100 GFLOP/s = 0.1 TFLOP/s
    assert attainable_tflops(1.0, peaks, "x") == pytest.approx(0.1)
    # Very high AI -> clamped at compute peak
    assert attainable_tflops(1e6, peaks, "x") == pytest.approx(10.0)


def test_classify_memory_vs_compute_bound():
    peaks = DevicePeaks(mem_bandwidth_gb_s=100.0, peak_tflops={"x": 10.0})
    ridge = peaks.ridge_ai("x")  # 100 FLOP/byte
    mem_op = RooflinePoint("rmsnorm", flops=ridge * 10, bytes_moved=10)  # AI=ridge... below
    low = RooflinePoint("elementwise", flops=2.0, bytes_moved=8.0)       # AI=0.25, very low
    high = RooflinePoint("gemm", flops=1e9, bytes_moved=1.0)             # AI huge
    assert classify(low, peaks, "x") == "memory-bound"
    assert classify(high, peaks, "x") == "compute-bound"


def test_from_latency_computes_tflops():
    # 2e12 FLOPs in 2 ms -> 1e15 FLOP/s = 1000 TFLOP/s
    p = RooflinePoint.from_latency("op", flops=2e12, bytes_moved=1e9, latency_ms=2.0)
    assert p.measured_tflops == pytest.approx(1000.0)


def test_utilization_fraction():
    peaks = DevicePeaks(mem_bandwidth_gb_s=100.0, peak_tflops={"x": 10.0})
    # AI=1 -> ceiling 0.1 TFLOP/s; measured 0.05 -> 50% util
    p = RooflinePoint("op", flops=1.0, bytes_moved=1.0, measured_tflops=0.05)
    assert utilization(p, peaks, "x") == pytest.approx(0.5)


def test_utilization_nan_without_measurement():
    peaks = DevicePeaks()
    p = RooflinePoint("op", flops=1.0, bytes_moved=1.0)
    assert math.isnan(utilization(p, peaks, "fp16_tensor"))
