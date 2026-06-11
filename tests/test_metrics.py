"""Unit tests for the measurement math (no GPU / torch needed).

This is the methodology core, so it gets real tests: percentile correctness,
throughput derivation, speedup, CoV, and the NaN/empty edge cases the sweep
relies on when a point OOMs.
"""

import math

import numpy as np
import pytest

from inferbench.bench import metrics
from inferbench.config import BenchConfig, LatencyResult, load_results, save_results


def test_percentile_matches_numpy():
    data = [float(x) for x in range(1, 101)]  # 1..100
    for q in (0, 25, 50, 90, 99, 100):
        assert metrics.percentile(data, q) == pytest.approx(np.percentile(data, q))


def test_percentile_empty_is_nan():
    assert math.isnan(metrics.percentile([], 50))


def test_percentile_out_of_range_raises():
    with pytest.raises(ValueError):
        metrics.percentile([1.0], 101)


def test_summarize_basic_stats():
    # latencies in ms; constant 10ms -> p50=10, throughput at bs=4 = 400 smp/s
    s = metrics.summarize([10.0] * 50, batch_size=4)
    assert s["p50_ms"] == pytest.approx(10.0)
    assert s["mean_ms"] == pytest.approx(10.0)
    assert s["std_ms"] == pytest.approx(0.0)
    assert s["throughput_samples_s"] == pytest.approx(400.0)


def test_summarize_throughput_uses_median_not_mean():
    # One huge outlier should NOT tank throughput because we use the median.
    data = [10.0] * 99 + [1000.0]
    s = metrics.summarize(data, batch_size=1)
    assert s["p50_ms"] == pytest.approx(10.0)
    # median-based throughput ~ 100 smp/s; a mean-based one would be ~50.
    assert s["throughput_samples_s"] == pytest.approx(100.0, rel=0.05)
    assert s["mean_ms"] > 15.0  # mean is dragged up by the outlier


def test_summarize_empty_all_nan():
    s = metrics.summarize([], batch_size=1)
    assert all(math.isnan(v) for v in s.values())


def test_speedup():
    assert metrics.speedup(20.0, 10.0) == pytest.approx(2.0)
    assert math.isnan(metrics.speedup(20.0, 0.0))
    assert math.isnan(metrics.speedup(float("nan"), 10.0))


def test_coefficient_of_variation():
    assert metrics.coefficient_of_variation([10.0] * 10) == pytest.approx(0.0)
    cov = metrics.coefficient_of_variation([9.0, 11.0, 10.0, 10.0])
    assert cov > 0.0
    assert math.isnan(metrics.coefficient_of_variation([]))


def test_benchconfig_validation():
    with pytest.raises(ValueError):
        BenchConfig(timed_iters=0)
    with pytest.raises(ValueError):
        BenchConfig(batch_sizes=())


def test_result_roundtrip_json(tmp_path):
    r = LatencyResult(
        backend="tensorrt",
        workload="resnet50",
        batch_size=4,
        dtype="fp16",
        latencies_ms=[1.0, 2.0, 3.0],
        p50_ms=2.0,
    )
    path = tmp_path / "r.json"
    save_results([r], str(path))
    loaded = load_results(str(path))
    assert len(loaded) == 1
    assert loaded[0].backend == "tensorrt"
    assert loaded[0].latencies_ms == [1.0, 2.0, 3.0]


def test_result_to_row_drops_raw_latencies():
    r = LatencyResult("trt", "resnet50", 1, "fp16", latencies_ms=[1.0, 2.0])
    row = r.to_row()
    assert "latencies_ms" not in row
    assert row["backend"] == "trt"
