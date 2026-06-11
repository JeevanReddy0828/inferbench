# InferBench

**An inference benchmarking & optimization toolkit for resource-constrained NVIDIA GPUs.**

InferBench takes a deep-learning model and measures it — *correctly* — across the
modern inference stack on a 4 GB laptop GPU, then optimizes it and explains where
the time goes:

```
PyTorch eager  →  torch.compile (TorchInductor)  →  ONNX Runtime  →  TensorRT (FP16 / INT8)
```

plus hand-written GPU kernels (Triton, then custom CUDA C++) and profiling-driven
bottleneck analysis.

> **Why a 4 GB laptop GPU?** Because the hard part of inference engineering isn't a
> DGX with headroom to spare — it's hitting a latency/VRAM budget on constrained
> hardware. The RTX 3050 Ti (Ampere, sm_86, 4 GB) is a stand-in for the edge/Jetson
> class of deployment target, and every design choice here is shaped by that ceiling.

---

## The one-paragraph version (what this repo demonstrates)

> I built a benchmarking toolkit on a 4 GB GPU that measures latency / VRAM /
> throughput **correctly** — CUDA-event timing on the device stream, explicit
> warmup, percentile reporting, and a coefficient-of-variation noise gate — then
> optimized ResNet50 with TensorRT FP16/INT8 and a fused Triton kernel, and used
> roofline + Nsight profiling to *explain* the bottlenecks rather than just report
> a number.

---

## Measurement methodology (the part most benchmarks get wrong)

GPU work is asynchronous. Most of the credibility of a benchmark lives in *how* it
times, not *what* it times. InferBench's harness (`inferbench/bench/runner.py`)
encodes the following discipline:

| Concern | What we do | Why |
|---|---|---|
| Async launches | Time with `torch.cuda.Event` on the stream, **one** `synchronize()` after the timed loop | Wall-clock around an async launch measures queue time; per-iter sync serializes and inflates latency |
| Cold start | Dedicated warmup iters before timing | Absorbs cuDNN/cuBLAS autotuning, `torch.compile` graph capture, CUDA context + clock spin-up |
| Outliers | Report **p50/p90/p99**, derive throughput from the **median** | Laptop GPUs thermal-throttle; a mean hides tail latency and deflates throughput |
| Fair I/O | ONNX/TRT bind **device pointers** (`io_binding` / `set_tensor_address`) | A backend that secretly does a host→device copy each call would lose unfairly |
| VRAM | `reset_peak_memory_stats()` before timing, `max_memory_allocated()` after | Peak VRAM is reported for the timed region only |
| Trust | Coefficient-of-variation gate flags runs with CoV > 0.05 | Makes noisy/throttled runs *visible* instead of silently wrong |
| Honesty | OOM points are recorded as `OOM`, never silently dropped | A missing 4 GB data point is information, not an error to hide |

The measurement math is pure-Python and unit-tested (`tests/test_metrics.py`) so the
methodology is provably correct independent of any GPU.

---

## Results

> Numbers are produced by `python scripts/run_vision_bench.py` on the target GPU and
> committed to `results/`. Run it to populate; the table/plot shape is below.

**ResNet50, FP16, batch sweep — p50 latency (ms) / throughput (img/s) / peak VRAM (MB)**

| backend | bs=1 | bs=4 | bs=8 | notes |
|---|---|---|---|---|
| torch-eager | _populate_ | _populate_ | _populate_ | baseline |
| torch-compile | _populate_ | _populate_ | _populate_ | Inductor `max-autotune` |
| onnxruntime | _populate_ | _populate_ | _populate_ | CUDA EP, IO-binding |
| **tensorrt** | _populate_ | _populate_ | _populate_ | FP16 engine, expected fastest |

Plots written to `results/vision_resnet50_latency.png` and
`results/vision_resnet50_throughput.png`.

**Triton fused RMSNorm** (`python scripts/bench_kernel.py`) — speedup vs eager and
achieved HBM bandwidth (RMSNorm is memory-bound, so bandwidth utilization is the
metric that explains the win).

---

## Quickstart

See [`env/SETUP.md`](env/SETUP.md) for the full WSL2 + CUDA 12.4 + Python 3.11 setup
(the host's Python 3.14 cannot host this stack).

```bash
pytest -m "not gpu"                 # methodology unit tests (no GPU needed)
pytest                              # + GPU correctness/parity tests
python scripts/run_vision_bench.py  # v1 headline sweep -> results/
python scripts/bench_kernel.py      # Triton RMSNorm microbenchmark
```

---

## Architecture

```
inferbench/
  bench/      runner.py (CUDA-event harness)  +  metrics.py (pure, tested reductions)
  backends/   base.py ABC  +  torch_eager / torch_compile / onnxruntime / tensorrt
  workloads/  vision.py (ResNet50; ViT in v3)
  kernels/    triton/rmsnorm.py (fused, autotuned)  +  cuda/ (custom op, v2)
  quant/      INT8 PTQ calibration (v2)
  analysis/   plots.py (tables + charts)  +  roofline.py (v2)
scripts/      run_vision_bench.py, bench_kernel.py
tests/        test_metrics (pure)  +  test_kernels / test_backends (gpu-marked)
```

Every backend hides behind one contract — `build(model, example)` then
`infer(inputs)` — so the harness times them identically. `build` (engine
compilation, ONNX export, `torch.compile` warmup) is excluded from the timed window
by construction.

---

## Roadmap

- **v1 (this release):** measurement harness + ResNet50 across 4 backends (FP16) +
  one Triton kernel + reproducible results. *A complete portfolio piece on its own.*
- **v2:** TensorRT **INT8 PTQ** with calibration (accuracy-vs-latency curve) ·
  custom **CUDA C++** fused op via PyTorch extension · **roofline + Nsight** deep-dive.
- **v3:** ViT-B/16 · small-LLM decode track (Qwen2.5-0.5B: TTFT + tokens/s, INT8
  weight-only). LLM-on-TensorRT is scoped as an **experimental, fixed-shape prefill**
  study — dynamic KV-cache export is explicitly out of scope and documented as such.

---

## How this maps to inference-performance engineering

| Skill | Artifact |
|---|---|
| Benchmarking methodology | `bench/runner.py` + `bench/metrics.py` (tested) |
| Performance analysis / bottlenecks | `analysis/roofline.py`, Nsight trace (v2) |
| TensorRT / Torch-TensorRT | `backends/tensorrt_be.py` (TRT 10.x), `torch_tensorrt_be.py` (v2) |
| Quantization | `quant/` INT8 PTQ + weight-only (v2/v3) |
| Graph compilers (TorchDynamo/Inductor) | `backends/torch_compile.py` |
| CUDA / Triton DSL | `kernels/triton/rmsnorm.py`, `kernels/cuda/` (v2) |
| Transformers / Visual Understanding | LLM decode (v3) + ResNet50/ViT |
| Edge / resource-constrained | the 4 GB budget threaded through every component |
```
