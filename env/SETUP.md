# Environment Setup

Target hardware: **NVIDIA RTX 3050 Ti Laptop, 4 GB VRAM, Ampere (sm_86)**.

The host already has NVIDIA driver **576.80** (CUDA 12.x capable). The host's
system Python is **3.14**, which has **no wheels** for PyTorch / TensorRT / ONNX
Runtime / Triton. So we build inside a **Python 3.11** environment. The two
supported paths are below — **WSL2 is recommended** because Triton and custom
CUDA C++ extensions are Linux-first.

---

## Path A — WSL2 + Ubuntu 22.04 (recommended)

### 1. Install WSL2 with GPU passthrough
From an **elevated PowerShell** on Windows:
```powershell
wsl --install -d Ubuntu-22.04
wsl --update
```
GPU passthrough uses the **existing Windows driver (576.80)** — do **not** install
an NVIDIA driver inside WSL. Only the CUDA *toolkit* goes in WSL.

Verify inside Ubuntu:
```bash
nvidia-smi    # should list the RTX 3050 Ti
```

### 2. CUDA toolkit (12.4) inside WSL
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version   # expect release 12.4
```

### 3. Python 3.11 env + the DL stack
```bash
sudo apt-get -y install python3.11 python3.11-venv python3.11-dev build-essential
python3.11 -m venv ~/.venvs/inferbench
source ~/.venvs/inferbench/bin/activate
pip install -U pip wheel

# PyTorch (cu124) — must match the CUDA 12.4 toolkit so the v2 CUDA extension builds.
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Triton ships with the linux torch wheel; if not present:
pip install triton==3.1.0

# ONNX + ONNX Runtime GPU + TensorRT (10.x) + polygraphy
pip install onnx==1.17.0 onnxruntime-gpu==1.20.1
pip install tensorrt==10.7.0
pip install polygraphy==0.49.9

# Project + dev tooling
pip install -e ".[dev]"
```

> **CUDA pin rule:** the torch wheel CUDA version (cu124) and the system CUDA
> toolkit (12.4) must match — the v2 custom CUDA extension compiles against the
> system toolkit and links against torch's runtime. Mismatches surface as
> `undefined symbol` at import.

### 4. Smoke test
```bash
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
import tensorrt as trt; print("tensorrt", trt.__version__)
import onnxruntime as ort; print("ort providers", ort.get_available_providers())
import triton; print("triton", triton.__version__)
PY
```
Expected: `cuda? True`, device `NVIDIA GeForce RTX 3050 Ti Laptop GPU`, and
`CUDAExecutionProvider` in the ORT provider list.

---

## Path B — Native Windows (fallback, reduced scope)

Use this only if WSL2 is not an option. Triton + custom CUDA C++ are degraded
on native Windows; v1 (eager / compile / ONNX / TensorRT FP16) still works.

```powershell
# Install Python 3.11 from python.org first, then:
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip wheel
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install onnx==1.17.0 onnxruntime-gpu==1.20.1 tensorrt==10.7.0 polygraphy==0.49.9
pip install -e ".[dev]"
# Triton on Windows (flaky): pip install triton-windows   # optional, for the kernel demo
```

---

## Running

```bash
# Pure-Python methodology tests (run anywhere, no GPU):
pytest -m "not gpu"

# Full suite incl. GPU correctness/parity:
pytest

# v1 headline sweep:
python scripts/run_vision_bench.py

# Triton kernel microbenchmark:
python scripts/bench_kernel.py
```

## VRAM budgeting (4 GB)
- Vision batch sizes capped at 8; larger points are recorded as `OOM`, not dropped.
- TensorRT workspace pool limited to 3 GB (see `tensorrt_be.py`).
- `nvidia-smi -l 1` in a second terminal while sweeping to watch headroom.
