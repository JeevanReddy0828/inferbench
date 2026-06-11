#!/usr/bin/env python
"""TensorRT INT8 PTQ study: FP16 vs INT8 latency (and top-1 accuracy delta).

Builds two ResNet50 TensorRT engines — FP16 and INT8 (entropy-calibrated) — and
reports the speedup and, if a validation folder is provided, the top-1 accuracy
delta. This is the v2 "accuracy-vs-latency tradeoff" deliverable.

  # latency only (random calibration — accuracy NOT meaningful, just the pipeline):
  python scripts/run_int8_study.py

  # real study (calibration + accuracy on ImageNet-style folders):
  python scripts/run_int8_study.py --calib-dir /data/imagenet/calib \
                                   --val-dir   /data/imagenet/val --num-calib-batches 16
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inferbench.bench.runner import time_callable  # noqa: E402
from inferbench.config import BenchConfig  # noqa: E402
from inferbench.workloads.vision import VisionWorkload  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TensorRT INT8 PTQ study (ResNet50)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--calib-dir", default=None, help="ImageFolder of calibration images")
    p.add_argument("--val-dir", default=None, help="ImageFolder for top-1 accuracy")
    p.add_argument("--num-calib-batches", type=int, default=16)
    p.add_argument("--warmup-iters", type=int, default=25)
    p.add_argument("--timed-iters", type=int, default=100)
    return p.parse_args()


def top1_accuracy(backend, val_dir: str, batch_size: int) -> float:
    import torch
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = ImageFolder(val_dir, transform=tfm)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, drop_last=True)
    correct = total = 0
    for imgs, labels in loader:
        imgs = imgs.cuda().half()
        out = backend.infer(imgs)
        pred = out.argmax(dim=-1).cpu()
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / max(total, 1)


def main() -> int:
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    import torch

    from inferbench.backends.tensorrt_be import TensorRTBackend
    from inferbench.quant import EntropyCalibrator, build_calibration_batches

    if not torch.cuda.is_available():
        print("ERROR: CUDA required.", file=sys.stderr)
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    wl = VisionWorkload(name="resnet50")
    model = wl.build_model()
    bs = args.batch_size
    example = wl.example_input(bs)
    cfg = BenchConfig(warmup_iters=args.warmup_iters, timed_iters=args.timed_iters,
                      batch_sizes=(bs,))
    inp = wl.example_input(bs).cuda().half()

    # --- FP16 baseline engine ---
    print("building FP16 engine ...")
    fp16 = TensorRTBackend(dtype="fp16", opt_bs=bs, max_bs=bs, min_bs=bs).build(model, example)
    r_fp16 = time_callable(lambda: fp16.infer(inp), backend="trt-fp16",
                           workload="resnet50", batch_size=bs, dtype="fp16", config=cfg)

    # --- INT8 engine (entropy-calibrated) ---
    print("calibrating + building INT8 engine ...")
    batches = build_calibration_batches(args.calib_dir, bs, args.num_calib_batches)
    cache = os.path.join(RESULTS_DIR, "resnet50_int8_calib.cache")
    calibrator = EntropyCalibrator(batches, cache_file=cache)
    int8 = TensorRTBackend(dtype="fp16", opt_bs=bs, max_bs=bs, min_bs=bs,
                           calibrator=calibrator).build(model, example)
    r_int8 = time_callable(lambda: int8.infer(inp), backend="trt-int8",
                           workload="resnet50", batch_size=bs, dtype="int8", config=cfg)

    # --- report ---
    speedup = r_fp16.p50_ms / r_int8.p50_ms if r_int8.p50_ms > 0 else float("nan")
    print("\n=== ResNet50 @ bs={} ===".format(bs))
    print(f"FP16 : p50={r_fp16.p50_ms:7.3f} ms  thru={r_fp16.throughput_samples_s:8.1f}  "
          f"vram={r_fp16.peak_vram_mb:6.0f} MB")
    print(f"INT8 : p50={r_int8.p50_ms:7.3f} ms  thru={r_int8.throughput_samples_s:8.1f}  "
          f"vram={r_int8.peak_vram_mb:6.0f} MB")
    print(f"INT8 speedup over FP16: {speedup:.2f}x")

    if args.val_dir:
        acc_fp16 = top1_accuracy(fp16, args.val_dir, bs)
        acc_int8 = top1_accuracy(int8, args.val_dir, bs)
        print(f"\ntop-1: FP16={acc_fp16:.2f}%  INT8={acc_int8:.2f}%  "
              f"delta={acc_int8 - acc_fp16:+.2f} pts")
    else:
        print("\n[note] no --val-dir: accuracy not measured. Latency above is valid; "
              "INT8 accuracy requires a real validation set (and real --calib-dir).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
