"""Smoke + parity tests for the inference backends.

Marked `gpu`. Each backend must (a) build and run, and (b) produce a
classification output that agrees with the eager baseline's top-1 prediction —
catching export/precision bugs that a pure latency sweep would happily report
as a "fast" but wrong engine.
"""

import pytest

torch = pytest.importorskip("torch")

from inferbench.backends import get_backend  # noqa: E402
from inferbench.workloads.vision import VisionWorkload  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_eager_backend_runs():
    wl = VisionWorkload(name="resnet50")
    model = wl.build_model()
    be = get_backend("eager")(dtype="fp16").build(model, wl.example_input(2))
    out = be.infer(wl.example_input(2).cuda().half())
    assert out.shape == (2, 1000)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("backend_name", ["compile", "onnx", "tensorrt"])
def test_backend_top1_matches_eager(backend_name):
    """Optimized backends must agree with eager on the predicted class."""
    wl = VisionWorkload(name="resnet50")
    model = wl.build_model()
    x = wl.example_input(2).cuda().half()

    eager = get_backend("eager")(dtype="fp16").build(model, wl.example_input(2))
    ref_top1 = eager.infer(x).argmax(dim=-1).cpu()

    try:
        be = get_backend(backend_name)(dtype="fp16").build(model, wl.example_input(2))
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"{backend_name} unavailable in this env: {exc}")

    out = be.infer(x)
    # ONNX backend returns an io-binding; pull the tensor for comparison.
    if hasattr(out, "copy_outputs_to_cpu"):
        got_top1 = torch.as_tensor(out.copy_outputs_to_cpu()[0]).argmax(dim=-1)
    else:
        got_top1 = out.argmax(dim=-1).cpu()

    assert torch.equal(got_top1, ref_top1), f"{backend_name} top-1 disagrees with eager"
