"""Backend abstraction.

Contract (kept intentionally minimal — see build philosophy in the plan):

  backend = SomeBackend(dtype="fp16")
  backend.build(model, example_inputs)   # one-time: compile/export/build engine
  out = backend.infer(inputs)            # hot path: enqueue one forward pass
  backend.name                           # label for tables/plots

`infer` must be allocation-light and must NOT synchronize — the timing harness
owns synchronization. `build` may be slow (TRT engine build, torch.compile
warmup) and is excluded from timed measurements by construction.
"""

from __future__ import annotations

import abc
from typing import Any


class Backend(abc.ABC):
    #: short, stable label used in result rows / plot legends
    name: str = "base"

    def __init__(self, dtype: str = "fp16", device: str = "cuda") -> None:
        self.dtype = dtype
        self.device = device

    @abc.abstractmethod
    def build(self, model: Any, example_inputs: Any) -> "Backend":
        """Prepare the backend for inference. Returns self for chaining."""

    @abc.abstractmethod
    def infer(self, inputs: Any) -> Any:
        """Run one forward pass and return the output tensor(s)."""

    def teardown(self) -> None:
        """Release backend-held resources (engines, sessions). Optional."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r}, dtype={self.dtype!r})"
