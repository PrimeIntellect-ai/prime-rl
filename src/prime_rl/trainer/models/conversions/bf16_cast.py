"""Plain dtype-cast conversion. Despite the name, casts to whatever dtype
the destination buffer is — bf16, fp32, etc. Registered as ``"passthrough"``.
"""

from __future__ import annotations

from torch import Tensor

from prime_rl.trainer.models.conversions import register


def passthrough(src: Tensor, out: Tensor, scale_out: Tensor | None = None) -> None:
    assert scale_out is None, "passthrough conversion takes no scale buffer"
    out.copy_(src.to(out.dtype))


register("passthrough", passthrough, requires_scale=False)
