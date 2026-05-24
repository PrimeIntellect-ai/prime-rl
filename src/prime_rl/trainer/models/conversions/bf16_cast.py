"""Plain dtype-cast conversions for non-FP8 destination buffers."""

from __future__ import annotations

import torch
from torch import Tensor

from prime_rl.trainer.models.conversions import register


def bf16_cast(src: Tensor, out: Tensor, scale_out: Tensor | None = None) -> None:
    assert scale_out is None, "bf16_cast conversion takes no scale buffer"
    out.copy_(src.to(torch.bfloat16))


def fp32_cast(src: Tensor, out: Tensor, scale_out: Tensor | None = None) -> None:
    assert scale_out is None, "fp32_cast conversion takes no scale buffer"
    out.copy_(src.to(torch.float32))


register("bf16_cast", bf16_cast, requires_scale=False)
register("fp32_cast", fp32_cast, requires_scale=False)
