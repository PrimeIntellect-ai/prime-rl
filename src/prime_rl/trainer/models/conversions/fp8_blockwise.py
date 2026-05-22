"""FP8 e4m3 blockwise quantization, 128x128 blocks. Registered as ``"fp8_128x128"``.

Dispatches between the 2D linear layer path and the 3D stacked-expert path
based on ``src.ndim``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from prime_rl.trainer.models.conversions import register
from prime_rl.trainer.models.fp8 import fp8_block_quantize, grouped_fp8_block_quantize


def fp8_128x128(src: Tensor, out: Tensor, scale_out: Tensor | None) -> None:
    assert scale_out is not None, "fp8_128x128 requires a scale_out buffer"
    if src.ndim == 3:
        grouped_fp8_block_quantize(src, out=out, sf=scale_out)
    else:
        fp8_block_quantize(src, out=out, sf=scale_out)


register(
    "fp8_128x128",
    fp8_128x128,
    requires_scale=True,
    dst_dtype=torch.float8_e4m3fn,
)
