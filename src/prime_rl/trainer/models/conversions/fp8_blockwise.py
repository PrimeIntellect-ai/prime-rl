"""FP8 e4m3 blockwise quantization, 128x128 blocks. Registered as ``"fp8_128x128"``.

Dispatches between the 2D linear layer path and the 3D stacked-expert path
based on ``src.ndim``. Tagged with ``compile_target="deep_gemm_fp8"`` so
receivers running DeepGemm kernels can filter for it via the v2 MX
client's ``discover_v2_sources(compile_target_filter=…)`` (Phase 3b).
"""

from __future__ import annotations

from torch import Tensor

from prime_rl.trainer.models.conversions import (
    COMPILE_TARGET_DEEPGEMM_FP8,
    register,
    register_default_rule,
)
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
    compile_target=COMPILE_TARGET_DEEPGEMM_FP8,
    compile_metadata={
        "dtype": "e4m3",
        "scale_layout": "blockwise",
        "block_size": [128, 128],
    },
)

# HF config signature for DeepGemm-style FP8: 128x128 blockwise.
register_default_rule(
    lambda quant: (
        quant.get("quant_method") == "fp8"
        and tuple(quant.get("weight_block_size") or ()) == (128, 128)
    ),
    "fp8_128x128",
)
