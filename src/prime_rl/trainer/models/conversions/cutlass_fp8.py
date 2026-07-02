"""Cutlass-style FP8 e4m3 with per-output-channel scaling. Registered as
``"cutlass_fp8_e4m3_per_channel"``.

Layout contract (matches cutlass ``scaled_mm`` + vLLM's native FP8 path):

* 2D linear weights: ``W.shape == (out_features, in_features)``, scale is
  one float32 per output row → ``scale.shape == (out_features,)``.
* 3D stacked-expert MoE weights:
  ``W.shape == (num_local_experts, out_features, in_features)``, scale is
  one float32 per (expert, output-row) → ``scale.shape == (num_local_experts, out_features)``.

Dispatches between the 2D and 3D paths via :func:`fp8_per_channel_quantize_into`
based on ``src.ndim`` — same dispatch convention as ``fp8_128x128``.

Tagged with ``compile_target="cutlass_fp8"`` so receivers running cutlass
kernels can filter for it via the v2 MX client's
``discover_v2_sources(compile_target_filter={"cutlass_fp8"})`` (Phase 3b,
``ai-dynamo/modelexpress:kavink/post-2389-phase3-4``).

``compile_metadata`` documents the byte-affecting choices:

* ``dtype``: ``"e4m3"`` (vs ``"e5m2"`` for higher-range cutlass variants —
  add as a separate entry when needed).
* ``scale_layout``: ``"per_channel"`` — receiver must allocate a 1D scale
  per output row, not a 2D blockwise scale.
* ``scale_axis``: ``-1`` — reduction was over the input-features axis;
  receiver dequantizes by broadcasting scale along the same axis.
* ``activation_scheme``: ``"dynamic"`` — matches HF's
  ``quantization_config.activation_scheme="dynamic"`` for cutlass FP8.

Adding a sibling cutlass entry (e.g. per-token activations, e5m2, etc.) is
~80 LOC in another file that calls :func:`register` and
:func:`register_default_rule` for its own HF-config signature.
"""

from __future__ import annotations

from torch import Tensor

from prime_rl.trainer.models.conversions import (
    COMPILE_TARGET_CUTLASS_FP8,
    register,
    register_default_rule,
)
from prime_rl.trainer.models.fp8 import fp8_per_channel_quantize_into


def cutlass_fp8_e4m3_per_channel(
    src: Tensor,
    out: Tensor,
    scale_out: Tensor | None,
) -> None:
    """Quantize ``src`` (bf16 or fp32) into per-channel FP8 e4m3.

    Writes into preallocated ``out`` (e4m3) + ``scale_out`` (float32).
    Dispatches 2D vs 3D via ``src.ndim`` — same convention as
    ``fp8_128x128``.
    """
    assert scale_out is not None, (
        "cutlass_fp8_e4m3_per_channel requires a scale_out buffer"
    )
    fp8_per_channel_quantize_into(src, out=out, sf=scale_out)


register(
    "cutlass_fp8_e4m3_per_channel",
    cutlass_fp8_e4m3_per_channel,
    requires_scale=True,
    compile_target=COMPILE_TARGET_CUTLASS_FP8,
    compile_metadata={
        "dtype": "e4m3",
        "scale_layout": "per_channel",
        "scale_axis": -1,
        "activation_scheme": "dynamic",
    },
)


def _is_cutlass_fp8_per_channel(quant: dict) -> bool:
    """HF ``quantization_config`` signature for cutlass per-channel FP8.

    Two recognised shapes:

    * ``{"quant_method": "fp8", "weight_block_size": None,
        "activation_scheme": "dynamic"}`` — what vLLM and most cutlass-
        targeting checkpoints publish.
    * ``{"quant_method": "fp8", "quant_format": "cutlass"}`` — used by a
        few model cards (Qwen3-MoE FP8 cutlass variants in particular)
        that disambiguate cutlass from DeepGemm by setting an explicit
        format string instead of leaving ``weight_block_size`` empty.

    The DeepGemm 128x128 rule (registered earlier) takes precedence when
    both predicates would match because that rule was registered before
    this one in ``_DEFAULT_RULES``.
    """
    if quant.get("quant_method") != "fp8":
        return False
    if quant.get("quant_format") == "cutlass":
        return True
    block_size = tuple(quant.get("weight_block_size") or ())
    activation_scheme = quant.get("activation_scheme")
    return block_size == () and activation_scheme == "dynamic"


register_default_rule(_is_cutlass_fp8_per_channel, "cutlass_fp8_e4m3_per_channel")
