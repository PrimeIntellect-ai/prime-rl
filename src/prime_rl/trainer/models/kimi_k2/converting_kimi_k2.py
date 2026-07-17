"""Kimi K2 weight conversion. The MoE layout (per-expert gate/up/down + shared-experts +
`e_score_correction_bias` load-balancing term) is byte-identical to what GLM-4 MoE already
converts, since both follow the standard DeepSeek-style naming — reused as-is, no adaptation.
"""

from torch import Tensor

from prime_rl.trainer.models.glm4_moe.converting_glm4_moe import (
    convert_hf_layer_to_tt,
    convert_hf_to_tt_moe,
    convert_tt_layer_to_hf,
    convert_tt_to_hf_moe,
)

_LANGUAGE_MODEL_PREFIX = "language_model."


def strip_multimodal_wrapper(state_dict: dict[str, Tensor]) -> None:
    """Drop everything but the text backbone from a K2.5/K2.6/K2.7-style multimodal
    checkpoint, in-place.

    Those checkpoints wrap this exact text backbone (`language_model.model.layers.N...`)
    alongside a vision tower. We only support the text backbone (see `KimiK2Config`'s
    docstring) — un-prefix the language-model keys and drop everything else (vision tower,
    projector, etc.) whatever they're named, since we never need them. A no-op for a plain
    (non-multimodal) Kimi-K2 checkpoint, which carries no `language_model.`-prefixed keys.
    """
    prefixed = {k: v for k, v in state_dict.items() if k.startswith(_LANGUAGE_MODEL_PREFIX)}
    if not prefixed:
        return  # plain Kimi-K2 checkpoint, nothing to strip

    state_dict.clear()
    for key, value in prefixed.items():
        state_dict[key[len(_LANGUAGE_MODEL_PREFIX) :]] = value


__all__ = [
    "convert_hf_layer_to_tt",
    "convert_hf_to_tt_moe",
    "convert_tt_layer_to_hf",
    "convert_tt_to_hf_moe",
    "strip_multimodal_wrapper",
]
