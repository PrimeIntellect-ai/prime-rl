"""Kimi K2 DSA weight conversion. Reuses kimi_k2's conversion as-is: the indexer's HF and
prime parameter names are identical (no fusion needed for training — only vLLM's fp8 kernel
serving format fuses indexer keys, which this family doesn't implement), and the MoE/
multimodal-wrapper handling is architecture-identical to the dense kimi_k2 family.
"""

from prime_rl.trainer.models.kimi_k2.converting_kimi_k2 import (
    convert_hf_layer_to_tt,
    convert_hf_to_tt_moe,
    convert_tt_layer_to_hf,
    convert_tt_to_hf_moe,
    strip_multimodal_wrapper,
)

__all__ = [
    "convert_hf_layer_to_tt",
    "convert_hf_to_tt_moe",
    "convert_tt_layer_to_hf",
    "convert_tt_to_hf_moe",
    "strip_multimodal_wrapper",
]
