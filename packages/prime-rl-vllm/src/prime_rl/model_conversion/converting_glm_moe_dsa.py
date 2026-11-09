"""HF<->PrimeRL weight conversion for GLM-MoE-DSA.

The MoE layout is identical to GLM4-MoE. Attention and sparse-indexer
parameters retain their names in both formats.
"""

from __future__ import annotations

from prime_rl.model_conversion.converting_glm4_moe import glm_moe_layer_ops
from prime_rl.model_conversion.ops import ConvOp


def conversion_chain(config) -> list[ConvOp]:
    ops: list[ConvOp] = []
    for layer_idx in range(config.num_hidden_layers):
        ops.extend(glm_moe_layer_ops(layer_idx))
    return ops
