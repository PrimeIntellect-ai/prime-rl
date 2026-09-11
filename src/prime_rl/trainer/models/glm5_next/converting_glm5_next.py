"""HF<->PrimeRL weight conversion for text-only GLM-5.3."""

from __future__ import annotations

from prime_rl.trainer.models.conversion_ops import ConvOp, Drop, PrefixRename
from prime_rl.trainer.models.glm4_moe.converting_glm4_moe import glm_moe_layer_ops


def conversion_chain(config) -> list[ConvOp]:
    ops: list[ConvOp] = [
        PrefixRename("model.language_model.", "model."),
        Drop("model.visual.", is_prefix=True),
        Drop("visual.", is_prefix=True),
    ]
    for layer_idx in range(config.num_hidden_layers):
        ops.extend(glm_moe_layer_ops(layer_idx))
        p = f"model.layers.{layer_idx}.self_attn.indexer"
        ops.append(Drop(f"{p}.index_kpool_compress_ape"))
        ops.append(Drop(f"{p}.index_kpool_compress_gate"))

    num_extra_layers = getattr(config, "num_nextn_predict_layers", 0) or 0
    for layer_idx in range(config.num_hidden_layers, config.num_hidden_layers + num_extra_layers):
        ops.append(Drop(f"model.layers.{layer_idx}.", is_prefix=True))

    return ops
