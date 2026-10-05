"""Published <-> PrimeRL weight conversion for DeepSeek-V4.1.

The published `deepseek-ai/DeepSeek-V4.1-Flash` keys use DeepSeek's compact naming (`attn`, `ffn`,
`wkv`, `wq_a`, `hc_attn_fn`, per-expert `w1`/`w2`/`w3`, `engram.embed`) with no `model.` prefix.
The vision tower, its aligner and the DSpark draft head (`mtp.*`) are not trained and are dropped,
as is the gate's vision-token routing bias.
"""

from __future__ import annotations

from torch import Tensor

from prime_rl.trainer.models.conversion_ops import ConvOp, Drop, PrefixRename, Rename, routed_experts_op


def _layer_ops(config, layer_idx: int) -> list[ConvOp]:
    p = f"layers.{layer_idx}"
    attn, moe = f"{p}.attn", f"{p}.ffn"
    prime_attn, prime_moe = f"{p}.self_attn", f"{p}.mlp"
    ops: list[ConvOp] = [
        Rename(f"{p}.attn_norm.weight", f"{p}.input_layernorm.weight"),
        Rename(f"{p}.ffn_norm.weight", f"{p}.post_attention_layernorm.weight"),
        Rename(f"{p}.hc_attn_fn", f"{p}.attn_hc.fn"),
        Rename(f"{p}.hc_attn_base", f"{p}.attn_hc.base"),
        Rename(f"{p}.hc_attn_scale", f"{p}.attn_hc.scale"),
        Rename(f"{p}.hc_ffn_fn", f"{p}.ffn_hc.fn"),
        Rename(f"{p}.hc_ffn_base", f"{p}.ffn_hc.base"),
        Rename(f"{p}.hc_ffn_scale", f"{p}.ffn_hc.scale"),
        Rename(f"{attn}.wq_a.weight", f"{prime_attn}.q_a_proj.weight"),
        Rename(f"{attn}.q_norm.weight", f"{prime_attn}.q_a_norm.weight"),
        Rename(f"{attn}.wq_b.weight", f"{prime_attn}.q_b_proj.weight"),
        Rename(f"{attn}.wkv.weight", f"{prime_attn}.kv_proj.weight"),
        Rename(f"{attn}.kv_norm.weight", f"{prime_attn}.kv_norm.weight"),
        Rename(f"{attn}.wo_a.weight", f"{prime_attn}.o_a_proj.weight"),
        Rename(f"{attn}.wo_b.weight", f"{prime_attn}.o_b_proj.weight"),
        Rename(f"{attn}.attn_sink", f"{prime_attn}.sinks"),
        Rename(f"{attn}.compressor.wkv.weight", f"{prime_attn}.compressor.kv_proj.weight"),
        Rename(f"{attn}.compressor.wgate.weight", f"{prime_attn}.compressor.gate_proj.weight"),
        Rename(f"{attn}.compressor.norm.weight", f"{prime_attn}.compressor.kv_norm.weight"),
        Rename(f"{attn}.indexer.wq_b.weight", f"{prime_attn}.indexer.q_b_proj.weight"),
        Rename(f"{attn}.indexer.weights_proj.weight", f"{prime_attn}.indexer.weights_proj.weight"),
        Rename(f"{attn}.indexer.wk.weight", f"{prime_attn}.indexer.k_proj.weight"),
        Rename(f"{attn}.indexer.k_norm.weight", f"{prime_attn}.indexer.k_norm.weight"),
        Rename(f"{moe}.gate.weight", f"{prime_moe}.router.gate.weight"),
        Rename(f"{moe}.gate.bias", f"{prime_moe}.router.selection_bias"),
        Drop(f"{moe}.gate.bias_vl"),
        Rename(f"{moe}.shared_experts.w1.weight", f"{prime_moe}.shared_expert.gate_proj.weight"),
        Rename(f"{moe}.shared_experts.w2.weight", f"{prime_moe}.shared_expert.down_proj.weight"),
        Rename(f"{moe}.shared_experts.w3.weight", f"{prime_moe}.shared_expert.up_proj.weight"),
        routed_experts_op(
            p,
            hf_experts="ffn.experts",
            prime_experts="mlp.experts",
            proj_order=(("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")),
        ),
    ]
    if layer_idx in config.engram_layer_ids:
        # The engrams live beside the decoder layers, not inside them.
        engram = f"{p}.engram"
        prime_engram = f"engrams.{layer_idx}"
        ops += [
            Rename(f"{engram}.embed.weight", f"{prime_engram}.embed.weight"),
            Rename(f"{engram}.wkv.weight", f"{prime_engram}.wkv.weight"),
            Rename(f"{engram}.q_weight", f"{prime_engram}.q_weight"),
            Rename(f"{engram}.k_weight", f"{prime_engram}.k_weight"),
        ]
    return ops


def conversion_chain(config) -> list[ConvOp]:
    text_config = getattr(config, "text_config", config)
    model_prefix = "model."
    ops: list[ConvOp] = [
        Drop("mtp.", is_prefix=True),
        Drop("vision.", is_prefix=True),
        Drop("aligner.", is_prefix=True),
        Drop("image_start"),
        Drop("image_end"),
        Drop("image_newline"),
        Rename("head.weight", "lm_head.weight"),
    ]
    for layer_idx in range(text_config.num_hidden_layers):
        ops.extend(_layer_ops(text_config, layer_idx))
    ops += [
        Rename("embed.weight", f"{model_prefix}embed_tokens.weight"),
        Rename("norm.weight", f"{model_prefix}norm.weight"),
        PrefixRename("layers.", f"{model_prefix}layers."),
        PrefixRename("engrams.", f"{model_prefix}engrams."),
    ]
    return ops


def is_hf_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return any(name.endswith("ffn.gate.weight") or "ffn.shared_experts." in name for name in state_dict)


def is_prime_state_dict(state_dict: dict[str, Tensor]) -> bool:
    # The NCCL weight broadcast converts one layer at a time plus a final group of every
    # non-layer key, so the latter must be recognizable on its own, hence `lm_head`.
    return any(
        name.endswith("mlp.router.gate.weight") or "mlp.shared_expert." in name or name == "lm_head.weight"
        for name in state_dict
    )
