"""On-disk <-> PrimeRL weight conversion for DeepSeek V4.

"On-disk" means the safetensors keys DeepSeek publishes for
`deepseek-ai/DeepSeek-V4-Flash-0731`, listed in its `model.safetensors.index.json`: a compact
naming (`attn`, `ffn`, `wkv`, `wq_a`, `hc_attn_base`, per-expert `w1`/`w2`/`w3`) that carries no
`model.` prefix on any of its 72317 keys.

Nothing else translates those names. The pinned `transformers` ships no DeepSeek V4 model and no
`conversion_mapping` entry for one, and prime-rl reads the raw safetensors directly for DCP
sharding rather than going through `from_pretrained`, so the chain below is the only mapping
between the published checkpoint and prime-rl's module tree.

It runs both ways: forward for `load_dcp_from_hf`, and reversed (`convert_to_hf`) for the weight
broadcast, whose output vLLM's own loader reads back under the published names.
"""

from __future__ import annotations

from prime_rl.trainer.models.conversion_ops import (
    ConvOp,
    Drop,
    PrefixRename,
    Rename,
    routed_experts_op,
)


def _layer_ops(layer_idx: int, layer_type: str) -> list[ConvOp]:
    """One layer's published names -> prime-rl's module names, one hop per key."""
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
        # Attention's own kv_norm keeps its leaf name; only the parent module differs.
        Rename(f"{attn}.kv_norm.weight", f"{prime_attn}.kv_norm.weight"),
        Rename(f"{attn}.wo_a.weight", f"{prime_attn}.o_a_proj.weight"),
        Rename(f"{attn}.wo_b.weight", f"{prime_attn}.o_b_proj.weight"),
        Rename(f"{attn}.attn_sink", f"{prime_attn}.sinks"),
        Rename(f"{moe}.gate.weight", f"{prime_moe}.router.gate.weight"),
        Rename(f"{moe}.gate.bias", f"{prime_moe}.router.selection_bias"),
        Rename(f"{moe}.gate.tid2eid", f"{prime_moe}.router.tid2eid"),
        Rename(f"{moe}.shared_experts.w1.weight", f"{prime_moe}.shared_expert.gate_proj.weight"),
        Rename(f"{moe}.shared_experts.w2.weight", f"{prime_moe}.shared_expert.down_proj.weight"),
        Rename(f"{moe}.shared_experts.w3.weight", f"{prime_moe}.shared_expert.up_proj.weight"),
        # The routed experts' per-expert w1/w2/w3 stack into prime's per-expert-batched tensors,
        # named as in every other prime-rl MoE.
        routed_experts_op(
            p,
            hf_experts="ffn.experts",
            prime_experts="mlp.experts",
            proj_order=(("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")),
        ),
    ]
    if layer_type in ("compressed_sparse_attention", "heavily_compressed_attention"):
        comp, prime_comp = f"{attn}.compressor", f"{prime_attn}.compressor"
        ops += [
            Rename(f"{comp}.wkv.weight", f"{prime_comp}.kv_proj.weight"),
            Rename(f"{comp}.wgate.weight", f"{prime_comp}.gate_proj.weight"),
            Rename(f"{comp}.norm.weight", f"{prime_comp}.kv_norm.weight"),
            Rename(f"{comp}.ape", f"{prime_comp}.position_bias"),
        ]
    if layer_type == "compressed_sparse_attention":
        # PrimeRL hangs the indexer off the CSA compressor rather than beside it, so its whole
        # subtree moves one level in. Its own (much narrower) compressor nests inside either way.
        idx, prime_idx = f"{attn}.indexer", f"{prime_attn}.compressor.indexer"
        ops += [
            Rename(f"{idx}.wq_b.weight", f"{prime_idx}.q_b_proj.weight"),
            Rename(f"{idx}.weights_proj.weight", f"{prime_idx}.scorer.weights_proj.weight"),
            Rename(f"{idx}.compressor.wkv.weight", f"{prime_idx}.compressor.kv_proj.weight"),
            Rename(f"{idx}.compressor.wgate.weight", f"{prime_idx}.compressor.gate_proj.weight"),
            Rename(f"{idx}.compressor.norm.weight", f"{prime_idx}.compressor.kv_norm.weight"),
            Rename(f"{idx}.compressor.ape", f"{prime_idx}.compressor.position_bias"),
        ]
    return ops


def conversion_chain(config) -> list[ConvOp]:
    # Neither HF nor prime-rl instantiates the multi-token-prediction heads a V4 checkpoint
    # ships. They sit at the top level (`mtp.0.hc_attn_base`, ...), never nested inside a layer.
    ops: list[ConvOp] = [
        Drop("mtp.", is_prefix=True),
        Rename("head.weight", "lm_head.weight"),
    ]
    for layer_idx in range(config.num_hidden_layers):
        ops.extend(_layer_ops(layer_idx, config.layer_types[layer_idx]))
    # Every op above is written in the bare on-disk spelling, so the `model.` prefix that
    # prime-rl's module tree carries goes on last: the non-layer parameters individually, and
    # everything under `layers.` in one reparenting pass.
    ops += [
        Rename("embed.weight", "model.embed_tokens.weight"),
        Rename("norm.weight", "model.norm.weight"),
        Rename("hc_head_fn", "model.hc_head.hc_fn"),
        Rename("hc_head_base", "model.hc_head.hc_base"),
        Rename("hc_head_scale", "model.hc_head.hc_scale"),
        PrefixRename("layers.", "model.layers."),
    ]
    return ops
