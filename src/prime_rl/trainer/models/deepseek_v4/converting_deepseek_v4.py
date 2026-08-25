"""HF<->PrimeRL weight conversion for DeepSeek V4.

Real DeepSeek V4 checkpoints (what `save_pretrained` writes, and what a Hub download
contains) are **not** in `transformers`' own native module-attribute naming: `transformers`
carries a generic, bidirectional checkpoint-conversion registry
(`transformers.conversion_mapping`, `"deepseek_v4"` entry) that HF applies automatically
inside `from_pretrained`/`save_pretrained` to translate between DeepSeek's compact on-disk
names (`attn`, `ffn`, `wkv`, `wq_a`, `hc_attn_base`, per-expert `w1`/`w2`/`w3`, ...) and the
names `DeepseekV4ForCausalLM`'s own `nn.Module` tree actually uses (`self_attn`, `mlp`,
`kv_proj`, `q_a_proj`, `attn_hc.base`, fused `gate_up_proj`/`down_proj`, ...). prime-rl's own
loading path reads the raw on-disk state dict directly (for DCP sharding) and never goes
through `from_pretrained`, so it has to replicate that on-disk -> HF-native step itself before
the small remaining HF-native -> PrimeRL delta below (confirmed empirically against a real
saved checkpoint from `scripts/mini_moe.py --arch deepseek_v4`, not derivable from the
in-memory `state_dict()` parity test alone, which only ever exercised the HF-native side).

The genuinely PrimeRL-specific delta, once on HF-native names, is small: PrimeRL's shared
`MoE` owns the router and the aux-loss-free load-balancing bias one level above where HF hangs
them (off the router itself), and names its shared expert in the singular. Attention and its
compressors, the hyper-connections, and the routed experts' fused `gate_up_proj` / `down_proj`
already match HF-native shapes exactly once the on-disk step above has run.

The two MoE layer types have different key sets: a hash layer carries `mlp.tid2eid` and no
`mlp.expert_bias`, a standard one the other way round. Every op is present-guarded, so the
same list is emitted for both.
"""

from __future__ import annotations

from prime_rl.trainer.models.conversion_ops import Concatenate, ConvOp, Drop, PrefixRename, Rename, Stack


def _on_disk_attn_ops(layer_idx: int, layer_type: str) -> list[ConvOp]:
    """DeepSeek's on-disk attention naming -> `transformers`-native `self_attn.*`."""
    p = f"model.layers.{layer_idx}"
    ops: list[ConvOp] = [
        PrefixRename(f"{p}.attn.", f"{p}.self_attn."),
        Rename(f"{p}.self_attn.wkv.weight", f"{p}.self_attn.kv_proj.weight"),
        Rename(f"{p}.self_attn.norm.weight", f"{p}.self_attn.kv_norm.weight"),
        Rename(f"{p}.self_attn.q_norm.weight", f"{p}.self_attn.q_a_norm.weight"),
        Rename(f"{p}.self_attn.wq_a.weight", f"{p}.self_attn.q_a_proj.weight"),
        Rename(f"{p}.self_attn.wq_b.weight", f"{p}.self_attn.q_b_proj.weight"),
        Rename(f"{p}.self_attn.wo_a.weight", f"{p}.self_attn.o_a_proj.weight"),
        Rename(f"{p}.self_attn.wo_b.weight", f"{p}.self_attn.o_b_proj.weight"),
        Rename(f"{p}.self_attn.attn_sink", f"{p}.self_attn.sinks"),
        Rename(f"{p}.attn_norm.weight", f"{p}.input_layernorm.weight"),
        Rename(f"{p}.ffn_norm.weight", f"{p}.post_attention_layernorm.weight"),
        Rename(f"{p}.hc_attn_fn", f"{p}.attn_hc.fn"),
        Rename(f"{p}.hc_attn_base", f"{p}.attn_hc.base"),
        Rename(f"{p}.hc_attn_scale", f"{p}.attn_hc.scale"),
        Rename(f"{p}.hc_ffn_fn", f"{p}.ffn_hc.fn"),
        Rename(f"{p}.hc_ffn_base", f"{p}.ffn_hc.base"),
        Rename(f"{p}.hc_ffn_scale", f"{p}.ffn_hc.scale"),
    ]
    if layer_type in ("compressed_sparse_attention", "heavily_compressed_attention"):
        c = f"{p}.self_attn.compressor"
        ops += [
            Rename(f"{c}.wkv.weight", f"{c}.kv_proj.weight"),
            Rename(f"{c}.wgate.weight", f"{c}.gate_proj.weight"),
            Rename(f"{c}.norm.weight", f"{c}.kv_norm.weight"),
            Rename(f"{c}.ape", f"{c}.position_bias"),
        ]
    if layer_type == "compressed_sparse_attention":
        # The indexer's own (much narrower) internal compressor: on disk it nests as
        # `indexer.compressor.*`; PrimeRL's `DeepseekV4Indexer` inherits the compressor base
        # directly, so its `kv_proj`/`gate_proj`/`kv_norm`/`position_bias` sit straight on
        # `compressor.indexer`, one nesting level shallower than the on-disk layout.
        idx = f"{p}.self_attn.indexer"
        cidx = f"{c}.indexer"
        ops += [
            Rename(f"{idx}.wq_b.weight", f"{cidx}.q_b_proj.weight"),
            Rename(f"{idx}.weights_proj.weight", f"{cidx}.scorer.weights_proj.weight"),
            Rename(f"{idx}.compressor.wkv.weight", f"{cidx}.kv_proj.weight"),
            Rename(f"{idx}.compressor.wgate.weight", f"{cidx}.gate_proj.weight"),
            Rename(f"{idx}.compressor.norm.weight", f"{cidx}.kv_norm.weight"),
            Rename(f"{idx}.compressor.ape", f"{cidx}.position_bias"),
        ]
    return ops


def _on_disk_moe_ops(layer_idx: int) -> list[ConvOp]:
    """DeepSeek's on-disk `ffn.*` naming -> `transformers`-native `mlp.*`, including fusing
    the on-disk per-expert `w1`/`w2`/`w3` into PrimeRL's (and HF-native's) fused
    `gate_up_proj` / `down_proj`."""
    p = f"model.layers.{layer_idx}"
    experts = f"{p}.mlp.experts"
    shared = f"{p}.mlp.shared_experts"
    return [
        PrefixRename(f"{p}.ffn.", f"{p}.mlp."),
        Rename(f"{p}.mlp.gate.bias", f"{p}.mlp.gate.e_score_correction_bias"),
        Rename(f"{shared}.w1.weight", f"{shared}.gate_proj.weight"),
        Rename(f"{shared}.w2.weight", f"{shared}.down_proj.weight"),
        Rename(f"{shared}.w3.weight", f"{shared}.up_proj.weight"),
        Stack(stacked=f"{experts}._gate_stack", item=f"{experts}.{{e}}.w1.weight"),
        Stack(stacked=f"{experts}._up_stack", item=f"{experts}.{{e}}.w3.weight"),
        Concatenate(
            combined=f"{experts}.gate_up_proj",
            parts=[f"{experts}._gate_stack", f"{experts}._up_stack"],
            dim=1,
        ),
        Stack(stacked=f"{experts}.down_proj", item=f"{experts}.{{e}}.w2.weight"),
    ]


def _layer_ops(layer_idx: int, layer_type: str) -> list[ConvOp]:
    prefix = f"model.layers.{layer_idx}.mlp"
    ops = _on_disk_attn_ops(layer_idx, layer_type) + _on_disk_moe_ops(layer_idx)
    ops += [
        Rename(f"{prefix}.gate.weight", f"{prefix}.router.gate.weight"),
        Rename(f"{prefix}.gate.e_score_correction_bias", f"{prefix}.expert_bias"),
        Rename(f"{prefix}.gate.tid2eid", f"{prefix}.tid2eid"),
        PrefixRename(f"{prefix}.shared_experts.", f"{prefix}.shared_expert."),
    ]
    return ops


def conversion_chain(config) -> list[ConvOp]:
    # Neither HF nor prime-rl instantiates the multi-token-prediction heads a V4 checkpoint
    # ships. HF drops them with `_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]`,
    # which matches at either nesting depth, hence the two prefixes.
    ops: list[ConvOp] = [
        Drop("mtp.", is_prefix=True),
        Drop("model.mtp.", is_prefix=True),
        Rename("head.weight", "lm_head.weight"),
    ]
    for layer_idx in range(config.num_hidden_layers):
        ops.extend(_layer_ops(layer_idx, config.layer_types[layer_idx]))
    return ops
