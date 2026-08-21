"""HF<->PrimeRL weight conversion for DeepSeek V4.

Only the MoE block moves. prime-rl's shared `MoE` owns the router and the aux-loss-free
load-balancing bias one level above where HF hangs them (off the router itself), and names
its shared expert in the singular. Everything else already carries HF's names: attention
and its compressors, the hyper-connections, and -- unlike every other prime-rl MoE -- the
routed experts, whose fused `gate_up_proj` / `down_proj` match HF's shapes exactly.

The two MoE layer types have different key sets: a hash layer carries `mlp.tid2eid` and no
`mlp.expert_bias`, a standard one the other way round. Every op is present-guarded, so the
same list is emitted for both.
"""

from __future__ import annotations

from prime_rl.trainer.models.conversion_ops import ConvOp, Drop, PrefixRename, Rename


def _layer_ops(layer_idx: int) -> list[ConvOp]:
    prefix = f"model.layers.{layer_idx}.mlp"
    return [
        Rename(f"{prefix}.gate.weight", f"{prefix}.router.gate.weight"),
        Rename(f"{prefix}.gate.e_score_correction_bias", f"{prefix}.expert_bias"),
        Rename(f"{prefix}.gate.tid2eid", f"{prefix}.tid2eid"),
        PrefixRename(f"{prefix}.shared_experts.", f"{prefix}.shared_expert."),
    ]


def conversion_chain(config) -> list[ConvOp]:
    # Neither HF nor prime-rl instantiates the multi-token-prediction heads a V4 checkpoint
    # ships. HF drops them with `_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]`,
    # which matches at either nesting depth, hence the two prefixes.
    ops: list[ConvOp] = [Drop("mtp.", is_prefix=True), Drop("model.mtp.", is_prefix=True)]
    for layer_idx in range(config.num_hidden_layers):
        ops.extend(_layer_ops(layer_idx))
    return ops
