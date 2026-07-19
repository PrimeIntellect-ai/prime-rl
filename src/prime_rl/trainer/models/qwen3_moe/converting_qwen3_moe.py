"""HF<->PrimeRL weight conversion for Qwen3-MoE."""

from __future__ import annotations

from prime_rl.trainer.models.conversion_ops import ConvOp, Rename, routed_experts_op


def conversion_chain(config) -> list[ConvOp]:
    ops: list[ConvOp] = []
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}"
        ops.append(Rename(f"{prefix}.mlp.gate.weight", f"{prefix}.mlp.router.gate.weight"))
        ops.append(routed_experts_op(prefix, hf_experts="mlp.experts", tt_experts="mlp.experts", fused=True))
    return ops
