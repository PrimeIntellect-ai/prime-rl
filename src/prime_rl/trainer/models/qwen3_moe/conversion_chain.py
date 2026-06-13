"""Declarative HF<->prime conversion chain for Qwen3-MoE.

Equivalence to the legacy imperative converter (``converting_qwen3_moe``) is
checked in tests/unit/train/models/conversions/test_qwen3_moe_chain.py.

Per layer: router ``mlp.gate.weight`` <-> ``mlp.router.gate.weight`` and the
routed experts (per-expert gate/down/up -> stacked w1/w2/w3, with the fused
transformers-v5 ``gate_up_proj`` input also accepted). No shared experts."""

from __future__ import annotations

from prime_rl.trainer.models.conversion_chains import _routed_experts_op
from prime_rl.trainer.models.conversion_ops import ConvOp, Rename


def build_qwen3_moe_chain(num_layers: int) -> list[ConvOp]:
    ops: list[ConvOp] = []
    for i in range(num_layers):
        p = f"model.layers.{i}"
        ops.append(Rename(f"{p}.mlp.gate.weight", f"{p}.mlp.router.gate.weight"))
        ops.append(_routed_experts_op(p, hf_experts="mlp.experts", tt_experts="mlp.experts", fused=True))
    return ops
