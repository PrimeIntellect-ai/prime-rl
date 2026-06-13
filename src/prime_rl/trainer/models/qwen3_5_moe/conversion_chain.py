"""Declarative conversion chain for qwen3_5_moe.

Mirrors :func:`prime_rl.trainer.models.conversion_chains.build_qwen3_moe_chain`
and adds the shared-expert / shared-expert-gate renames specific to qwen3_5_moe.
Equivalence to the legacy imperative converter in ``converting_qwen3_5_moe.py``
is checked in tests/unit/train/models/conversions/test_qwen3_5_moe_chain.py.
"""

from __future__ import annotations

from prime_rl.trainer.models.conversion_chains import _routed_experts_op
from prime_rl.trainer.models.conversion_ops import ConvOp, Rename


def build_qwen3_5_moe_chain(num_layers: int) -> list[ConvOp]:
    ops: list[ConvOp] = []
    for i in range(num_layers):
        p = f"model.layers.{i}"
        # Router: mlp.gate.weight -> mlp.router.gate.weight
        ops.append(Rename(f"{p}.mlp.gate.weight", f"{p}.mlp.router.gate.weight"))
        # Routed experts: per-expert (gate/down/up) or fused gate_up_proj -> w1/w2/w3
        ops.append(_routed_experts_op(p, hf_experts="mlp.experts", tt_experts="mlp.experts", fused=True))
        # Shared expert: mlp.shared_expert.{gate,down,up}_proj.weight -> shared_expert.{w1,w2,w3}.weight
        ops.append(Rename(f"{p}.mlp.shared_expert.gate_proj.weight", f"{p}.shared_expert.w1.weight"))
        ops.append(Rename(f"{p}.mlp.shared_expert.down_proj.weight", f"{p}.shared_expert.w2.weight"))
        ops.append(Rename(f"{p}.mlp.shared_expert.up_proj.weight", f"{p}.shared_expert.w3.weight"))
        # Shared expert gate
        ops.append(Rename(f"{p}.mlp.shared_expert_gate.weight", f"{p}.shared_expert_gate.weight"))
    return ops
