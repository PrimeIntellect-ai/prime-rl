"""Declarative HF<->prime conversion chain for MiniMax M2.

Equivalence to the legacy imperative converter
(:mod:`prime_rl.trainer.models.minimax_m2.converting_minimax_m2`) is checked in
``tests/unit/train/models/conversions/test_minimax_m2_chain.py``.
"""

from __future__ import annotations

from prime_rl.trainer.models.conversion_chains import _routed_experts_op
from prime_rl.trainer.models.conversion_ops import ConvOp, Drop, Rename

# HF stores per-expert projections under the literal names w1/w2/w3 (nn.Linear,
# so each carries a `.weight` suffix); prime stacks them as w1/w2/w3.
_MINIMAX_PROJ_ORDER = (("w1", "w1"), ("w2", "w2"), ("w3", "w3"))


def build_minimax_m2_chain(num_layers: int) -> list[ConvOp]:
    ops: list[ConvOp] = []
    for i in range(num_layers):
        p = f"model.layers.{i}"
        ops.append(Rename(f"{p}.block_sparse_moe.gate.weight", f"{p}.mlp.router.gate.weight"))
        ops.append(Rename(f"{p}.block_sparse_moe.e_score_correction_bias", f"{p}.mlp.expert_bias"))
        ops.append(
            _routed_experts_op(
                p,
                hf_experts="block_sparse_moe.experts",
                tt_experts="mlp.experts",
                proj_order=_MINIMAX_PROJ_ORDER,
            )
        )
        ops.append(Drop(f"{p}.mlp.tokens_per_expert"))
    return ops
