"""HF<->prime weight conversion for MiniMax M2, as a declarative op chain.

Per layer: router ``block_sparse_moe.gate.weight`` <-> ``mlp.router.gate.weight``,
the ``e_score_correction_bias`` <-> ``mlp.expert_bias``, and the routed experts
(per-expert w1/w2/w3 ``nn.Linear`` <-> stacked w1/w2/w3). The prime-only runtime
buffer ``mlp.tokens_per_expert`` is dropped on the way to HF.
"""

from __future__ import annotations

from prime_rl.trainer.models.conversion_ops import ConvOp, Drop, Rename, routed_experts_op

# HF stores per-expert projections under the literal names w1/w2/w3 (nn.Linear,
# so each carries a `.weight` suffix); prime stacks them as w1/w2/w3.
_MINIMAX_PROJ_ORDER = (("w1", "w1"), ("w2", "w2"), ("w3", "w3"))


def conversion_chain(config) -> list[ConvOp]:
    ops: list[ConvOp] = []
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        ops.append(Rename(f"{p}.block_sparse_moe.gate.weight", f"{p}.mlp.router.gate.weight"))
        ops.append(Rename(f"{p}.block_sparse_moe.e_score_correction_bias", f"{p}.mlp.expert_bias"))
        ops.append(
            routed_experts_op(
                p,
                hf_experts="block_sparse_moe.experts",
                tt_experts="mlp.experts",
                proj_order=_MINIMAX_PROJ_ORDER,
            )
        )
        ops.append(Drop(f"{p}.mlp.tokens_per_expert"))
    return ops
