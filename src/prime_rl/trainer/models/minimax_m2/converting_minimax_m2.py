"""HF<->prime weight conversion for MiniMax M2, as a declarative op chain.

Per layer: router ``block_sparse_moe.gate.weight`` <-> ``mlp.router.gate.weight``,
the ``e_score_correction_bias`` <-> ``mlp.router.selection_bias``, and the routed experts
(per-expert w1/w2/w3 ``nn.Linear`` <-> stacked gate/down/up projections). The prime-only runtime
buffer ``mlp.tokens_per_expert`` is dropped on the way to HF.
"""

from __future__ import annotations

from prime_rl.trainer.models.conversion_ops import ConvOp, Drop, Rename, routed_experts_op

# HF stores per-expert projections under the literal names w1/w2/w3 (nn.Linear,
# so each carries a `.weight` suffix); PrimeRL uses canonical projection names.
_MINIMAX_PROJ_ORDER = (("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3"))


def conversion_chain(config) -> list[ConvOp]:
    ops: list[ConvOp] = []
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        ops.append(Rename(f"{p}.block_sparse_moe.gate.weight", f"{p}.mlp.router.gate.weight"))
        ops.append(Rename(f"{p}.block_sparse_moe.e_score_correction_bias", f"{p}.mlp.router.selection_bias"))
        ops.append(
            routed_experts_op(
                p,
                hf_experts="block_sparse_moe.experts",
                prime_experts="mlp.experts",
                proj_order=_MINIMAX_PROJ_ORDER,
            )
        )
        ops.append(Drop(f"{p}.mlp.tokens_per_expert"))
    return ops
