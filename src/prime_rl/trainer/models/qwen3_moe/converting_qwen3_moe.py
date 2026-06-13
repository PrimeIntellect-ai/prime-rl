"""HF<->prime weight conversion for Qwen3-MoE, as a declarative op chain.

Per layer: router ``mlp.gate.weight`` <-> ``mlp.router.gate.weight`` and the
routed experts (per-expert gate/down/up <-> stacked w1/w2/w3, with the fused
transformers-v5 ``gate_up_proj`` input also accepted). No shared experts.
"""

from __future__ import annotations

from prime_rl.trainer.models.conversion_ops import ConvOp, Rename, routed_experts_op


def conversion_chain(config) -> list[ConvOp]:
    ops: list[ConvOp] = []
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        ops.append(Rename(f"{p}.mlp.gate.weight", f"{p}.mlp.router.gate.weight"))
        ops.append(routed_experts_op(p, hf_experts="mlp.experts", tt_experts="mlp.experts", fused=True))
    return ops
