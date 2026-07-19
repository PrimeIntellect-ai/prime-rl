"""HF<->prime weight conversion for Qwen3.5-MoE, as a declarative op chain.

Per layer: router ``mlp.gate.weight`` <-> ``mlp.router.gate.weight``, the routed
experts (per-expert gate/down/up <-> stacked w1/w2/w3, with the fused
transformers-v5 ``gate_up_proj`` input also accepted), the shared expert
(``mlp.shared_expert.{gate,down,up}_proj`` <-> ``shared_expert.{w1,w2,w3}``) and
its gate (``mlp.shared_expert_gate`` <-> ``shared_expert_gate``).
"""

from __future__ import annotations

from prime_rl.trainer.models.conversion_ops import ConvOp, Rename, routed_experts_op


def _conversion_chain(config, model_prefix: str) -> list[ConvOp]:
    ops: list[ConvOp] = []
    for i in range(config.num_hidden_layers):
        p = f"{model_prefix}.layers.{i}"
        # Router: mlp.gate.weight -> mlp.router.gate.weight
        ops.append(Rename(f"{p}.mlp.gate.weight", f"{p}.mlp.router.gate.weight"))
        # Routed experts: per-expert (gate/down/up) or fused gate_up_proj -> w1/w2/w3
        ops.append(routed_experts_op(p, hf_experts="mlp.experts", tt_experts="mlp.experts", fused=True))
        # Shared expert: mlp.shared_expert.{gate,down,up}_proj.weight -> shared_expert.{w1,w2,w3}.weight
        ops.append(Rename(f"{p}.mlp.shared_expert.gate_proj.weight", f"{p}.shared_expert.w1.weight"))
        ops.append(Rename(f"{p}.mlp.shared_expert.down_proj.weight", f"{p}.shared_expert.w2.weight"))
        ops.append(Rename(f"{p}.mlp.shared_expert.up_proj.weight", f"{p}.shared_expert.w3.weight"))
        # Shared expert gate
        ops.append(Rename(f"{p}.mlp.shared_expert_gate.weight", f"{p}.shared_expert_gate.weight"))
    return ops


def conversion_chain(config) -> list[ConvOp]:
    text_config = getattr(config, "text_config", config)
    return _conversion_chain(text_config, "model") + _conversion_chain(text_config, "model.language_model")
