"""HF<->prime weight conversion for Qwen3.5-MoE, as a declarative op chain.

Per layer: the router and shared-expert output gate are nested under their
PrimeRL modules, and routed experts use stacked canonical projections.
"""

from __future__ import annotations

from prime_rl.trainer.models.conversion_ops import ConvOp, Rename, routed_experts_op


def _conversion_chain(config, model_prefix: str) -> list[ConvOp]:
    ops: list[ConvOp] = []
    for i in range(config.num_hidden_layers):
        p = f"{model_prefix}.layers.{i}"
        # Router: mlp.gate.weight -> mlp.router.gate.weight
        ops.append(Rename(f"{p}.mlp.gate.weight", f"{p}.mlp.router.gate.weight"))
        ops.append(Rename(f"{p}.mlp.shared_expert_gate.weight", f"{p}.mlp.shared_expert.output_gate.weight"))
        ops.append(routed_experts_op(p, hf_experts="mlp.experts", prime_experts="mlp.experts", fused=True))
    return ops


def conversion_chain(config) -> list[ConvOp]:
    text_config = getattr(config, "text_config", config)
    return _conversion_chain(text_config, "model") + _conversion_chain(text_config, "model.language_model")
