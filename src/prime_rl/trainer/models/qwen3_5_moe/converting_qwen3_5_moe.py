"""HF<->prime weight conversion for Qwen3.5-MoE, as a declarative op chain.

Per layer: router ``mlp.gate.weight`` <-> ``mlp.router.gate.weight`` and the
routed experts from the source layout <-> stacked canonical projections.
"""

from __future__ import annotations

from prime_rl.trainer.models.conversion_ops import ConvOp, Rename, routed_experts_op


def _conversion_chain(config, model_prefix: str) -> list[ConvOp]:
    ops: list[ConvOp] = []
    for i in range(config.num_hidden_layers):
        p = f"{model_prefix}.layers.{i}"
        # Router: mlp.gate.weight -> mlp.router.gate.weight
        ops.append(Rename(f"{p}.mlp.gate.weight", f"{p}.mlp.router.gate.weight"))
        ops.append(routed_experts_op(p, hf_experts="mlp.experts", prime_experts="mlp.experts", fused=True))
    return ops


def conversion_chain(config) -> list[ConvOp]:
    text_config = getattr(config, "text_config", config)
    return _conversion_chain(text_config, "model") + _conversion_chain(text_config, "model.language_model")
