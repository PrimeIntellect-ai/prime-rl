"""HF<->prime weight conversion for Laguna, as a declarative op chain.

Laguna specifics:

* Router: HF ``mlp.gate.weight`` maps to prime ``mlp.router.gate.weight``.
* Expert bias: the HF input is *either* ``mlp.experts.e_score_correction_bias``
  *or* ``mlp.gate.e_score_correction_bias`` (the former wins when both are
  present); it maps to prime ``mlp.expert_bias``. Backward always emits the
  canonical ``mlp.experts.e_score_correction_bias``.
* Routed experts: HF per-expert ``mlp.experts.{e}.{gate,down,up}_proj.weight``
  *or* the fused ``mlp.experts.gate_up_proj`` / ``down_proj`` layout stack into
  prime ``mlp.experts.{w1,w2,w3}`` (w1=gate, w2=down, w3=up).
* Shared expert: the HF input is *either* ``mlp.shared_expert.*`` *or*
  ``mlp.shared_experts.*`` (singular wins); it maps to prime
  ``shared_expert.{w1,w2,w3}.weight`` (no ``mlp.`` prefix, keeps ``.weight``).
  Backward always emits the singular ``mlp.shared_expert.*``.
* Prime-only ``mlp.tokens_per_expert`` is dropped on the way back to HF.
"""

from __future__ import annotations

from prime_rl.trainer.models.conversion_ops import (
    GATE_DOWN_UP,
    Conditional,
    ConvOp,
    Drop,
    Rename,
    routed_experts_op,
)


def conversion_chain(config) -> list[ConvOp]:
    ops: list[ConvOp] = []
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"

        # Router gate.
        ops.append(Rename(f"{p}.mlp.gate.weight", f"{p}.mlp.router.gate.weight"))

        # Expert bias. The base Rename handles the canonical `experts.` input and
        # the (always-singular) backward; the conditional adds the forward-only
        # `gate.` alternate, firing only when the canonical input is absent so it
        # matches the imperative's "experts wins" preference.
        ops.append(Rename(f"{p}.mlp.experts.e_score_correction_bias", f"{p}.mlp.expert_bias"))
        ops.append(
            Conditional(
                predicate=lambda sd, p=p: (
                    f"{p}.mlp.gate.e_score_correction_bias" in sd
                    and f"{p}.mlp.experts.e_score_correction_bias" not in sd
                ),
                then=[Rename(f"{p}.mlp.gate.e_score_correction_bias", f"{p}.mlp.expert_bias")],
            )
        )

        # Routed experts (per-expert or fused gate_up).
        ops.append(routed_experts_op(p, hf_experts="mlp.experts", tt_experts="mlp.experts", fused=True))

        # Shared expert. Base Renames handle the singular input + singular
        # backward; the conditional adds the forward-only plural alternate.
        for wn, hf_proj in GATE_DOWN_UP:
            ops.append(Rename(f"{p}.mlp.shared_expert.{hf_proj}.weight", f"{p}.shared_expert.{wn}.weight"))
        ops.append(
            Conditional(
                predicate=lambda sd, p=p: (
                    f"{p}.mlp.shared_experts.gate_proj.weight" in sd
                    and f"{p}.mlp.shared_expert.gate_proj.weight" not in sd
                ),
                then=[
                    Rename(f"{p}.mlp.shared_experts.{hf_proj}.weight", f"{p}.shared_expert.{wn}.weight")
                    for wn, hf_proj in GATE_DOWN_UP
                ],
            )
        )

        # Prime-only runtime buffer.
        ops.append(Drop(f"{p}.mlp.tokens_per_expert"))
    return ops
