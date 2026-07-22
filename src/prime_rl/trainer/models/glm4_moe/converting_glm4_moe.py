"""HF<->prime weight conversion for GLM-4 MoE, as a declarative op chain.

GLM-4 MoE specifics:

* Router: HF ``mlp.gate.weight`` -> prime ``mlp.router.gate.weight``.
* Expert bias: HF ``mlp.gate.e_score_correction_bias`` -> prime
  ``mlp.expert_bias``.
* Routed experts: HF per-expert ``mlp.experts.{e}.{gate,down,up}_proj.weight``
  stack into prime ``mlp.experts.{w1,w2,w3}`` along dim 0; the fused
  transformers-v5 ``mlp.experts.gate_up_proj`` / ``down_proj`` layout is also
  accepted on the way in (split along dim 1).
* Shared experts: HF ``mlp.shared_experts.{gate,down,up}_proj.weight`` map to
  prime ``mlp.shared_expert.{w1,w2,w3}`` (no ``.weight`` suffix on prime; this
  is a plain rename forward). On the way back to HF, GLM strips a leading
  singleton dim from each shared-expert tensor when present
  (:class:`~prime_rl.trainer.models.conversion_ops.SqueezeLeading`,
  backward-only).
* Prime-only runtime buffer ``mlp.tokens_per_expert`` is dropped on the way
  back to HF.

The per-layer MoE op builder (:func:`glm_moe_layer_ops`) is shared verbatim
with GLM-MoE-DSA, whose attention/MLA weights pass through untouched.
"""

from __future__ import annotations

from prime_rl.trainer.models.conversion_ops import (
    GATE_DOWN_UP,
    ConvOp,
    Drop,
    Rename,
    SqueezeLeading,
    routed_experts_op,
)


def glm_moe_layer_ops(layer_idx: int) -> list[ConvOp]:
    """The HF <-> PrimeRL MoE ops for one GLM-style layer.

    Shared by GLM-4 MoE and GLM-MoE-DSA (their HF <-> PrimeRL MoE conversion is
    identical; DSA's attention/MLA weights pass through untouched)."""
    p = f"model.layers.{layer_idx}.mlp"
    ops: list[ConvOp] = [
        Rename(f"{p}.gate.weight", f"{p}.router.gate.weight"),
        Rename(f"{p}.gate.e_score_correction_bias", f"{p}.expert_bias"),
        routed_experts_op(
            f"model.layers.{layer_idx}",
            hf_experts="mlp.experts",
            prime_experts="mlp.experts",
            fused=True,
        ),
    ]
    for wn, hf_proj in GATE_DOWN_UP:
        # SqueezeLeading is backward-only and must run *after* the rename has
        # restored the HF key, so it precedes the Rename in the forward list
        # (apply_prime_to_hf plays the chain in reverse).
        ops.append(SqueezeLeading(f"{p}.shared_experts.{hf_proj}.weight"))
        ops.append(Rename(f"{p}.shared_experts.{hf_proj}.weight", f"{p}.shared_expert.{wn}"))
    ops.append(Drop(f"{p}.tokens_per_expert"))
    return ops


def conversion_chain(config) -> list[ConvOp]:
    ops: list[ConvOp] = []
    for i in range(config.num_hidden_layers):
        ops.extend(glm_moe_layer_ops(i))
    return ops
