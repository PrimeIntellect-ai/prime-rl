"""Declarative HF<->prime conversion chain for GLM-4 MoE.

Mirrors :mod:`prime_rl.trainer.models.conversion_chains`; equivalence to the
legacy imperative converter (``converting_glm4_moe``) is checked in
``tests/unit/train/models/conversions/test_glm4_moe_chain.py``.

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

The hf<->tt MoE conversion is shared verbatim with GLM-MoE-DSA, so the
per-layer op builder lives here and is reused by that model's chain.
"""

from __future__ import annotations

from prime_rl.trainer.models.conversion_chains import _routed_experts_op
from prime_rl.trainer.models.conversion_ops import ConvOp, Drop, Rename, SqueezeLeading

# prime w1=gate, w2=down, w3=up.
_GATE_DOWN_UP = (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj"))


def glm_moe_layer_ops(layer_idx: int) -> list[ConvOp]:
    """The hf<->tt MoE ops for one GLM-style layer.

    Shared by GLM-4 MoE and GLM-MoE-DSA (their hf<->tt MoE conversion is
    identical; DSA's attention/MLA weights pass through untouched)."""
    p = f"model.layers.{layer_idx}.mlp"
    ops: list[ConvOp] = [
        Rename(f"{p}.gate.weight", f"{p}.router.gate.weight"),
        Rename(f"{p}.gate.e_score_correction_bias", f"{p}.expert_bias"),
        _routed_experts_op(
            f"model.layers.{layer_idx}",
            hf_experts="mlp.experts",
            tt_experts="mlp.experts",
            fused=True,
        ),
    ]
    for wn, hf_proj in _GATE_DOWN_UP:
        # SqueezeLeading is backward-only and must run *after* the rename has
        # restored the HF key, so it precedes the Rename in the forward list
        # (apply_tt_to_hf plays the chain in reverse).
        ops.append(SqueezeLeading(f"{p}.shared_experts.{hf_proj}.weight"))
        ops.append(Rename(f"{p}.shared_experts.{hf_proj}.weight", f"{p}.shared_expert.{wn}"))
    ops.append(Drop(f"{p}.tokens_per_expert"))
    return ops


def build_glm4_moe_chain(num_layers: int) -> list[ConvOp]:
    ops: list[ConvOp] = []
    for i in range(num_layers):
        ops.extend(glm_moe_layer_ops(i))
    return ops
