"""Declarative HF<->prime conversion chain for GLM-MoE-DSA.

Equivalence to the legacy imperative converter (``converting_glm_moe_dsa``) is
checked in ``tests/unit/train/models/conversions/test_glm_moe_dsa_chain.py``.

GLM-MoE-DSA's hf<->tt MoE conversion is identical to GLM-4 MoE (router rename,
expert-bias rename, fused-or-per-expert routed experts, shared experts with the
same backward leading-dim squeeze, and dropping the prime-only
``tokens_per_expert`` buffer), so it reuses
:func:`prime_rl.trainer.models.glm4_moe.conversion_chain.glm_moe_layer_ops`.

The attention / MLA (DSA) weights (e.g. ``self_attn.q_a_proj.weight``,
``self_attn.kv_a_proj_with_mqa.weight``) are *not* renamed in the hf<->tt
direction — they pass through untouched. (The ``convert_tt_layer_to_vllm_kernel``
fusion is a separate, lossy terminal path and is intentionally not encoded
here.)
"""

from __future__ import annotations

from prime_rl.trainer.models.conversion_ops import ConvOp
from prime_rl.trainer.models.glm4_moe.conversion_chain import glm_moe_layer_ops


def build_glm_moe_dsa_chain(num_layers: int) -> list[ConvOp]:
    ops: list[ConvOp] = []
    for i in range(num_layers):
        ops.extend(glm_moe_layer_ops(i))
    return ops
