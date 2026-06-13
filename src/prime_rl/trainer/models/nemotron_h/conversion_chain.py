"""Declarative HF<->prime conversion chain for NemotronH.

NemotronH is the most involved conversion: a unified HF ``mixer`` namespace is
split into prime's ``mamba`` / ``self_attn`` / ``mlp`` by layer type, the
checkpoint uses a ``backbone.`` prefix, and the MoE router bias is shifted by
its per-tensor min on the way in (and intentionally *not* restored on the way
out — a lossy roundtrip the chain reproduces via a :class:`MapValue` whose
backward is the identity). Experts are up/down only (no gate); a dummy ``w3``
of shape ``(0,)`` is synthesised for expert-parallel compatibility.
"""

from __future__ import annotations

import torch

from prime_rl.trainer.models.conversion_ops import (
    Conditional,
    ConvOp,
    Drop,
    MapValue,
    MoEExperts,
    PrefixRename,
    Rename,
    Synthetic,
    key_present,
)


def _empty_w3(prefix: str):
    def factory(sd):
        w1 = f"{prefix}.mlp.experts.w1"
        device = sd[w1].device if w1 in sd else "cpu"
        return torch.empty(0, device=device)

    return factory


def _moe_layer_ops(prefix: str) -> list[ConvOp]:
    return [
        # Router: gate.weight -> router.gate (note: prime drops the .weight),
        # plus the load-balancing bias which is shifted by its min on the way in
        # and not undone on the way out (MapValue backward = identity).
        Rename(f"{prefix}.mixer.gate.weight", f"{prefix}.mlp.router.gate"),
        Rename(f"{prefix}.mixer.gate.e_score_correction_bias", f"{prefix}.mlp.router.e_score_correction_bias"),
        MapValue(
            f"{prefix}.mlp.router.e_score_correction_bias",
            forward=lambda x: x - x.min(),
            backward=lambda x: x,
        ),
        # Experts: w1=up, w2=down (no gate). HF is either per-expert weights or
        # a 3-D fused-at-experts-level up_proj/down_proj. Backward always emits
        # per-expert (the predicate's HF key is absent in prime -> else branch).
        Conditional(
            predicate=key_present(f"{prefix}.mixer.experts.up_proj"),
            then=[
                Rename(f"{prefix}.mixer.experts.up_proj", f"{prefix}.mlp.experts.w1"),
                Rename(f"{prefix}.mixer.experts.down_proj", f"{prefix}.mlp.experts.w2"),
            ],
            else_=[
                MoEExperts(
                    projs={
                        f"{prefix}.mlp.experts.w1": f"{prefix}.mixer.experts.{{e}}.up_proj.weight",
                        f"{prefix}.mlp.experts.w2": f"{prefix}.mixer.experts.{{e}}.down_proj.weight",
                    }
                )
            ],
        ),
        Synthetic(f"{prefix}.mlp.experts.w3", factory=_empty_w3(prefix)),
        PrefixRename(f"{prefix}.mixer.shared_experts.", f"{prefix}.mlp.shared_expert."),
        PrefixRename(f"{prefix}.mixer.fc1_latent_proj.", f"{prefix}.mlp.fc1_latent_proj."),
        PrefixRename(f"{prefix}.mixer.fc2_latent_proj.", f"{prefix}.mlp.fc2_latent_proj."),
    ]


def build_nemotron_h_chain(layers_block_type: list[str]) -> list[ConvOp]:
    """``layers_block_type[i]`` is one of ``"mamba"``, ``"attention"``, ``"moe"``."""
    ops: list[ConvOp] = [
        # Global. Listed first so the backbone<->model swap is played LAST on the
        # way back (everything must be in model.* form before re-prefixing).
        PrefixRename("backbone.", "model."),
        Drop("mtp.", is_prefix=True),
        Rename("model.embeddings.weight", "model.embed_tokens.weight"),
        Rename("model.norm_f.weight", "model.norm.weight"),
    ]
    for i, layer_type in enumerate(layers_block_type):
        prefix = f"model.layers.{i}"
        if layer_type == "mamba":
            ops.append(PrefixRename(f"{prefix}.mixer.", f"{prefix}.mamba."))
        elif layer_type == "attention":
            ops.append(PrefixRename(f"{prefix}.mixer.", f"{prefix}.self_attn."))
        elif layer_type == "moe":
            ops.extend(_moe_layer_ops(prefix))
        else:
            raise ValueError(f"unknown NemotronH layer type {layer_type!r}")
    return ops
