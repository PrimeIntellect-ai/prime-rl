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
    PrefixRename,
    Rename,
    Stack,
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
                Stack(stacked=f"{prefix}.mlp.experts.w1", item=f"{prefix}.mixer.experts.{{e}}.up_proj.weight"),
                Stack(stacked=f"{prefix}.mlp.experts.w2", item=f"{prefix}.mixer.experts.{{e}}.down_proj.weight"),
            ],
        ),
        Synthetic(f"{prefix}.mlp.experts.w3", factory=_empty_w3(prefix)),
        PrefixRename(f"{prefix}.mixer.shared_experts.", f"{prefix}.mlp.shared_expert."),
        PrefixRename(f"{prefix}.mixer.fc1_latent_proj.", f"{prefix}.mlp.fc1_latent_proj."),
        PrefixRename(f"{prefix}.mixer.fc2_latent_proj.", f"{prefix}.mlp.fc2_latent_proj."),
    ]


def _layer_op(prefix: str) -> ConvOp:
    """One uniform op for any layer: detect its type from a signature key and
    dispatch. No ``layers_block_type`` needed — the unified HF ``mixer.``
    namespace is disambiguated by which sub-key is present (and, on the way
    back, by which prime namespace is present, so the predicates work both
    directions). Mamba/attention keep a bulk ``PrefixRename`` (robust to params
    we didn't enumerate); MoE needs its specific ops (and the gated
    ``Synthetic`` w3, which is why this is a Conditional rather than a plain
    catch-all)."""
    is_attention = lambda sd: (  # noqa: E731
        f"{prefix}.mixer.q_proj.weight" in sd or f"{prefix}.self_attn.q_proj.weight" in sd
    )
    is_moe = lambda sd: f"{prefix}.mixer.gate.weight" in sd or f"{prefix}.mlp.router.gate" in sd  # noqa: E731
    return Conditional(
        is_attention,
        then=[PrefixRename(f"{prefix}.mixer.", f"{prefix}.self_attn.")],
        else_=[
            Conditional(
                is_moe,
                then=_moe_layer_ops(prefix),
                else_=[PrefixRename(f"{prefix}.mixer.", f"{prefix}.mamba.")],
            )
        ],
    )


def conversion_chain(config) -> list[ConvOp]:
    """Uniform per-layer dispatch — no ``layers_block_type`` required."""
    ops: list[ConvOp] = [
        # Global. Listed first so the backbone<->model swap is played LAST on the
        # way back (everything must be in model.* form before re-prefixing).
        PrefixRename("backbone.", "model."),
        Drop("mtp.", is_prefix=True),
        Rename("model.embeddings.weight", "model.embed_tokens.weight"),
        Rename("model.norm_f.weight", "model.norm.weight"),
    ]
    ops.extend(_layer_op(f"model.layers.{i}") for i in range(config.num_hidden_layers))
    return ops
