"""Declarative HF<->prime conversion chain for NemotronH.

NemotronH is the most involved conversion: a unified HF ``mixer`` namespace is
split into prime's ``mamba`` / ``self_attn`` / ``mlp`` by layer type, the
checkpoint uses a ``backbone.`` prefix, and the MoE router bias is shifted by
its per-tensor min on the way in (and intentionally *not* restored on the way
out — a lossy roundtrip the chain reproduces via a :class:`MapValue` whose
backward is the identity). Experts are up/down only, with no gate projection.
"""

from __future__ import annotations

from prime_rl.trainer.models.conversion_ops import (
    Conditional,
    ConvOp,
    Drop,
    MapValue,
    PrefixRename,
    Rename,
    Stack,
    key_present,
)


def _moe_layer_ops(prefix: str) -> list[ConvOp]:
    return [
        # Router gate and selection bias.
        # plus the load-balancing bias which is shifted by its min on the way in
        # and not undone on the way out (MapValue backward = identity).
        Rename(f"{prefix}.mixer.gate.weight", f"{prefix}.mlp.router.gate.weight"),
        Rename(f"{prefix}.mixer.gate.e_score_correction_bias", f"{prefix}.mlp.router.selection_bias"),
        MapValue(
            f"{prefix}.mlp.router.selection_bias",
            forward=lambda x: x - x.min(),
            backward=lambda x: x,
        ),
        # Experts are up/down only. HF is either per-expert weights or
        # a 3-D fused-at-experts-level up_proj/down_proj. Backward always emits
        # per-expert (the predicate's HF key is absent in prime -> else branch).
        Conditional(
            predicate=key_present(f"{prefix}.mixer.experts.up_proj"),
            then=[
                Rename(f"{prefix}.mixer.experts.up_proj", f"{prefix}.mlp.experts.up_proj"),
                Rename(f"{prefix}.mixer.experts.down_proj", f"{prefix}.mlp.experts.down_proj"),
            ],
            else_=[
                Stack(
                    stacked=f"{prefix}.mlp.experts.up_proj",
                    item=f"{prefix}.mixer.experts.{{e}}.up_proj.weight",
                ),
                Stack(
                    stacked=f"{prefix}.mlp.experts.down_proj",
                    item=f"{prefix}.mixer.experts.{{e}}.down_proj.weight",
                ),
            ],
        ),
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
    we didn't enumerate); MoE needs its specific ops."""
    is_attention = lambda sd: (  # noqa: E731
        f"{prefix}.mixer.q_proj.weight" in sd or f"{prefix}.self_attn.q_proj.weight" in sd
    )
    is_moe = lambda sd: (  # noqa: E731
        f"{prefix}.mixer.gate.weight" in sd or f"{prefix}.mlp.router.gate.weight" in sd
    )
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
