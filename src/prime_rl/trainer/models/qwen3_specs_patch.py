"""ConversionSpec patch for HF Qwen3 (dense) so PI's NIXLWeightBroadcast works.

PI's ``models/__init__.py`` only registers a custom ``PreTrainedModelPrimeRL``
subclass for ``glm_moe_dsa``, ``glm4_moe``, ``qwen3_moe``, ``qwen3_5_moe``,
``llama``, ``minimax_m2``, ``afmoe``, ``nemotron_h``. Plain dense Qwen3
(0.6B, 1.7B, 4B, 8B, 14B, 32B) falls through to HF's ``Qwen3ForCausalLM``,
which doesn't define ``conversion_specs(layer_idx)`` — so
``TransportPlan`` crashes at construction with ``AttributeError``.

This module monkey-patches HF Qwen3 to expose ``conversion_specs`` and
``non_layer_conversion_specs`` so PI's NIXL transport works on dense
Qwen3 unchanged. Imported from ``prime_rl/trainer/models/__init__.py``
side-effect-only.

The spec table mirrors vLLM's Qwen3 weight loading:

* ``q_proj`` + ``k_proj`` + ``v_proj`` → ``qkv_proj`` (concat dim 0).
* ``gate_proj`` + ``up_proj`` → ``gate_up_proj`` (concat dim 0).
* Everything else passthrough (o_proj, down_proj, input_layernorm,
  post_attention_layernorm, q_norm, k_norm, embed_tokens, model.norm,
  lm_head if not tied).

All BF16, no FP8. For an FP8 variant of Qwen3, swap ``QuantizationSpec``
default for the relevant specs.
"""

from __future__ import annotations

import torch

from prime_rl.trainer.models.conversion_spec import ConversionSpec, QuantizationSpec


# Per-layer specs for HF Qwen3. dst suffix is appended to "model.layers.{i}.".
_QWEN3_LAYER: tuple[ConversionSpec, ...] = (
    # Pre-attention RMSNorm
    ConversionSpec(
        "input_layernorm.weight",
        ("input_layernorm.weight",),
    ),
    # Pre-MLP RMSNorm
    ConversionSpec(
        "post_attention_layernorm.weight",
        ("post_attention_layernorm.weight",),
    ),
    # Q/K/V → fused qkv_proj. vLLM's Qwen3 model concatenates (q, k, v)
    # along dim 0; PI's ConversionSpec preserves source order, so the
    # tuple order here matters.
    ConversionSpec(
        "self_attn.qkv_proj.weight",
        (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
        ),
        cat_dim=0,
    ),
    # Output projection (no fusion)
    ConversionSpec(
        "self_attn.o_proj.weight",
        ("self_attn.o_proj.weight",),
    ),
    # Qwen3-specific RMSNorm on per-head Q (after q_proj split)
    ConversionSpec(
        "self_attn.q_norm.weight",
        ("self_attn.q_norm.weight",),
    ),
    # Qwen3-specific RMSNorm on per-head K
    ConversionSpec(
        "self_attn.k_norm.weight",
        ("self_attn.k_norm.weight",),
    ),
    # MLP gate + up → fused gate_up_proj (gate first, then up)
    ConversionSpec(
        "mlp.gate_up_proj.weight",
        (
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
        ),
        cat_dim=0,
    ),
    # MLP down projection
    ConversionSpec(
        "mlp.down_proj.weight",
        ("mlp.down_proj.weight",),
    ),
)


def _qwen3_conversion_specs(self, layer_idx: int) -> tuple[ConversionSpec, ...]:
    """Per-layer spec table — Qwen3 dense has no MoE so it's identical
    for every layer. ``layer_idx`` accepted for API parity with
    PI's ``GlmMoeDsaForCausalLM``."""
    return _QWEN3_LAYER


def _qwen3_non_layer_conversion_specs(self) -> tuple[ConversionSpec, ...]:
    """Non-per-layer tensors. Without these NIXL never updates them on
    inference, causing KL drift as trainer gradients advance them but
    inference stays at initial load (the lesson PI's PR #2326 iter8
    documented)."""
    specs: tuple[ConversionSpec, ...] = (
        ConversionSpec(
            "model.embed_tokens.weight",
            ("model.embed_tokens.weight",),
        ),
        ConversionSpec(
            "model.norm.weight",
            ("model.norm.weight",),
        ),
    )
    if not getattr(self.config, "tie_word_embeddings", False):
        specs = specs + (
            ConversionSpec(
                "lm_head.weight",
                ("lm_head.weight",),
            ),
        )
    return specs


def _patch_hf_qwen3_class() -> None:
    """Add `conversion_specs` and `non_layer_conversion_specs` to HF
    `Qwen3ForCausalLM`. Idempotent; safe to call repeatedly.

    Also patches the FSDP wrapper class if accessible — though FSDP2's
    fully_shard typically forwards `__getattr__` to the underlying
    module, so the bare class patch is usually enough.
    """
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM  # type: ignore
    except ImportError:
        # transformers without Qwen3 — nothing to patch.
        return

    # Idempotent guard
    if getattr(Qwen3ForCausalLM, "_prime_rl_qwen3_patched", False):
        return

    Qwen3ForCausalLM.conversion_specs = _qwen3_conversion_specs  # type: ignore[attr-defined]
    Qwen3ForCausalLM.non_layer_conversion_specs = _qwen3_non_layer_conversion_specs  # type: ignore[attr-defined]
    Qwen3ForCausalLM._prime_rl_qwen3_patched = True  # type: ignore[attr-defined]


_patch_hf_qwen3_class()
