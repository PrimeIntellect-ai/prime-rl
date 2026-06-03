"""Vision-Language Model (VLM) support utilities.

Central registry for VLM model families. All model-specific knowledge
lives here. Add new VLM families by extending VLM_REGISTRY.

For custom models not in the registry, set overrides in config:
    [model.vlm]
    vision_encoder_attr = "model.my_vision"
    language_model_attr = "model.my_lm"
"""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig


@dataclass(frozen=True)
class VLMModelInfo:
    """Per-model-family VLM architecture metadata."""

    vision_encoder_attr: str
    language_model_attr: str


PackedMMPositionStrategy: TypeAlias = Literal["none", "pass_1d"]
PACKED_MM_ATTN_IMPLS = ("flash_attention_2", "flash_attention_3", "fa4")


# Central registry: model_type -> architecture info.
VLM_REGISTRY: dict[str, VLMModelInfo] = {
    "qwen3_vl": VLMModelInfo(vision_encoder_attr="model.visual", language_model_attr="model.language_model"),
    "qwen3_5": VLMModelInfo(vision_encoder_attr="model.visual", language_model_attr="model.language_model"),
    "qwen3_5_moe": VLMModelInfo(vision_encoder_attr="model.visual", language_model_attr="model.language_model"),
    "qwen3_vl_moe": VLMModelInfo(vision_encoder_attr="model.visual", language_model_attr="model.language_model"),
}

# Text-only default
DEFAULT_LAYER_PREFIX = "model.layers."


# ---------------------------------------------------------------------------
# Model component access
# ---------------------------------------------------------------------------


def get_vision_encoder(model: nn.Module, override: str | None = None) -> nn.Module | None:
    """Get the vision encoder module.

    Checks: config override -> registry. Returns None if not found.
    Raises ValueError on a bad config override.
    """
    if override is not None:
        result = _resolve_attr(model, override)
        if result is None:
            raise ValueError(f"vlm.vision_encoder_attr='{override}' does not resolve on this model")
        return result

    info = _get_model_info(model)
    if info is not None:
        return _resolve_attr(model, info.vision_encoder_attr)

    return None


def get_language_model(model: nn.Module, override: str | None = None) -> nn.Module:
    """Get the language model module (the part with transformer layers).

    Checks: config override -> registry -> model.model (text-only default).
    Raises ValueError on a bad config override.
    """
    if override is not None:
        result = _resolve_attr(model, override)
        if result is None:
            raise ValueError(f"vlm.language_model_attr='{override}' does not resolve on this model")
        return result

    info = _get_model_info(model)
    if info is not None:
        result = _resolve_attr(model, info.language_model_attr)
        if result is not None:
            return result

    # Text-only models: language model is directly at model.model
    return model.model


def is_vlm_architecture(model_config: PretrainedConfig) -> bool:
    """Check if the model config belongs to a known VLM architecture."""
    return _get_model_info_from_config(model_config) is not None


def get_packed_mm_position_strategy(model: nn.Module) -> PackedMMPositionStrategy:
    """Return the model's packed multimodal position strategy.

    ``pass_1d`` is intentionally narrow: it means the VLM's language model
    consumes reset 1D ``position_ids`` and derives packed attention boundaries
    from them. HF Qwen-style MRoPE models need model-computed 3D/4-row positions
    and therefore remain ``none`` until a dedicated builder exists.
    """
    for candidate in _iter_wrapped_modules(model):
        strategy = getattr(candidate, "packed_mm_position_strategy", None)
        if strategy in ("none", "pass_1d"):
            return strategy

        model_type = getattr(getattr(candidate, "config", None), "model_type", None)
        if model_type == "qwen3_5_moe" and getattr(candidate, "_is_vlm", False):
            return "pass_1d"

    return "none"


def get_packed_mm_disabled_reasons(
    model: nn.Module,
    *,
    enabled: bool,
    attn_impl: str,
    cp_enabled: bool,
    cp_size: int | None = None,
) -> list[str]:
    """Return reasons multimodal packing should be disabled for this runtime."""
    strategy = get_packed_mm_position_strategy(model)
    reasons = []
    if not enabled:
        reasons.append("trainer.pack_multimodal=false")
    if strategy != "pass_1d":
        reasons.append(f"position_strategy={strategy}")
    if attn_impl not in PACKED_MM_ATTN_IMPLS:
        reasons.append(f"attn={attn_impl}")
    if cp_enabled:
        cp_label = cp_size if cp_size is not None else "enabled"
        reasons.append(f"cp={cp_label}")
    return reasons


def get_layer_prefix(model_config: PretrainedConfig, override: str | None = None) -> str:
    """Return the weight key prefix for language model layers.

    Derived from language_model_attr + '.layers.' for registered VLMs,
    or 'model.layers.' for text-only / unknown models.
    """
    if override is not None:
        return override
    info = _get_model_info_from_config(model_config)
    if info is not None:
        return info.language_model_attr + ".layers."
    return DEFAULT_LAYER_PREFIX


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _get_model_info(model: nn.Module) -> VLMModelInfo | None:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return VLM_REGISTRY.get(model_type) if model_type else None


def _get_model_info_from_config(model_config: PretrainedConfig) -> VLMModelInfo | None:
    model_type = getattr(model_config, "model_type", None)
    return VLM_REGISTRY.get(model_type) if model_type else None


def _resolve_attr(obj, dotted_path: str):
    """Resolve a dotted attribute path like 'model.visual' on an object."""
    for part in dotted_path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _iter_wrapped_modules(model: nn.Module):
    """Yield a module and common wrapper inners without depending on wrapper types."""
    seen: set[int] = set()
    current = model
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "module", None)
