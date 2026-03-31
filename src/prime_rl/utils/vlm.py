"""Vision-Language Model (VLM) support utilities.

Central registry for VLM model families. All model-specific knowledge
lives here. Add new VLM families by extending VLM_REGISTRY.

For custom models not in the registry, set overrides in config:
    [model.vlm]
    vision_encoder_attr = "model.my_vision"
    language_model_attr = "model.my_lm"
"""

from dataclasses import dataclass

import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig


@dataclass(frozen=True)
class VLMModelInfo:
    """Per-model-family VLM architecture metadata."""

    vision_encoder_attr: str
    language_model_attr: str


# Central registry: model_type -> architecture info.
VLM_REGISTRY: dict[str, VLMModelInfo] = {
    "qwen3_vl": VLMModelInfo(vision_encoder_attr="model.visual", language_model_attr="model.language_model"),
    "qwen3_5": VLMModelInfo(vision_encoder_attr="model.visual", language_model_attr="model.language_model"),
    "qwen3_5_moe": VLMModelInfo(vision_encoder_attr="model.visual", language_model_attr="model.language_model"),
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
