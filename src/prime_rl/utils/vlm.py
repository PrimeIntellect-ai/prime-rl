"""Vision-Language Model (VLM) support utilities.

This module provides a single source of truth for supported VLM models.
"""

import fnmatch
from dataclasses import dataclass

import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig

# Whitelist of supported VLM model patterns (supports wildcards)
# Add new patterns here as they are tested and supported
SUPPORTED_VLM_PATTERNS = [
    "Qwen/Qwen3-VL*",
    "Qwen/Qwen3.5*",
]

DEFAULT_LAYER_PREFIX = "model.layers."
DEFAULT_VISION_ENCODER_ATTR = None  # auto-detect


@dataclass
class VLMModelInfo:
    """Per-model-type VLM metadata."""

    layer_prefix: str
    vision_encoder_attr: str


# Central registry: model_type -> VLM metadata.
# Add new VLM model types here — this is the single source of truth.
VLM_REGISTRY: dict[str, VLMModelInfo] = {
    "qwen3_vl": VLMModelInfo(layer_prefix="model.language_model.layers.", vision_encoder_attr="model.visual"),
    "qwen3_5": VLMModelInfo(layer_prefix="model.language_model.layers.", vision_encoder_attr="model.visual"),
    "qwen3_5_moe": VLMModelInfo(layer_prefix="model.language_model.layers.", vision_encoder_attr="model.visual"),
}

# Derived from the registry — used by is_vlm_config()
SUPPORTED_VLM_MODEL_TYPES = set(VLM_REGISTRY)

# Known vision encoder attribute names across VLM families (checked in order)
_KNOWN_VISION_ATTRS = [
    "model.visual",  # Qwen3-VL, Qwen3.5
    "model.vision_tower",  # LLaVA
    "model.vision_model",  # Idefics3 / SmolVLM
    "visual",  # Qwen2-VL (top-level)
]


def get_layer_prefix(model_config: PretrainedConfig, override: str | None = None) -> str:
    """Return the layer key prefix for a model config.

    Args:
        model_config: The model's HF config.
        override: Explicit prefix from user config. Takes precedence over registry.
    """
    if override is not None:
        return override
    model_type = getattr(model_config, "model_type", None)
    info = VLM_REGISTRY.get(model_type)
    if info is not None:
        return info.layer_prefix
    return DEFAULT_LAYER_PREFIX


def get_vision_encoder(model: nn.Module, override: str | None = None) -> nn.Module | None:
    """Get the vision encoder component from a VLM model.

    Args:
        model: The full model.
        override: Explicit dotted attribute path (e.g. 'model.my_vision').
                  Takes precedence over registry and auto-detection.
    """
    if override is not None:
        return _resolve_attr(model, override)

    # Try registry first
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    info = VLM_REGISTRY.get(model_type) if model_type else None
    if info is not None:
        result = _resolve_attr(model, info.vision_encoder_attr)
        if result is not None:
            return result

    # Fall back to probing known attribute names
    for attr_path in _KNOWN_VISION_ATTRS:
        result = _resolve_attr(model, attr_path)
        if result is not None:
            return result

    return None


def _resolve_attr(obj, dotted_path: str):
    """Resolve a dotted attribute path like 'model.visual' on an object."""
    for part in dotted_path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def is_vlm_model(model_name: str) -> bool:
    """Check if a model is a supported vision-language model by name pattern.

    Args:
        model_name: The model name or path (e.g., "Qwen/Qwen3-VL-4B-Instruct")

    Returns:
        True if the model matches a supported VLM pattern
    """
    model_name_lower = model_name.lower()
    return any(fnmatch.fnmatch(model_name_lower, pattern.lower()) for pattern in SUPPORTED_VLM_PATTERNS)


def is_vlm_config(model_config: PretrainedConfig) -> bool:
    """Check if a loaded model config is a VLM by its model_type.

    This catches VLMs loaded from local paths where the name doesn't match
    the hub patterns.
    """
    return getattr(model_config, "model_type", None) in SUPPORTED_VLM_MODEL_TYPES


def resolve_is_vlm(vlm_flag: bool | None, model_name: str) -> bool:
    """Resolve whether the model is a VLM using the explicit config flag or auto-detection.

    Args:
        vlm_flag: Explicit config value. None means auto-detect.
        model_name: Model name for auto-detection fallback.
    """
    if vlm_flag is not None:
        return vlm_flag
    return is_vlm_model(model_name)
