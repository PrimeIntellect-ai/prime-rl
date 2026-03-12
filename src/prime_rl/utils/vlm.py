"""Vision-Language Model (VLM) support utilities.

This module provides a single source of truth for supported VLM models.
"""

import fnmatch

from transformers.configuration_utils import PretrainedConfig

# Whitelist of supported VLM model patterns (supports wildcards)
# Add new patterns here as they are tested and supported
SUPPORTED_VLM_PATTERNS = [
    "Qwen/Qwen3-VL*",
    "Qwen/Qwen3.5*",
]

DEFAULT_LAYER_PREFIX = "model.layers."

# Per-VLM registry: model_type -> layer key prefix.
# VLM models nest the text decoder under a different prefix
# (e.g. 'model.language_model.layers.' instead of 'model.layers.').
# Add new VLM model types here — this is the single source of truth.
VLM_REGISTRY: dict[str, str] = {
    "qwen3_vl": "model.language_model.layers.",
    "qwen3_5": "model.language_model.layers.",
    "qwen3_5_moe": "model.language_model.layers.",
}

# Derived from the registry — used by is_vlm_config()
SUPPORTED_VLM_MODEL_TYPES = set(VLM_REGISTRY)


def get_layer_prefix(model_config: PretrainedConfig) -> str:
    """Return the layer key prefix for a model config.

    VLM models nest their text decoder under a different prefix
    (e.g. 'model.language_model.layers.') while standard decoder
    models use 'model.layers.'.
    """
    model_type = getattr(model_config, "model_type", None)
    return VLM_REGISTRY.get(model_type, DEFAULT_LAYER_PREFIX)


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
