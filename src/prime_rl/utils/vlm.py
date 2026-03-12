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

# model_type values that correspond to composite VLM configs
SUPPORTED_VLM_MODEL_TYPES = {
    "qwen3_5_moe",
    "qwen2_5_vl",
    "qwen3_vl",
}


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
