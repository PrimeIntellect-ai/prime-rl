"""Vision-Language Model (VLM) support utilities.

This module provides a single source of truth for supported VLM models.
"""

import fnmatch

# Whitelist of supported VLM model patterns (supports wildcards)
# Add new patterns here as they are tested and supported
SUPPORTED_VLM_PATTERNS = [
    "Qwen/Qwen3-VL*",
]


def is_vlm_model(model_name: str) -> bool:
    """Check if a model is a supported vision-language model.

    Args:
        model_name: The model name or path (e.g., "Qwen/Qwen3-VL-4B-Instruct")

    Returns:
        True if the model matches a supported VLM pattern
    """
    model_name_lower = model_name.lower()
    return any(fnmatch.fnmatch(model_name_lower, pattern.lower()) for pattern in SUPPORTED_VLM_PATTERNS)
