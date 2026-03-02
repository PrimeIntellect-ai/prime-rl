"""Vision-Language Model (VLM) support utilities.

This module provides a single source of truth for supported VLM models.
"""

import fnmatch
from pathlib import Path

# Whitelist of supported VLM model patterns (supports wildcards)
# Add new patterns here as they are tested and supported
SUPPORTED_VLM_PATTERNS = [
    "Qwen/Qwen3-VL*",
]

# model_type values from HuggingFace configs that indicate a VLM
SUPPORTED_VLM_MODEL_TYPES = {"qwen3_vl"}


def is_vlm_model(model_name: str) -> bool:
    """Check if a model is a supported vision-language model.

    Checks the model name against known patterns first. For local paths,
    also reads the config to check the model_type (needed for mini/custom
    checkpoints that don't match the HF naming convention).

    Args:
        model_name: The model name or path (e.g., "Qwen/Qwen3-VL-4B-Instruct")

    Returns:
        True if the model matches a supported VLM pattern or model_type
    """
    model_name_lower = model_name.lower()
    if any(fnmatch.fnmatch(model_name_lower, pattern.lower()) for pattern in SUPPORTED_VLM_PATTERNS):
        return True

    # For local paths or custom HF repos, check the config's model_type
    config_path = Path(model_name) / "config.json"
    if config_path.exists():
        import json

        with open(config_path) as f:
            config_data = json.load(f)
        return config_data.get("model_type", "") in SUPPORTED_VLM_MODEL_TYPES

    return False
