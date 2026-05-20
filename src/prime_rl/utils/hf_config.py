from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PreTrainedConfig

_YARN_FLOAT_FIELDS = frozenset(
    {
        "attention_factor",
        "beta_fast",
        "beta_slow",
        "factor",
        "mscale",
        "mscale_all_dim",
        "rope_theta",
    }
)

_PATCHED_TRANSFORMERS_CONFIG = False


def _is_int_but_not_bool(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _normalize_rope_mapping(rope_mapping: Any) -> bool:
    if not isinstance(rope_mapping, dict):
        return False

    if any(key in rope_mapping for key in ("rope_type", "type", "factor", "beta_fast", "beta_slow")):
        rope_type = rope_mapping.get("rope_type", rope_mapping.get("type"))
        if rope_type != "yarn":
            return False

        changed = False
        for key in _YARN_FLOAT_FIELDS:
            value = rope_mapping.get(key)
            if _is_int_but_not_bool(value):
                rope_mapping[key] = float(value)
                changed = True
        return changed

    changed = False
    for value in rope_mapping.values():
        changed = _normalize_rope_mapping(value) or changed
    return changed


def normalize_rope_numeric_types(config_dict: dict[str, Any]) -> bool:
    """Normalize semantically-float RoPE fields before Transformers validates configs."""
    changed = False
    changed = _normalize_rope_mapping(config_dict.get("rope_parameters")) or changed
    changed = _normalize_rope_mapping(config_dict.get("rope_scaling")) or changed
    return changed


def patch_transformers_config_rope_numeric_types() -> None:
    """Patch the HF config ingestion boundary to normalize legacy RoPE JSON types."""
    global _PATCHED_TRANSFORMERS_CONFIG
    if _PATCHED_TRANSFORMERS_CONFIG:
        return

    original_get_config_dict = PreTrainedConfig.get_config_dict.__func__

    @classmethod
    def get_config_dict_with_normalized_rope(cls, *args, **kwargs):
        config_dict, remaining_kwargs = original_get_config_dict(cls, *args, **kwargs)
        if isinstance(config_dict, dict):
            normalize_rope_numeric_types(config_dict)
        return config_dict, remaining_kwargs

    PreTrainedConfig.get_config_dict = get_config_dict_with_normalized_rope
    _PATCHED_TRANSFORMERS_CONFIG = True
