"""Model-free conversion mappings shared by training and checkpoint export."""

from importlib import import_module
from types import SimpleNamespace

from prime_rl.utils import glm_weight_conversion
from prime_rl.utils.conversion_ops import ConvOp

_MODULES = {
    "afmoe": "afmoe",
    "deepseek_v4": "deepseek_v4",
    "gpt_oss": "gpt_oss",
    "laguna": "laguna",
    "minimax_m2": "minimax_m2",
    "nemotron_h": "nemotron_h",
    "qwen3_moe": "qwen3_moe",
    "qwen3_5": "qwen3_5",
    "qwen3_5_moe": "qwen3_5",
    "qwen3_5_text": "qwen3_5",
    "qwen3_5_moe_text": "qwen3_5",
    "qwen4_exp": "qwen3_8_flash_next",
    "qwen4_exp_text": "qwen3_8_flash_next",
}
CUSTOM_MODEL_TYPES = frozenset({"llama", "qwen3", "glm4_moe", "glm_moe_dsa", *_MODULES})

_CONFIG_CLASSES = {
    "afmoe": ("afmoe", "AfmoeConfig"),
    "deepseek_v4": ("deepseek_v4", "DeepseekV4Config"),
    "gpt_oss": ("gpt_oss", "GptOssConfig"),
    "laguna": ("laguna", "LagunaConfig"),
    "minimax_m2": ("minimax_m2", "MiniMaxM2Config"),
    "nemotron_h": ("nemotron_h", "NemotronHConfig"),
    "qwen3_moe": ("qwen3_moe", "Qwen3MoeConfig"),
    "qwen3_5": ("qwen3_5", "Qwen3_5Config"),
    "qwen3_5_moe": ("qwen3_5", "Qwen3_5MoeConfig"),
    "qwen3_5_text": ("qwen3_5", "Qwen3_5TextConfig"),
    "qwen3_5_moe_text": ("qwen3_5", "Qwen3_5MoeTextConfig"),
    "qwen4_exp": ("qwen3_8_flash_next", "Qwen3_8FlashNextConfig"),
    "qwen4_exp_text": ("qwen3_8_flash_next", "Qwen3_8FlashNextTextConfig"),
    "glm4_moe": ("glm4_moe", "Glm4MoeConfig"),
    "glm_moe_dsa": ("glm_moe_dsa", "GlmMoeDsaConfig"),
}


def _module(model_type: str):
    if model_type in {"glm4_moe", "glm_moe_dsa"}:
        return glm_weight_conversion
    name = _MODULES.get(model_type)
    return import_module(f"{__name__}.{name}") if name else None


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def resolve_config(config):
    """Apply native config defaults without training imports or remote code."""
    if not isinstance(config, dict):
        return config
    model_type = config.get("model_type")
    if model_type in _CONFIG_CLASSES:
        module_name, class_name = _CONFIG_CLASSES[model_type]
        module = import_module(f"{__name__}.configs.{module_name}")
        return getattr(module, class_name)(**config)
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if model_type in CONFIG_MAPPING:
        return AutoConfig.for_model(model_type, **{key: value for key, value in config.items() if key != "model_type"})
    return _namespace(config)


def conversion_chain(config) -> list[ConvOp]:
    config = resolve_config(config)
    module = _module(config.model_type)
    return module.conversion_chain(config) if module else []


def keep_in_fp32(model_type: str, key: str) -> bool:
    module = _module(model_type)
    return module.keep_in_fp32(key) if module else False
