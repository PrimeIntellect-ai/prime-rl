"""Reuse trainer configuration, conversion chains and dtype policy without a model instance."""

from types import SimpleNamespace

from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from prime_rl.trainer.models import _CUSTOM_CAUSAL_LM_BY_MODEL_TYPE

CUSTOM_MODEL_TYPES = frozenset(_CUSTOM_CAUSAL_LM_BY_MODEL_TYPE)


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def resolve_config(config):
    """Build a real config from a dict without weights or remote code.

    Unregistered model types (trusted custom code) fall back to a plain
    namespace so callers treat them as opaque dicts.
    """
    if not isinstance(config, dict):
        return config
    model_type = config.get("model_type")
    if model_type in CONFIG_MAPPING:
        return AutoConfig.for_model(model_type, **{key: value for key, value in config.items() if key != "model_type"})
    return _namespace(config)


def conversion_chain(config) -> list:
    """The trainer's HF <-> PrimeRL conversion ops for this config."""
    config = resolve_config(config)
    model_cls = _CUSTOM_CAUSAL_LM_BY_MODEL_TYPE.get(getattr(config, "model_type", None))
    return model_cls.conversion_chain(config) if model_cls else []


def keep_in_fp32(model_type: str, key: str) -> bool:
    """Whether the trainer stores this tensor in FP32 during weight transfer."""
    model_cls = _CUSTOM_CAUSAL_LM_BY_MODEL_TYPE.get(model_type)
    return bool(model_cls and model_cls.keep_in_fp32_for_weight_transfer(key))


__all__ = ["CUSTOM_MODEL_TYPES", "conversion_chain", "keep_in_fp32", "resolve_config"]
