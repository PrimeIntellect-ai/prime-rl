"""Reuse trainer configuration, conversion chains and dtype policy without a model instance."""

from types import SimpleNamespace

import torch
from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from prime_rl.trainer.models import _CUSTOM_CAUSAL_LM_BY_MODEL_TYPE, get_custom_causal_lm_cls
from prime_rl.utils.export_config import uses_tied_embeddings

CUSTOM_MODEL_TYPES = frozenset(_CUSTOM_CAUSAL_LM_BY_MODEL_TYPE)

Shape = tuple[int, ...]


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


def prime_schema_details(
    config, *, attn_implementation: str | None = None
) -> tuple[dict[str, Shape], dict[str, str], dict[str, torch.dtype]]:
    """Canonical PrimeRL state names, tied aliases and dtypes from a meta-model build."""
    resolved = resolve_config(config)
    model_cls = get_custom_causal_lm_cls(resolved)
    if attn_implementation == "auto":
        # Meta validation never runs attention; use the custom model's default backend.
        attn_implementation = "flash_attention_3"
    with torch.device("meta"):
        model = model_cls._from_config(resolved, attn_implementation=attn_implementation)
    try:
        state = model.state_dict()
        nonmeta = [name for name, tensor in state.items() if tensor.device.type != "meta"]
        if nonmeta:
            raise ValueError(
                f"Custom model_type={resolved.model_type!r} allocates non-meta state "
                f"during schema construction: {nonmeta[:5]}"
            )
        schema = {name: tuple(tensor.shape) for name, tensor in state.items()}
        dtypes = {name: tensor.dtype for name, tensor in state.items()}
        aliases: dict[str, str] = {}
        canonical: dict[int, str] = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            if name not in schema:
                continue
            identity = id(parameter)
            if identity in canonical:
                aliases[name] = canonical[identity]
            else:
                canonical[identity] = name
        if uses_tied_embeddings(resolved):
            embedding = model.get_input_embeddings().weight
            aliases["lm_head.weight"] = next(
                name for name, parameter in model.named_parameters(remove_duplicate=False) if parameter is embedding
            )
        return schema, aliases, dtypes
    finally:
        del model


def conversion_chain(config) -> list:
    """The trainer's HF <-> PrimeRL conversion ops for this config."""
    config = resolve_config(config)
    model_cls = _CUSTOM_CAUSAL_LM_BY_MODEL_TYPE.get(getattr(config, "model_type", None))
    return model_cls.conversion_chain(config) if model_cls else []


def keep_in_fp32(model_type: str, key: str) -> bool:
    """Whether the trainer stores this tensor in FP32 during weight transfer."""
    model_cls = _CUSTOM_CAUSAL_LM_BY_MODEL_TYPE.get(model_type)
    return bool(model_cls and model_cls.keep_in_fp32_for_weight_transfer(key))


__all__ = ["CUSTOM_MODEL_TYPES", "conversion_chain", "keep_in_fp32", "prime_schema_details", "resolve_config"]
