"""Validate HF-native checkpoint layouts using weight-free meta models."""

from collections.abc import Mapping
from copy import deepcopy

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

Shape = tuple[int, ...]


class ModelSchemaUnavailable(ValueError):
    """Transformers has no config or model mapping for this architecture."""


def hf_schema_details(
    config: dict, model_config: dict | None = None
) -> tuple[dict[str, Shape], dict[str, str], dict[str, torch.dtype]]:
    """Return state shapes, tied aliases and dtypes, retaining no tensors.

    Aliases map duplicate parameter names to the first canonical parameter name.
    Native classes never fetch repository files. Custom config/model code is
    executed only with explicit model_config.trust_remote_code authorization;
    only code files are requested, never pretrained model weights.
    """
    attributes = deepcopy(config)
    model_type = attributes.pop("model_type", None)
    if model_type is None:
        raise ValueError("HF-native export requires config.model_type")
    model_config = model_config or {}
    trusted = model_config.get("trust_remote_code", False)
    auto_map = config.get("auto_map") or {}
    source = model_config.get("name")
    code_options = {
        key: model_config[key] for key in ("revision", "code_revision") if model_config.get(key) is not None
    }
    if trusted and auto_map and not source:
        raise ValueError("Trusted custom HF schema construction requires model_config.name")
    if trusted and "AutoConfig" in auto_map:
        config_class = get_class_from_dynamic_module(auto_map["AutoConfig"], source, **code_options)
        hf_config = config_class.from_dict(deepcopy(config))
    else:
        if model_type not in CONFIG_MAPPING:
            raise ModelSchemaUnavailable(
                f"No native Transformers config for model_type={model_type!r}; "
                "custom code requires explicit model_config.name and trust_remote_code=True"
            )
        hf_config = AutoConfig.for_model(model_type, **attributes)
    config_class = type(hf_config)
    remote_auto = next(
        (name for name in ("AutoModelForImageTextToText", "AutoModelForCausalLM") if name in auto_map), None
    )
    with torch.device("meta"):
        if trusted and remote_auto is not None:
            model_class = get_class_from_dynamic_module(auto_map[remote_auto], source, **code_options)
            model = model_class._from_config(hf_config, attn_implementation="eager")
        elif config_class in AutoModelForCausalLM._model_mapping:
            model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=False, attn_implementation="eager")
        elif config_class in AutoModelForImageTextToText._model_mapping:
            model = AutoModelForImageTextToText.from_config(
                hf_config, trust_remote_code=False, attn_implementation="eager"
            )
        else:
            raise ModelSchemaUnavailable(f"No causal-LM or image-text model for model_type={model_type!r}")
    try:
        state = model.state_dict()
        if any(tensor.device.type != "meta" for tensor in state.values()):
            raise ValueError(f"Native model_type={model_type!r} allocates non-meta state during schema construction")
        schema = {name: tuple(tensor.shape) for name, tensor in state.items()}
        dtypes = {name: tensor.dtype for name, tensor in state.items()}
        aliases = {}
        canonical = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            if name not in schema:
                continue
            identity = id(parameter)
            if identity in canonical:
                aliases[name] = canonical[identity]
            else:
                canonical[identity] = name
        return schema, aliases, dtypes
    finally:
        del model


def hf_schema_metadata(config: dict, model_config: dict | None = None) -> tuple[dict[str, Shape], dict[str, str]]:
    """Return state shapes and tied aliases from one weight-free model build."""
    schema, aliases, _ = hf_schema_details(config, model_config)
    return schema, aliases


def expected_hf_schema(config: dict, model_config: dict | None = None) -> dict[str, Shape]:
    """Return every HF state-dict key and shape, including tied aliases."""
    return hf_schema_metadata(config, model_config)[0]


def matches_hf_schema(
    logical_shapes: Mapping[str, Shape], schema: Mapping[str, Shape], aliases: Mapping[str, str]
) -> bool:
    """Compare exact keys/shapes, allowing omitted aliases of present tied weights."""
    if logical_shapes.keys() - schema.keys():
        return False
    for name, shape in schema.items():
        if name in logical_shapes:
            if tuple(logical_shapes[name]) != tuple(shape):
                return False
        elif name not in aliases or aliases[name] not in logical_shapes:
            return False
    return True


def is_hf_identity_layout(logical_shapes: Mapping[str, Shape], config: dict, model_config: dict | None = None) -> bool:
    """Whether canonical checkpoint metadata already has the native HF layout."""
    schema, aliases = hf_schema_metadata(config, model_config)
    return matches_hf_schema(logical_shapes, schema, aliases)
