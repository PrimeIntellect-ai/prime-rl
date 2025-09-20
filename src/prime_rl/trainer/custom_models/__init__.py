## a bit of context here, this basically copy AutoModelForCausalLM from transformers, but use our own model instead

from collections import OrderedDict

from transformers import AutoConfig
from transformers.models.auto.auto_factory import (
    _BaseAutoModelClass,
    _get_model_class,
    _LazyAutoMapping,
    auto_class_update,
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.llama.configuration_llama import LlamaConfig

from prime_rl.trainer.custom_models.glm import Glm4MoeConfig, Glm4MoeForCausalLM, Glm4MoeModel
from prime_rl.trainer.custom_models.llama import LlamaForCausalLM, LlamaModel

# Make custom config discoverable by AutoConfig
AutoConfig.register("glm4_moe", Glm4MoeConfig, exist_ok=True)

# Minimal PRIME-RL-only mappings (use OrderedDict, not dict)
_CUSTOM_BASE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, OrderedDict())
_CUSTOM_BASE_MAPPING.register(LlamaConfig, LlamaModel, exist_ok=True)
_CUSTOM_BASE_MAPPING.register(Glm4MoeConfig, Glm4MoeModel, exist_ok=True)

_CUSTOM_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, OrderedDict())
_CUSTOM_CAUSAL_LM_MAPPING.register(LlamaConfig, LlamaForCausalLM, exist_ok=True)
_CUSTOM_CAUSAL_LM_MAPPING.register(Glm4MoeConfig, Glm4MoeForCausalLM, exist_ok=True)


class AutoModelPrimeRL(_BaseAutoModelClass):
    _model_mapping = _CUSTOM_BASE_MAPPING


AutoModelPrimeRL = auto_class_update(AutoModelPrimeRL)


class AutoModelForCausalLMPrimeRL(_BaseAutoModelClass):
    _model_mapping = _CUSTOM_CAUSAL_LM_MAPPING


AutoModelForCausalLMPrimeRL = auto_class_update(AutoModelForCausalLMPrimeRL, head_doc="causal language modeling")


def get_model_cls(config):
    return _get_model_class(config, AutoModelForCausalLMPrimeRL._model_mapping)


__all__ = [
    "AutoModelPrimeRL",
    "AutoModelForCausalLMPrimeRL",
    "get_model_cls",
    "LlamaModel",
    "LlamaForCausalLM",
    "Glm4MoeConfig",
    "Glm4MoeModel",
    "Glm4MoeForCausalLM",
]
