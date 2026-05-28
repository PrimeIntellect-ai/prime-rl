## a bit of context here, this basically copy AutoModelForCausalLM from transformers, but use our own model instead

import importlib
from collections import OrderedDict
from dataclasses import dataclass

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.olmo3.configuration_olmo3 import Olmo3Config
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from prime_rl.trainer.models.afmoe.configuration_afmoe import AfmoeConfig
from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from prime_rl.trainer.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from prime_rl.trainer.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from prime_rl.trainer.models.laguna.configuration_laguna import LagunaConfig
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, cast_float_and_contiguous
from prime_rl.trainer.models.minimax_m2.configuration_minimax_m2 import MiniMaxM2Config
from prime_rl.trainer.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from prime_rl.trainer.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from prime_rl.trainer.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig


@dataclass(frozen=True)
class _LazyModelClass:
    module_path: str
    class_name: str

    def load(self) -> type:
        module = importlib.import_module(self.module_path)
        return getattr(module, self.class_name)


def _lazy_model_cls(module_path: str, class_name: str) -> _LazyModelClass:
    return _LazyModelClass(module_path, class_name)


def _resolve_model_cls(model_cls: _LazyModelClass | type) -> type:
    if isinstance(model_cls, _LazyModelClass):
        return model_cls.load()
    return model_cls


class _PrimeRLLazyAutoMapping(_LazyAutoMapping):
    def __getitem__(self, key: type[PretrainedConfig]) -> type:
        return _resolve_model_cls(super().__getitem__(key))

    def get(self, key: type[PretrainedConfig], default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def values(self) -> list[type]:
        return [_resolve_model_cls(model_cls) for model_cls in super().values()]

    def items(self) -> list[tuple[type[PretrainedConfig], type]]:
        return [(config_cls, _resolve_model_cls(model_cls)) for config_cls, model_cls in super().items()]


# Make custom config discoverable by AutoConfig
AutoConfig.register("afmoe", AfmoeConfig, exist_ok=True)
AutoConfig.register("glm4_moe", Glm4MoeConfig, exist_ok=True)
AutoConfig.register("glm_moe_dsa", GlmMoeDsaConfig, exist_ok=True)
AutoConfig.register("laguna", LagunaConfig, exist_ok=True)
AutoConfig.register("minimax_m2", MiniMaxM2Config, exist_ok=True)
AutoConfig.register("nemotron_h", NemotronHConfig, exist_ok=True)
AutoConfig.register("qwen3_moe", Qwen3MoeConfig, exist_ok=True)
AutoConfig.register("qwen3_5_moe_text", Qwen3_5MoeConfig, exist_ok=True)
# GptOssConfig is just HF's class - already registered by transformers, no override needed.

_CUSTOM_CAUSAL_LM_MAPPING = _PrimeRLLazyAutoMapping(CONFIG_MAPPING_NAMES, OrderedDict())
_CUSTOM_CAUSAL_LM_MAPPING.register(
    LlamaConfig, _lazy_model_cls("prime_rl.trainer.models.llama", "LlamaForCausalLM"), exist_ok=True
)
_CUSTOM_CAUSAL_LM_MAPPING.register(
    Qwen3Config, _lazy_model_cls("prime_rl.trainer.models.qwen3", "Qwen3ForCausalLM"), exist_ok=True
)
_CUSTOM_CAUSAL_LM_MAPPING.register(
    AfmoeConfig, _lazy_model_cls("prime_rl.trainer.models.afmoe", "AfmoeForCausalLM"), exist_ok=True
)
_CUSTOM_CAUSAL_LM_MAPPING.register(
    Glm4MoeConfig, _lazy_model_cls("prime_rl.trainer.models.glm4_moe", "Glm4MoeForCausalLM"), exist_ok=True
)
_CUSTOM_CAUSAL_LM_MAPPING.register(
    GlmMoeDsaConfig, _lazy_model_cls("prime_rl.trainer.models.glm_moe_dsa", "GlmMoeDsaForCausalLM"), exist_ok=True
)
_CUSTOM_CAUSAL_LM_MAPPING.register(
    LagunaConfig, _lazy_model_cls("prime_rl.trainer.models.laguna", "LagunaForCausalLM"), exist_ok=True
)
_CUSTOM_CAUSAL_LM_MAPPING.register(
    MiniMaxM2Config, _lazy_model_cls("prime_rl.trainer.models.minimax_m2", "MiniMaxM2ForCausalLM"), exist_ok=True
)
_CUSTOM_CAUSAL_LM_MAPPING.register(
    NemotronHConfig, _lazy_model_cls("prime_rl.trainer.models.nemotron_h", "NemotronHForCausalLM"), exist_ok=True
)
_CUSTOM_CAUSAL_LM_MAPPING.register(
    Olmo3Config, _lazy_model_cls("prime_rl.trainer.models.olmo3", "Olmo3ForCausalLM"), exist_ok=True
)
_CUSTOM_CAUSAL_LM_MAPPING.register(
    Qwen3MoeConfig, _lazy_model_cls("prime_rl.trainer.models.qwen3_moe", "Qwen3MoeForCausalLM"), exist_ok=True
)
_CUSTOM_CAUSAL_LM_MAPPING.register(
    Qwen3_5MoeConfig,
    _lazy_model_cls("prime_rl.trainer.models.qwen3_5_moe", "Qwen3_5MoeForCausalLM"),
    exist_ok=True,
)
_CUSTOM_CAUSAL_LM_MAPPING.register(
    GptOssConfig, _lazy_model_cls("prime_rl.trainer.models.gpt_oss", "GptOssForCausalLM"), exist_ok=True
)
_CUSTOM_CAUSAL_LM_MAPPING.register(
    Gemma4TextConfig, _lazy_model_cls("prime_rl.trainer.models.gemma4", "Gemma4ForCausalLM"), exist_ok=True
)


class AutoModelForCausalLMPrimeRL(_BaseAutoModelClass):
    _model_mapping = _CUSTOM_CAUSAL_LM_MAPPING


AutoModelForCausalLMPrimeRL = auto_class_update(AutoModelForCausalLMPrimeRL, head_doc="causal language modeling")


def supports_custom_impl(model_config: PretrainedConfig) -> bool:
    """Check if the model configuration supports the custom PrimeRL implementation.

    Args:
        model_config: The model configuration to check.

    Returns:
        True if the model supports custom implementation, False otherwise.
    """
    return type(model_config) in _CUSTOM_CAUSAL_LM_MAPPING


# Mapping from HF composite VLM model_type to custom PrimeRL class.
# Used by get_model() to dispatch VLMs that have a custom text model implementation.
# Points to the same unified class — the config drives text-only vs VLM behavior.
_CUSTOM_VLM_MAPPING: dict[str, _LazyModelClass] = {
    "gemma4": _lazy_model_cls("prime_rl.trainer.models.gemma4", "Gemma4ForConditionalGeneration"),
    "qwen3_5_moe": _lazy_model_cls("prime_rl.trainer.models.qwen3_5_moe", "Qwen3_5MoeForCausalLM"),
}


def get_custom_vlm_cls(model_config: PretrainedConfig) -> type | None:
    """Return the custom PrimeRL VLM class for this config, or None if unsupported."""
    model_cls = _CUSTOM_VLM_MAPPING.get(getattr(model_config, "model_type", None))
    return _resolve_model_cls(model_cls) if model_cls is not None else None


__all__ = [
    "AutoModelForCausalLMPrimeRL",
    "PreTrainedModelPrimeRL",
    "supports_custom_impl",
    "get_custom_vlm_cls",
    "PrimeLmOutput",
    "cast_float_and_contiguous",
]
