## a bit of context here, this basically copy AutoModelForCausalLM from transformers, but use our own model instead

from collections import OrderedDict

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from prime_rl.trainer.models.afmoe import AfmoeConfig, AfmoeForCausalLM
from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig, Glm4MoeForCausalLM
from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
from prime_rl.trainer.models.gpt_oss import GptOssConfig, GptOssForCausalLM
from prime_rl.trainer.models.laguna import LagunaConfig, LagunaForCausalLM
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, cast_float_and_contiguous
from prime_rl.trainer.models.llama import LlamaForCausalLM
from prime_rl.trainer.models.minimax_m2 import MiniMaxM2Config, MiniMaxM2ForCausalLM
from prime_rl.trainer.models.nemotron_h import NemotronHConfig, NemotronHForCausalLM
from prime_rl.trainer.models.qwen3 import Qwen3ForCausalLM
from prime_rl.trainer.models.qwen3_5 import Qwen3_5ForCausalLM
from prime_rl.trainer.models.qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

# Make custom config discoverable by AutoConfig
AutoConfig.register("afmoe", AfmoeConfig, exist_ok=True)
AutoConfig.register("glm4_moe", Glm4MoeConfig, exist_ok=True)
AutoConfig.register("glm_moe_dsa", GlmMoeDsaConfig, exist_ok=True)
AutoConfig.register("laguna", LagunaConfig, exist_ok=True)
AutoConfig.register("minimax_m2", MiniMaxM2Config, exist_ok=True)
AutoConfig.register("nemotron_h", NemotronHConfig, exist_ok=True)
AutoConfig.register("qwen3_moe", Qwen3MoeConfig, exist_ok=True)
AutoConfig.register("qwen3_5_text", Qwen3_5TextConfig, exist_ok=True)
AutoConfig.register("qwen3_5_moe_text", Qwen3_5MoeConfig, exist_ok=True)
# GptOssConfig is just HF's class - already registered by transformers, no override needed.

_CUSTOM_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, OrderedDict())
_CUSTOM_CAUSAL_LM_MODELS: tuple[
    tuple[type[PretrainedConfig], type[PreTrainedModelPrimeRL]],
    ...,
] = (
    (LlamaConfig, LlamaForCausalLM),
    (Qwen3Config, Qwen3ForCausalLM),
    (AfmoeConfig, AfmoeForCausalLM),
    (Glm4MoeConfig, Glm4MoeForCausalLM),
    (GlmMoeDsaConfig, GlmMoeDsaForCausalLM),
    (LagunaConfig, LagunaForCausalLM),
    (MiniMaxM2Config, MiniMaxM2ForCausalLM),
    (NemotronHConfig, NemotronHForCausalLM),
    (Qwen3MoeConfig, Qwen3MoeForCausalLM),
    (Qwen3_5TextConfig, Qwen3_5ForCausalLM),
    (Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM),
    (GptOssConfig, GptOssForCausalLM),
)
for config_cls, model_cls in _CUSTOM_CAUSAL_LM_MODELS:
    _CUSTOM_CAUSAL_LM_MAPPING.register(config_cls, model_cls, exist_ok=True)

_CUSTOM_CAUSAL_LM_BY_MODEL_TYPE = {
    config_cls.model_type: model_cls for config_cls, model_cls in _CUSTOM_CAUSAL_LM_MODELS
}


class AutoModelForCausalLMPrimeRL(_BaseAutoModelClass):
    _model_mapping = _CUSTOM_CAUSAL_LM_MAPPING


AutoModelForCausalLMPrimeRL = auto_class_update(AutoModelForCausalLMPrimeRL, head_doc="causal language modeling")


def get_custom_causal_lm_cls(model_config: PretrainedConfig) -> type[PreTrainedModelPrimeRL]:
    """Resolve the PrimeRL model class from a possibly non-PrimeRL config instance."""
    return _CUSTOM_CAUSAL_LM_BY_MODEL_TYPE[model_config.model_type]


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
_CUSTOM_VLM_MAPPING: dict[str, type] = {
    "qwen3_5": Qwen3_5ForCausalLM,
    "qwen3_5_moe": Qwen3_5MoeForCausalLM,
}


def get_custom_vlm_cls(model_config: PretrainedConfig) -> type | None:
    """Return the custom PrimeRL VLM class for this config, or None if unsupported."""
    return _CUSTOM_VLM_MAPPING.get(getattr(model_config, "model_type", None))


__all__ = [
    "AutoModelForCausalLMPrimeRL",
    "PreTrainedModelPrimeRL",
    "get_custom_causal_lm_cls",
    "supports_custom_impl",
    "get_custom_vlm_cls",
    "PrimeLmOutput",
    "cast_float_and_contiguous",
]
