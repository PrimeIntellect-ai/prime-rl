## a bit of context here, this basically copy AutoModelForCausalLM from transformers, but use our own model instead

from collections import OrderedDict

from transformers import AutoConfig
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.llama.configuration_llama import LlamaConfig

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig, Glm4MoeForCausalLM
from prime_rl.trainer.models.llama import LlamaForCausalLM
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

# Make custom config discoverable by AutoConfig
AutoConfig.register("glm4_moe", Glm4MoeConfig, exist_ok=True)
AutoConfig.register("qwen3_moe", Qwen3MoeConfig, exist_ok=True)

_CUSTOM_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, OrderedDict())
_CUSTOM_CAUSAL_LM_MAPPING.register(LlamaConfig, LlamaForCausalLM, exist_ok=True)
_CUSTOM_CAUSAL_LM_MAPPING.register(Glm4MoeConfig, Glm4MoeForCausalLM, exist_ok=True)
_CUSTOM_CAUSAL_LM_MAPPING.register(Qwen3MoeConfig, Qwen3MoeForCausalLM, exist_ok=True)


class AutoModelForCausalLMPrimeRL(_BaseAutoModelClass):
    _model_mapping = _CUSTOM_CAUSAL_LM_MAPPING


AutoModelForCausalLMPrimeRL = auto_class_update(AutoModelForCausalLMPrimeRL, head_doc="causal language modeling")


__all__ = ["AutoModelForCausalLMPrimeRL", "PreTrainedModelPrimeRL"]
