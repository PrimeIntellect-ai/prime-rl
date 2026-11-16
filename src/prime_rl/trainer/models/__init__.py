## a bit of context here, this basically copy AutoModelForCausalLM from transformers, but use our own model instead

import json
from collections import OrderedDict
from pathlib import Path

from huggingface_hub import _CACHED_NO_EXIST, hf_hub_download, try_to_load_from_cache
from huggingface_hub.errors import RemoteEntryNotFoundError
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from prime_rl.trainer.models.afmoe import AfmoeConfig, AfmoeForCausalLM
from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig, Glm4MoeForCausalLM
from prime_rl.trainer.models.glm5_next import Glm5NextConfig, Glm5NextForCausalLM, Glm5NextTextConfig
from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
from prime_rl.trainer.models.gpt_oss import GptOssConfig, GptOssForCausalLM
from prime_rl.trainer.models.laguna import LagunaConfig, LagunaForCausalLM
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, cast_float_and_contiguous
from prime_rl.trainer.models.llama import LlamaForCausalLM
from prime_rl.trainer.models.minimax_m2 import MiniMaxM2Config, MiniMaxM2ForCausalLM
from prime_rl.trainer.models.muse_glimmer import MuseGlimmerConfig, MuseGlimmerForCausalLM, MuseGlimmerTextConfig
from prime_rl.trainer.models.nemotron_h import NemotronHConfig, NemotronHForCausalLM, NemotronHOmniConfig
from prime_rl.trainer.models.qwen3 import Qwen3ForCausalLM
from prime_rl.trainer.models.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from prime_rl.trainer.models.qwen3_5.modeling_clef_flash import ClefFlashForJointSchema
from prime_rl.trainer.models.qwen3_5.modeling_d3 import D3ForSequenceClassification
from prime_rl.trainer.models.qwen3_8_flash_next import (
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextForCausalLM,
    Qwen3_8FlashNextTextConfig,
)
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
from prime_rl.utils.logger import get_logger

# Make custom config discoverable by AutoConfig
AutoConfig.register("afmoe", AfmoeConfig, exist_ok=True)
AutoConfig.register("deepseek_v4", DeepseekV4Config, exist_ok=True)
AutoConfig.register("glm4_moe", Glm4MoeConfig, exist_ok=True)
AutoConfig.register("glm5_next", Glm5NextConfig, exist_ok=True)
AutoConfig.register("glm5_next_text", Glm5NextTextConfig, exist_ok=True)
AutoConfig.register("glm_moe_dsa", GlmMoeDsaConfig, exist_ok=True)
AutoConfig.register("gpt_oss", GptOssConfig, exist_ok=True)
AutoConfig.register("laguna", LagunaConfig, exist_ok=True)
AutoConfig.register("minimax_m2", MiniMaxM2Config, exist_ok=True)
AutoConfig.register("muse_glimmer", MuseGlimmerConfig, exist_ok=True)
AutoConfig.register("muse_glimmer_text", MuseGlimmerTextConfig, exist_ok=True)
AutoConfig.register("nemotron_h", NemotronHConfig, exist_ok=True)
AutoConfig.register("nemotron_h_omni", NemotronHOmniConfig, exist_ok=True)
AutoConfig.register("qwen3_moe", Qwen3MoeConfig, exist_ok=True)
AutoConfig.register("qwen3_5", Qwen3_5Config, exist_ok=True)
AutoConfig.register("qwen3_5_text", Qwen3_5TextConfig, exist_ok=True)
AutoConfig.register("qwen3_5_moe", Qwen3_5MoeConfig, exist_ok=True)
AutoConfig.register("qwen3_5_moe_text", Qwen3_5MoeTextConfig, exist_ok=True)
AutoConfig.register("qwen3_5_vision", Qwen3_5VisionConfig, exist_ok=True)
AutoConfig.register("qwen4_exp", Qwen3_8FlashNextConfig, exist_ok=True)
AutoConfig.register("qwen4_exp_text", Qwen3_8FlashNextTextConfig, exist_ok=True)

_CUSTOM_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, OrderedDict())
_CUSTOM_CAUSAL_LM_MODELS: tuple[
    tuple[type[PretrainedConfig], type[PreTrainedModelPrimeRL]],
    ...,
] = (
    (LlamaConfig, LlamaForCausalLM),
    (Qwen3Config, Qwen3ForCausalLM),
    (AfmoeConfig, AfmoeForCausalLM),
    (DeepseekV4Config, DeepseekV4ForCausalLM),
    (Glm4MoeConfig, Glm4MoeForCausalLM),
    (Glm5NextTextConfig, Glm5NextForCausalLM),
    (Glm5NextConfig, Glm5NextForCausalLM),
    (GlmMoeDsaConfig, GlmMoeDsaForCausalLM),
    (LagunaConfig, LagunaForCausalLM),
    (MiniMaxM2Config, MiniMaxM2ForCausalLM),
    (MuseGlimmerTextConfig, MuseGlimmerForCausalLM),
    (MuseGlimmerConfig, MuseGlimmerForCausalLM),
    (NemotronHConfig, NemotronHForCausalLM),
    (NemotronHOmniConfig, NemotronHForCausalLM),
    (Qwen3MoeConfig, Qwen3MoeForCausalLM),
    (Qwen3_5TextConfig, Qwen3_5ForCausalLM),
    (Qwen3_5MoeTextConfig, Qwen3_5ForCausalLM),
    (Qwen3_5Config, Qwen3_5ForCausalLM),
    (Qwen3_5MoeConfig, Qwen3_5ForCausalLM),
    (Qwen3_8FlashNextTextConfig, Qwen3_8FlashNextForCausalLM),
    (Qwen3_8FlashNextConfig, Qwen3_8FlashNextForCausalLM),
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
    "qwen3_5_moe": Qwen3_5ForCausalLM,
    "qwen4_exp": Qwen3_8FlashNextForCausalLM,
    "glm5_next": Glm5NextForCausalLM,
    "muse_glimmer": MuseGlimmerForCausalLM,
}


def get_custom_vlm_cls(model_config: PretrainedConfig) -> type | None:
    """Return the custom PrimeRL VLM class for this config, or None if unsupported."""
    return _CUSTOM_VLM_MAPPING.get(getattr(model_config, "model_type", None))


# The decision models vLLM 0.33 tells apart from their backbone by the checkpoint's architectures and a file beside its
# config.json, which it attaches to the config (transformers_utils/model_arch_config_convertor.py:622-651):
# (model_type, architecture, file) -> (class, config attribute).
_DECISION_MODELS: dict[tuple[str, str, str], tuple[type[PreTrainedModelPrimeRL], str]] = {
    ("qwen3_5", "Qwen3_5Model", "decision_config.json"): (D3ForSequenceClassification, "decision_config"),
    ("qwen3_5", "Qwen3_5ForConditionalGeneration", "joint_head_config.json"): (
        ClefFlashForJointSchema,
        "joint_head_config",
    ),
}
# Every stock Qwen3.5 checkpoint has Clef-Flash's architecture: a lookup of its file that fails warns, so that their runs
# go on. A failed lookup of d3's file raises.
_STOCK_ARCHITECTURES = {("qwen3_5", "Qwen3_5ForConditionalGeneration")}


def read_side_file(name: str, file: str) -> dict | None:
    """The JSON `file` beside the config of the checkpoint at `name` (a local directory or a Hub id), or None when the
    checkpoint has none. A Hub lookup that fails, e.g. offline or on a 5xx, raises, unless the HF cache records the
    file or its absence."""
    if Path(name).is_dir():
        path = Path(name) / file
        if not path.is_file():
            return None
    elif (cached := try_to_load_from_cache(name, file)) is _CACHED_NO_EXIST:
        return None
    elif isinstance(cached, str):
        path = Path(cached)
    else:
        try:
            path = Path(hf_hub_download(name, file))
        except RemoteEntryNotFoundError:
            return None
    return json.loads(path.read_text())


def get_decision_model_cls(model_config: PretrainedConfig, name: str) -> type[PreTrainedModelPrimeRL] | None:
    """The decision model class vLLM selects for the checkpoint at `name` (a local directory or a Hub id), or None.
    The file that selects it is read into the config attribute vLLM uses; ``config.architectures`` is left as is."""
    architectures = getattr(model_config, "architectures", None) or []
    for (model_type, architecture, file), (cls, attribute) in _DECISION_MODELS.items():
        if model_config.model_type != model_type or architectures != [architecture]:
            continue
        try:
            content = read_side_file(name, file)
        except OSError as error:
            if (model_type, architecture) not in _STOCK_ARCHITECTURES:
                raise OSError(f"Could not look up {file} for {name}, which selects {cls.__name__}: {error}") from error
            get_logger().warning(
                f"Could not look up {file} for {name} ({error}): if it is a release of {cls.__name__}, it trains as "
                "a plain Qwen3.5"
            )
            return None
        if content is not None:
            setattr(model_config, attribute, content)
            return cls
    return None


__all__ = [
    "AutoModelForCausalLMPrimeRL",
    "PreTrainedModelPrimeRL",
    "get_custom_causal_lm_cls",
    "supports_custom_impl",
    "get_custom_vlm_cls",
    "get_decision_model_cls",
    "PrimeLmOutput",
    "cast_float_and_contiguous",
]
