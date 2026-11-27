"""PrimeRL model implementations, registered by the HF config class they are built from."""

import torch
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from prime_rl.trainer.models.afmoe import AfmoeConfig, AfmoeForCausalLM
from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig, Glm4MoeForCausalLM
from prime_rl.trainer.models.glm5_next import Glm5NextConfig, Glm5NextForCausalLM, Glm5NextTextConfig
from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
from prime_rl.trainer.models.gpt_oss import GptOssConfig, GptOssForCausalLM
from prime_rl.trainer.models.inkling2 import Inkling2Config, Inkling2ForCausalLM, Inkling2TextConfig
from prime_rl.trainer.models.laguna import LagunaConfig, LagunaForCausalLM
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, cast_float_and_contiguous
from prime_rl.trainer.models.llama import LlamaForCausalLM
from prime_rl.trainer.models.minimax_m2 import MiniMaxM2Config, MiniMaxM2ForCausalLM
from prime_rl.trainer.models.muse_glimmer import MuseGlimmerConfig, MuseGlimmerForCausalLM, MuseGlimmerTextConfig
from prime_rl.trainer.models.nemotron_h import NemotronHConfig, NemotronHForCausalLM, NemotronHOmniConfig
from prime_rl.trainer.models.qwen3 import Qwen3ForCausalLM
from prime_rl.trainer.models.qwen3_5 import (
    ClefFlashForJointSchema,
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5ForSlotDecision,
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from prime_rl.trainer.models.qwen3_8_flash_next import (
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextForCausalLM,
    Qwen3_8FlashNextTextConfig,
)
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

# Make custom config discoverable by AutoConfig
AutoConfig.register("afmoe", AfmoeConfig, exist_ok=True)
AutoConfig.register("deepseek_v4", DeepseekV4Config, exist_ok=True)
AutoConfig.register("glm4_moe", Glm4MoeConfig, exist_ok=True)
AutoConfig.register("glm5_next", Glm5NextConfig, exist_ok=True)
AutoConfig.register("glm5_next_text", Glm5NextTextConfig, exist_ok=True)
AutoConfig.register("glm_moe_dsa", GlmMoeDsaConfig, exist_ok=True)
AutoConfig.register("gpt_oss", GptOssConfig, exist_ok=True)
AutoConfig.register("inkling2", Inkling2Config, exist_ok=True)
AutoConfig.register("inkling2_text", Inkling2TextConfig, exist_ok=True)
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
    (Inkling2TextConfig, Inkling2ForCausalLM),
    (Inkling2Config, Inkling2ForCausalLM),
)
_CUSTOM_CAUSAL_LM_MAPPING: dict[type[PretrainedConfig], type[PreTrainedModelPrimeRL]] = dict(_CUSTOM_CAUSAL_LM_MODELS)

_CUSTOM_CAUSAL_LM_BY_MODEL_TYPE = {
    config_cls.model_type: model_cls for config_cls, model_cls in _CUSTOM_CAUSAL_LM_MODELS
}
assert len(_CUSTOM_CAUSAL_LM_BY_MODEL_TYPE) == len(_CUSTOM_CAUSAL_LM_MODELS), (
    "registered config model types must be unique"
)

_TEXT_ONLY_COMPOSITE_CONFIGS: frozenset[type[PretrainedConfig]] = frozenset({Qwen3_5Config})

# Classes selected, before their family's, by an extra config file next to a stock config.json or by one of its keys.
_SELECTED_BY_EXTRA_CONFIG: tuple[type[PreTrainedModelPrimeRL], ...] = (ClefFlashForJointSchema, Qwen3_5ForSlotDecision)


def check_selected_by_extra_config(classes: tuple[type[PreTrainedModelPrimeRL], ...]) -> None:
    """Raise unless each class sets what selects it, an extra config file or a config.json key, and names at most one
    export file, and no two share a name or what selects them."""
    for cls in classes:
        by_file = cls.extra_config_file is not None
        if by_file == (cls.extra_config_key is not None):
            raise TypeError(f"{cls.__name__} must set either extra_config_file or extra_config_key to be selected")
        # The config already holds a key's value, under the key's name
        if not by_file and cls.extra_config_attr is not None:
            raise TypeError(f"{cls.__name__} must not set extra_config_attr to be selected by its config.json key")
        attrs = ("extra_config_attr",) if by_file else ()
        missing = [attr for attr in (*attrs, "stock_model_type", "stock_architecture") if getattr(cls, attr) is None]
        if missing:
            selector = "extra config file" if by_file else "config.json key"
            raise TypeError(f"{cls.__name__} must set {', '.join(missing)} to be selected by its {selector}")
        # Loading reads export files by their stored names, which could clash across files
        if len(cls.export_files) > 1:
            raise ValueError(f"{cls.__name__}.export_files may name one file")
    names = {cls.__name__ for cls in classes}
    if len(names) < len(classes):
        raise ValueError("Two classes selected by an extra config file or key have the same name")
    if names & {cls.stock_architecture for cls in classes}:
        raise ValueError("A class selected by an extra config file or key has the name of a stock architecture")
    # A file selects before a key, so one class of each may extend a stock config.json
    stock = {(cls.stock_model_type, cls.stock_architecture, cls.extra_config_key is None) for cls in classes}
    if len(stock) < len(classes):
        raise ValueError("Two classes extend the same stock config.json, both by an extra config file or both by a key")


check_selected_by_extra_config(_SELECTED_BY_EXTRA_CONFIG)


def get_selected_by_extra_config() -> tuple[type[PreTrainedModelPrimeRL], ...]:
    """The classes selected by an extra config file or a config.json key."""
    return _SELECTED_BY_EXTRA_CONFIG


def custom_causal_lm_from_config(
    config: PretrainedConfig, *, attn_implementation: str | None = None, dtype: torch.dtype | None = None
) -> PreTrainedModelPrimeRL:
    """Build the model registered for the config's class, from its text config for a text-only composite config."""
    model_cls = _CUSTOM_CAUSAL_LM_MAPPING[type(config)]
    if type(config) in _TEXT_ONLY_COMPOSITE_CONFIGS:
        config = config.get_text_config()
    return model_cls.from_config(config, attn_implementation=attn_implementation, dtype=dtype)


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


__all__ = [
    "PreTrainedModelPrimeRL",
    "custom_causal_lm_from_config",
    "get_custom_causal_lm_cls",
    "supports_custom_impl",
    "get_custom_vlm_cls",
    "get_selected_by_extra_config",
    "PrimeLmOutput",
    "cast_float_and_contiguous",
]
