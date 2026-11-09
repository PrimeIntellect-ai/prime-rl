"""Conversion between HF and PrimeRL state-dict layouts, importable without the trainer.

The trainer converts checkpoints with it, and inference engines convert the PrimeRL-native tensors a trainer
publishes (ModelExpress) back to HF names. Importing this module registers PrimeRL's config classes with
AutoConfig, so ``AutoConfig.from_pretrained`` returns the config classes the chains read.
"""

from collections.abc import Callable

from transformers import AutoConfig, PretrainedConfig

from prime_rl.model_conversion import (
    converting_afmoe,
    converting_deepseek_v4,
    converting_glm4_moe,
    converting_glm5_next,
    converting_glm_moe_dsa,
    converting_gpt_oss,
    converting_laguna,
    converting_minimax_m2,
    converting_nemotron_h,
    converting_qwen3_5,
    converting_qwen3_8_flash_next,
    converting_qwen3_moe,
)
from prime_rl.model_conversion.configuration_afmoe import AfmoeConfig
from prime_rl.model_conversion.configuration_deepseek_v4 import DeepseekV4Config
from prime_rl.model_conversion.configuration_glm4_moe import Glm4MoeConfig
from prime_rl.model_conversion.configuration_glm5_next import Glm5NextConfig, Glm5NextTextConfig
from prime_rl.model_conversion.configuration_glm_moe_dsa import GlmMoeDsaConfig
from prime_rl.model_conversion.configuration_gpt_oss import GptOssConfig
from prime_rl.model_conversion.configuration_laguna import LagunaConfig
from prime_rl.model_conversion.configuration_minimax_m2 import MiniMaxM2Config
from prime_rl.model_conversion.configuration_muse_glimmer import MuseGlimmerConfig, MuseGlimmerTextConfig
from prime_rl.model_conversion.configuration_nemotron_h import NemotronHConfig
from prime_rl.model_conversion.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from prime_rl.model_conversion.configuration_qwen3_8_flash_next import (
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextTextConfig,
)
from prime_rl.model_conversion.configuration_qwen3_moe import Qwen3MoeConfig
from prime_rl.model_conversion.ops import ConvOp

for _config_cls in (
    AfmoeConfig,
    DeepseekV4Config,
    Glm4MoeConfig,
    Glm5NextConfig,
    Glm5NextTextConfig,
    GlmMoeDsaConfig,
    GptOssConfig,
    LagunaConfig,
    MiniMaxM2Config,
    MuseGlimmerConfig,
    MuseGlimmerTextConfig,
    NemotronHConfig,
    Qwen3MoeConfig,
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5VisionConfig,
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextTextConfig,
):
    AutoConfig.register(_config_cls.model_type, _config_cls, exist_ok=True)

# HF model_type -> conversion chain. Model types missing here (llama, qwen3, muse_glimmer) keep HF names.
CONVERSION_CHAINS: dict[str, Callable[[PretrainedConfig], list[ConvOp]]] = {
    "afmoe": converting_afmoe.conversion_chain,
    "deepseek_v4": converting_deepseek_v4.conversion_chain,
    "glm4_moe": converting_glm4_moe.conversion_chain,
    "glm5_next": converting_glm5_next.conversion_chain,
    "glm5_next_text": converting_glm5_next.conversion_chain,
    "glm_moe_dsa": converting_glm_moe_dsa.conversion_chain,
    "gpt_oss": converting_gpt_oss.conversion_chain,
    "laguna": converting_laguna.conversion_chain,
    "minimax_m2": converting_minimax_m2.conversion_chain,
    "nemotron_h": converting_nemotron_h.conversion_chain,
    "qwen3_moe": converting_qwen3_moe.conversion_chain,
    "qwen3_5": converting_qwen3_5.conversion_chain,
    "qwen3_5_text": converting_qwen3_5.conversion_chain,
    "qwen3_5_moe": converting_qwen3_5.conversion_chain,
    "qwen3_5_moe_text": converting_qwen3_5.conversion_chain,
    "qwen4_exp": converting_qwen3_8_flash_next.conversion_chain,
    "qwen4_exp_text": converting_qwen3_8_flash_next.conversion_chain,
}


def conversion_chain(config: PretrainedConfig) -> list[ConvOp]:
    """Declarative operations converting between HF and PrimeRL state dicts for this config."""
    chain = CONVERSION_CHAINS.get(config.model_type)
    return chain(config) if chain else []
