import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from prime_rl.trainer.models.afmoe import AfmoeConfig, AfmoeForCausalLM
from prime_rl.trainer.models.base import PrimeModel
from prime_rl.trainer.models.config import AttnImplementation, PrimeModelConfig
from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig, Glm4MoeForCausalLM
from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
from prime_rl.trainer.models.gpt_oss import GptOssConfig, GptOssForCausalLM
from prime_rl.trainer.models.laguna import LagunaConfig, LagunaForCausalLM
from prime_rl.trainer.models.llama import LlamaConfig, LlamaForCausalLM
from prime_rl.trainer.models.minimax_m2 import MiniMaxM2Config, MiniMaxM2ForCausalLM
from prime_rl.trainer.models.nemotron_h import NemotronHConfig, NemotronHForCausalLM
from prime_rl.trainer.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
from prime_rl.trainer.models.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
)
from prime_rl.trainer.models.qwen3_8_flash_next import (
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextForCausalLM,
    Qwen3_8FlashNextTextConfig,
)
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

# `model_type` (from config.json) -> (config class, model class)
MODEL_REGISTRY: dict[str, tuple[type[PrimeModelConfig], type[PrimeModel]]] = {
    config_cls.model_type: (config_cls, model_cls)
    for config_cls, model_cls in (
        (LlamaConfig, LlamaForCausalLM),
        (Qwen3Config, Qwen3ForCausalLM),
        (AfmoeConfig, AfmoeForCausalLM),
        (DeepseekV4Config, DeepseekV4ForCausalLM),
        (Glm4MoeConfig, Glm4MoeForCausalLM),
        (GlmMoeDsaConfig, GlmMoeDsaForCausalLM),
        (GptOssConfig, GptOssForCausalLM),
        (LagunaConfig, LagunaForCausalLM),
        (MiniMaxM2Config, MiniMaxM2ForCausalLM),
        (NemotronHConfig, NemotronHForCausalLM),
        (Qwen3MoeConfig, Qwen3MoeForCausalLM),
        (Qwen3_5TextConfig, Qwen3_5ForCausalLM),
        (Qwen3_5MoeTextConfig, Qwen3_5ForCausalLM),
        (Qwen3_5Config, Qwen3_5ForCausalLM),
        (Qwen3_5MoeConfig, Qwen3_5ForCausalLM),
        (Qwen3_8FlashNextTextConfig, Qwen3_8FlashNextForCausalLM),
        (Qwen3_8FlashNextConfig, Qwen3_8FlashNextForCausalLM),
    )
}


def _lookup(model_type: str) -> tuple[type[PrimeModelConfig], type[PrimeModel]]:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"{model_type!r} has no PrimeRL model implementation. Supported model types: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_type]


def get_model_cls(model_type: str) -> type[PrimeModel]:
    return _lookup(model_type)[1]


def read_config_json(name_or_path: str) -> dict:
    """Read the raw ``config.json`` of a local checkpoint directory or a HuggingFace Hub repo."""
    path = Path(name_or_path)
    config_file = path / "config.json" if path.is_dir() else Path(hf_hub_download(name_or_path, "config.json"))
    return json.loads(config_file.read_text())


def build_model_config(
    config_dict: dict, attn_implementation: AttnImplementation = "flash_attention_2"
) -> PrimeModelConfig:
    """Validate a raw ``config.json`` dict into the config class registered for its ``model_type``."""
    config_cls, _ = _lookup(config_dict["model_type"])
    return config_cls.model_validate({**config_dict, "attn_implementation": attn_implementation})


def load_model_config(
    name_or_path: str, attn_implementation: AttnImplementation = "flash_attention_2"
) -> PrimeModelConfig:
    """Load the PrimeRL config of a local checkpoint directory or a HuggingFace Hub repo."""
    return build_model_config(read_config_json(name_or_path), attn_implementation)


__all__ = ["MODEL_REGISTRY", "build_model_config", "get_model_cls", "load_model_config", "read_config_json"]
