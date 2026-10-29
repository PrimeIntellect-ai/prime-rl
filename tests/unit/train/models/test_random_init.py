import pytest
import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from prime_rl.trainer.models import get_custom_causal_lm_cls
from prime_rl.trainer.models.afmoe import AfmoeConfig
from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig
from prime_rl.trainer.models.gpt_oss import GptOssConfig
from prime_rl.trainer.models.laguna import LagunaConfig
from prime_rl.trainer.models.minimax_m2 import MiniMaxM2Config
from prime_rl.trainer.models.muse_glimmer import MuseGlimmerTextConfig
from prime_rl.trainer.models.nemotron_h import NemotronHConfig
from prime_rl.trainer.models.qwen3_5 import Qwen3_5MoeTextConfig, Qwen3_5TextConfig
from prime_rl.trainer.models.qwen3_8_flash_next import Qwen3_8FlashNextTextConfig
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig

TINY = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 512,
    "num_experts": 4,
    "num_local_experts": 4,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
    "max_position_embeddings": 512,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "ngram_vocab_size_base": 1024,
    "ple_embed_dim": 64,
    "hc_lowrank": 16,
}

CONFIGS = [
    LlamaConfig,
    Qwen3Config,
    Qwen3MoeConfig,
    AfmoeConfig,
    Glm4MoeConfig,
    LagunaConfig,
    MiniMaxM2Config,
    MuseGlimmerTextConfig,
    NemotronHConfig,
    Qwen3_5TextConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_8FlashNextTextConfig,
    GptOssConfig,
]


def tiny_config(config_cls):
    config = config_cls()
    for name, value in TINY.items():
        if isinstance(getattr(config, name, None), int):
            setattr(config, name, value)
    config.pad_token_id = None
    config._attn_implementation = "flash_attention_2"
    return config


@pytest.mark.parametrize("config_cls", CONFIGS, ids=lambda cls: cls.__name__)
def test_random_init_initializes_every_parameter(config_cls):
    """The meta-device build materialized with `to_empty` holds garbage until `initialize_weights` runs."""
    config = tiny_config(config_cls)
    with torch.device("meta"):
        model = get_custom_causal_lm_cls(config)._from_config(config)
    model.to_empty(device="cpu")
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(float("nan"))

    model.initialize_weights()

    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), f"{name} was not initialized"
        # Biases, norm weights and some convolutions start at zero by design, weight matrices do not
        if param.ndim == 2 and not name.endswith("bias"):
            assert param.count_nonzero() > 0, f"{name} is all zeros"
