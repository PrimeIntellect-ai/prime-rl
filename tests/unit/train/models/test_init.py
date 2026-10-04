import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaConfig, Qwen3Config

from prime_rl.trainer.models import get_custom_causal_lm_cls
from prime_rl.trainer.models.afmoe import AfmoeConfig
from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig
from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaConfig
from prime_rl.trainer.models.gpt_oss import GptOssConfig
from prime_rl.trainer.models.laguna import LagunaConfig
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.layers.moe import GroupedExperts
from prime_rl.trainer.models.layers.norms import RMSNorm
from prime_rl.trainer.models.minimax_m2 import MiniMaxM2Config
from prime_rl.trainer.models.nemotron_h import NemotronHConfig
from prime_rl.trainer.models.qwen3_5 import Qwen3_5MoeTextConfig, Qwen3_5TextConfig
from prime_rl.trainer.models.qwen3_8_flash_next import Qwen3_8FlashNextTextConfig
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig

STD = 0.03
DENSE = dict(
    vocab_size=512,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    max_position_embeddings=256,
    pad_token_id=0,
    initializer_range=STD,
)
MOE = dict(DENSE, moe_intermediate_size=64, num_experts_per_tok=2)
LINEAR_ATTENTION = dict(
    layer_types=["linear_attention", "full_attention"],
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
)
CONFIGS = {
    "llama": lambda: LlamaConfig(**DENSE),
    "qwen3": lambda: Qwen3Config(**DENSE),
    "qwen3_moe": lambda: Qwen3MoeConfig(**MOE, num_experts=4),
    "glm4_moe": lambda: Glm4MoeConfig(**MOE, n_routed_experts=4, first_k_dense_replace=1),
    # The sparse MLA kernel fixes kv_lora_rank + qk_rope_head_dim = 576 (the defaults)
    "glm_moe_dsa": lambda: GlmMoeDsaConfig(**MOE, n_routed_experts=4, first_k_dense_replace=1, q_lora_rank=64),
    "minimax_m2": lambda: MiniMaxM2Config(**MOE, num_local_experts=4, rotary_dim=16),
    "laguna": lambda: LagunaConfig(
        **MOE,
        num_experts=4,
        shared_expert_intermediate_size=64,
        num_attention_heads_per_layer=[4, 4],
        layer_types=["full_attention", "sliding_attention"],
        mlp_layer_types=["dense", "sparse"],
    ),
    "afmoe": lambda: AfmoeConfig(
        **MOE, num_experts=4, num_dense_layers=1, layer_types=["sliding_attention", "full_attention"]
    ),
    "gpt_oss": lambda: GptOssConfig(**MOE, num_local_experts=4),
    "nemotron_h": lambda: NemotronHConfig(
        **MOE,
        n_routed_experts=4,
        mamba_num_heads=4,
        mamba_head_dim=32,
        ssm_state_size=32,
        n_groups=1,
        moe_shared_expert_intermediate_size=64,
        moe_latent_size=64,
        hybrid_override_pattern="M*E",
    ),
    "qwen3_5": lambda: Qwen3_5TextConfig(**DENSE, **LINEAR_ATTENTION),
    "qwen3_5_moe": lambda: Qwen3_5MoeTextConfig(
        **MOE, **LINEAR_ATTENTION, num_experts=4, shared_expert_intermediate_size=64
    ),
    # Linear attention only: indexed attention loads a GPU kernel at construction.
    "qwen3_8_flash_next": lambda: Qwen3_8FlashNextTextConfig(
        **dict(MOE, eos_token_id=1, bos_token_id=1),
        **dict(LINEAR_ATTENTION, layer_types=["linear_attention"] * 2),
        num_experts=4,
        shared_expert_intermediate_size=64,
        hc_count=2,
        hc_lowrank=8,
        ple_layer_ids=[1],
        ple_embed_dim=32,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=3,
    ),
    "deepseek_v4": lambda: DeepseekV4Config(
        **dict(MOE, num_hidden_layers=3, num_attention_heads=32, num_key_value_heads=1),
        q_lora_rank=64,
        o_groups=2,
        o_lora_rank=16,
        layer_types=["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"],
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
        n_routed_experts=4,
        num_hash_layers=1,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=2,
        sliding_window=8,
    ),
}


def build_from_scratch(arch: str, device: str) -> nn.Module:
    """The trainer's `model.init = "scratch"` path without FSDP: meta -> empty -> initialized."""
    torch.manual_seed(0)
    config = CONFIGS[arch]()
    config._attn_implementation = "flash_attention_2"
    config.tie_word_embeddings = False
    with torch.device("meta"):
        model = get_custom_causal_lm_cls(config)._from_config(config)
    model.to_empty(device=device)
    with torch.no_grad():
        for tensor in model.state_dict().values():
            tensor.fill_(float("nan") if tensor.is_floating_point() else -1)
    model.init_buffers_post_meta()
    model.initialize_weights()
    return model


@pytest.mark.parametrize("arch", CONFIGS)
def test_scratch_init_fills_every_weight(arch):
    model = build_from_scratch(arch, "cpu")

    for name, tensor in model.state_dict().items():
        if tensor.is_floating_point():
            assert torch.isfinite(tensor).all(), name
        else:
            assert (tensor >= 0).all(), name
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding, GroupedExperts)):
            for param_name, param in module.named_parameters(recurse=False):
                if "bias" in param_name:
                    assert (param == 0).all(), f"{name}.{param_name}"
                else:
                    # Five standard errors of the sample std and mean of a normal(0, STD) draw
                    tolerance = 5 / math.sqrt(param.numel())
                    assert param.std().item() == pytest.approx(STD, rel=tolerance), f"{name}.{param_name}"
                    assert param.mean().abs().item() < STD * tolerance, f"{name}.{param_name}"
        elif isinstance(module, RMSNorm):
            assert (module.weight == 1).all(), name


@pytest.mark.gpu
@pytest.mark.parametrize("arch", CONFIGS)
def test_scratch_init_starts_at_uniform_loss(arch):
    model = build_from_scratch(arch, "cuda").to(torch.bfloat16)
    inject_prime_lm_head(model)
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(1, vocab_size, (1, 64), device="cuda")

    logits = model(input_ids, seq_lens=torch.tensor([64], device="cuda"))["logits"]
    loss = F.cross_entropy(logits[0, :-1].float(), input_ids[0, 1:])

    # Small random logits: the loss sits just above the uniform-prediction loss
    assert loss.item() == pytest.approx(math.log(vocab_size), abs=0.25)
