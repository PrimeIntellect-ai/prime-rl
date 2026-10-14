import subprocess
import sys
from pathlib import Path

import pytest
import torch

from prime_rl.trainer.models import (
    AfmoeConfig,
    AutoModelForCausalLMPrimeRL,
    DeepseekV4Config,
    Glm4MoeConfig,
    GlmMoeDsaConfig,
    GptOssConfig,
    LagunaConfig,
    LlamaConfig,
    MiniMaxM2Config,
    NemotronHConfig,
    Qwen3_5Config,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
    Qwen3_8FlashNextTextConfig,
    Qwen3Config,
    Qwen3MoeConfig,
)
from prime_rl.trainer.models.base import AttentionFlops
from prime_rl.trainer.perf import FlopCounter

N = 256  # document length; sliding windows are 32 tokens and sparse attention reads 32 tokens
DIMS = dict(
    hidden_size=64, intermediate_size=128, vocab_size=128, num_attention_heads=4, num_key_value_heads=2, pad_token_id=0
)
MOE = dict(moe_intermediate_size=32, num_experts_per_tok=2)
# Softmax attention of 4 heads of qk/v dim 16 over `attended` keys per query, 6 FLOPs per multiply-add.
FULL = 6 * 4 * (16 + 16) * N * N
WINDOW = 6 * 4 * (16 + 16) * N * 32

MODELS = {
    "llama": (LlamaConfig(**DIMS, head_dim=16, num_hidden_layers=2), 2 * FULL),
    "qwen3": (Qwen3Config(**DIMS, head_dim=16, num_hidden_layers=2), 2 * FULL),
    "qwen3_moe": (Qwen3MoeConfig(**DIMS, **MOE, num_experts=4, num_hidden_layers=2), 2 * FULL),
    "glm4_moe": (Glm4MoeConfig(**DIMS, **MOE, n_routed_experts=4, num_hidden_layers=2), 2 * FULL),
    "minimax_m2": (MiniMaxM2Config(**DIMS, head_dim=16, num_local_experts=4, num_hidden_layers=2), 2 * FULL),
    "afmoe": (
        AfmoeConfig(
            **DIMS,
            **MOE,
            head_dim=16,
            num_experts=4,
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=32,
        ),
        WINDOW + FULL,
    ),
    "laguna": (
        LagunaConfig(
            **DIMS,
            **MOE,
            head_dim=16,
            num_experts=4,
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=32,
        ),
        WINDOW + FULL,
    ),
    "gpt_oss": (
        GptOssConfig(
            **DIMS,
            head_dim=16,
            num_local_experts=4,
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=32,
        ),
        WINDOW + FULL,
    ),
    # Mamba2: 4 heads, head_dim 16, state 8; two head_dim * state products per token and head.
    "nemotron_h": (
        NemotronHConfig(
            **DIMS,
            head_dim=16,
            hybrid_override_pattern="M*",
            mamba_num_heads=4,
            mamba_head_dim=16,
            ssm_state_size=8,
            n_groups=1,
        ),
        6 * 2 * 4 * 16 * 8 * N + FULL,
    ),
    # Gated delta rule: 4 value heads, key/value head dim 8; three key * value products per token and head.
    "qwen3_5": (
        Qwen3_5TextConfig(
            **DIMS,
            head_dim=16,
            layer_types=["linear_attention", "full_attention"],
            num_hidden_layers=2,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
        ),
        6 * 3 * 4 * 8 * 8 * N + FULL,
    ),
    "qwen3_5_vlm": (
        Qwen3_5Config(
            text_config=dict(
                **DIMS,
                head_dim=16,
                layer_types=["linear_attention", "full_attention"],
                num_hidden_layers=2,
                linear_num_key_heads=2,
                linear_num_value_heads=4,
                linear_key_head_dim=8,
                linear_value_head_dim=8,
            )
        ),
        6 * 3 * 4 * 8 * 8 * N + FULL,
    ),
    "qwen3_5_moe": (
        Qwen3_5MoeTextConfig(
            **DIMS,
            **MOE,
            num_experts=4,
            shared_expert_intermediate_size=32,
            head_dim=16,
            layer_types=["linear_attention", "full_attention"],
            num_hidden_layers=2,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
        ),
        6 * 3 * 4 * 8 * 8 * N + FULL,
    ),
    # Indexed attention reads 32 tokens; its indexer (2 heads of dim 8) scores the N / 4 compressed blocks.
    "qwen3_8_flash_next": (
        Qwen3_8FlashNextTextConfig(
            **DIMS,
            **MOE,
            num_experts=4,
            shared_expert_intermediate_size=32,
            head_dim=16,
            layer_types=["linear_attention", "full_attention"],
            num_hidden_layers=2,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            indexer_n_heads=2,
            indexer_head_dim=8,
            indexer_budget=32,
            indexer_compress_ratio=4,
        ),
        6 * 3 * 4 * 8 * 8 * N + 6 * 2 * 8 * N * (N // 4) + WINDOW,
    ),
    # MLA with qk = 16 nope + 8 rope, v = 16, reading index_topk = 32 tokens; the indexer (2 heads of dim 8)
    # scores all N tokens on the "full" layer, and the "shared" layer reuses its indices.
    "glm_moe_dsa": (
        GlmMoeDsaConfig(
            **DIMS,
            **MOE,
            n_routed_experts=4,
            num_hidden_layers=2,
            q_lora_rank=32,
            kv_lora_rank=16,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
            index_n_heads=2,
            index_head_dim=8,
            index_topk=32,
            indexer_types=["full", "shared"],
        ),
        2 * 6 * 4 * (24 + 16) * N * 32 + 6 * 2 * 8 * N * N,
    ),
    # Every layer attends a 32-token window. CSA (rate 4) adds index_topk = 16 of its N / 4 compressed
    # entries, which its indexer (2 heads of dim 8) scores; HCA (rate 16) adds all N / 16 = 16 entries.
    "deepseek_v4": (
        DeepseekV4Config(
            **DIMS | dict(num_attention_heads=32, num_key_value_heads=1),
            **MOE,
            n_routed_experts=4,
            head_dim=32,
            num_hidden_layers=3,
            layer_types=["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"],
            compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 16},
            sliding_window=32,
            index_n_heads=2,
            index_head_dim=8,
            index_topk=16,
        ),
        6 * 32 * (32 + 32) * N * (32 + 2 * (32 + 16)) + 6 * 2 * 8 * N * (N // 4),
    ),
}


def build(config):
    with torch.device("meta"):
        return AutoModelForCausalLMPrimeRL.from_config(config, attn_implementation="flash_attention_3")


@pytest.mark.parametrize("name", MODELS)
def test_attention_flops_match_hand_count(name: str):
    config, attention_flops = MODELS[name]
    counter = FlopCounter(build(config))
    assert counter([N]) - N * counter.matmul_flops_per_token == attention_flops > 0


@pytest.mark.parametrize("name", MODELS)
def test_attention_flops_modules_are_not_nested(name: str):
    model = build(MODELS[name][0])
    for module in model.modules():
        if isinstance(module, AttentionFlops):
            assert not any(isinstance(child, AttentionFlops) for child in list(module.modules())[1:])


def test_models_build_without_kernel_libraries():
    """The orchestrator builds the model on meta to count FLOPs, on hosts without the `gpu` extra."""
    blocked = ["fla", "flash_attn", "flash_attn_interface", "flash_mla", "mamba_ssm", "prime_kernels"]
    blocked += ["ring_flash_attn", "tilelang", "deep_gemm", "deep_ep", "sonicmoe", "quack", "cudnn", "torchao"]
    code = f"""
import sys
sys.modules.update(dict.fromkeys({blocked}))
sys.path.insert(0, {str(Path(__file__).parent)!r})
from test_perf import MODELS, N, build
from prime_rl.trainer.perf import FlopCounter
for config, attention_flops in MODELS.values():
    counter = FlopCounter(build(config))
    assert counter([N]) - N * counter.matmul_flops_per_token == attention_flops
"""
    subprocess.run([sys.executable, "-c", code], check=True)
