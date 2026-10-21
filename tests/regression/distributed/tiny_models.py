"""Seeded tiny HF checkpoints for every trainer arch, built from explicit config kwargs."""

from pathlib import Path

import torch
from torch import Tensor
from transformers import LlamaConfig, Qwen3Config

from prime_rl.trainer.models import get_custom_causal_lm_cls
from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.deepseek_v4.moe import DeepseekV4HashRouter
from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig
from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaConfig
from prime_rl.trainer.models.gpt_oss import GptOssConfig
from prime_rl.trainer.models.laguna import LagunaConfig
from prime_rl.trainer.models.layers.moe import MoE
from prime_rl.trainer.models.minimax_m2 import MiniMaxM2Config
from prime_rl.trainer.models.nemotron_h import NemotronHConfig
from prime_rl.trainer.models.qwen3_5 import Qwen3_5MoeTextConfig, Qwen3_5TextConfig
from prime_rl.trainer.models.qwen3_8_flash_next import Qwen3_8FlashNextTextConfig
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig
from prime_rl.utils.utils import default_dtype
from prime_rl.utils.weights import load_state_dict, save_state_dict

SEED = 0

TOKENIZERS = {
    "llama": "unsloth/Llama-3.2-1B-Instruct",
    "qwen3_moe": "Qwen/Qwen3-30B-A3B",
    "glm_moe_dsa": "zai-org/GLM-5",
    "deepseek_v4": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "qwen3_dense": "Qwen/Qwen3-0.6B",
    "qwen3_5_dense": "Qwen/Qwen3.5-0.8B",
    "glm4_moe": "zai-org/GLM-4.5-Air",
    "gpt_oss": "openai/gpt-oss-20b",
    "laguna": "poolside/Laguna-XS.2",
    "minimax_m2": "MiniMaxAI/MiniMax-M2.5",
    "nemotron_h": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "qwen3_5_moe": "Qwen/Qwen3.5-35B-A3B",
    "qwen4_exp": "Qwen/Qwen3.8-Flash-Next",
}

TINY_CONFIGS = {
    "qwen3_moe": Qwen3MoeConfig(
        vocab_size=151936,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=40960,
        rope_theta=1000000.0,
        moe_intermediate_size=256,
        num_experts=16,
        num_experts_per_tok=4,
        norm_topk_prob=True,
        bos_token_id=151643,
        eos_token_id=151645,
    ),
    "glm_moe_dsa": GlmMoeDsaConfig(
        vocab_size=154880,
        hidden_size=512,
        intermediate_size=1024,
        moe_intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=32,
        num_key_value_heads=32,
        n_shared_experts=1,
        n_routed_experts=16,
        num_experts_per_tok=4,
        first_k_dense_replace=1,
        kv_lora_rank=512,
        q_lora_rank=256,
        qk_rope_head_dim=64,
        qk_nope_head_dim=192,
        v_head_dim=256,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=256,
        rope_parameters={"rope_theta": 1000000, "rope_type": "default"},
        eos_token_id=[154820, 154827, 154829],
    ),
    "deepseek_v4": DeepseekV4Config(
        vocab_size=129280,
        hidden_size=512,
        moe_intermediate_size=256,
        num_hidden_layers=5,
        num_attention_heads=32,
        num_key_value_heads=1,
        head_dim=128,
        q_lora_rank=256,
        partial_rotary_factor=0.5,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        max_position_embeddings=65536,
        sliding_window=128,
        o_groups=2,
        o_lora_rank=128,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
            "compressed_sparse_attention",
            "sliding_attention",
        ],
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
        index_n_heads=64,
        index_head_dim=128,
        index_topk=32,
        n_routed_experts=16,
        num_experts_per_tok=4,
        n_shared_experts=1,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        num_hash_layers=2,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        rms_norm_eps=1e-6,
        bos_token_id=0,
        eos_token_id=1,
    ),
    "llama": LlamaConfig(
        vocab_size=128256,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=131072,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
    ),
    "qwen3_dense": Qwen3Config(
        vocab_size=151936,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=40960,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
    ),
    "qwen3_5_dense": Qwen3_5TextConfig(
        vocab_size=248320,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=65536,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_conv_kernel_dim=4,
    ),
    "glm4_moe": Glm4MoeConfig(
        vocab_size=151552,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=131072,
        rms_norm_eps=1e-5,
        rope_theta=1000000,
        attention_bias=True,
        partial_rotary_factor=0.5,
        moe_intermediate_size=256,
        n_routed_experts=16,
        num_experts_per_tok=4,
        n_shared_experts=1,
        first_k_dense_replace=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        use_qk_norm=False,
        pad_token_id=151329,
        eos_token_id=[151329, 151336, 151338],
    ),
    "gpt_oss": GptOssConfig(
        vocab_size=201088,
        hidden_size=512,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=131072,
        num_local_experts=16,
        num_experts_per_tok=4,
        sliding_window=128,
        rope_parameters={"rope_type": "default", "rope_theta": 150000.0},
    ),
    "laguna": LagunaConfig(
        vocab_size=100352,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_attention_heads_per_layer=[8] * 4,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 500000.0, "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
        },
        layer_types=["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
        sliding_window=512,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
        num_experts=16,
        num_experts_per_tok=4,
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        moe_routed_scaling_factor=2.5,
        pad_token_id=9,
        bos_token_id=2,
        eos_token_id=[2, 24],
    ),
    "minimax_m2": MiniMaxM2Config(
        vocab_size=200064,
        hidden_size=512,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_theta=5000000,
        rotary_dim=32,
        num_local_experts=16,
        num_experts_per_tok=4,
        scoring_func="sigmoid",
        use_routing_bias=True,
        use_qk_norm=True,
        qk_norm_type="per_layer",
    ),
    "nemotron_h": NemotronHConfig(
        vocab_size=131072,
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=262144,
        intermediate_size=1024,
        hybrid_override_pattern="ME*E",
        expand=2,
        mamba_num_heads=16,
        mamba_head_dim=64,
        ssm_state_size=64,
        n_groups=4,
        conv_kernel=4,
        chunk_size=64,
        n_routed_experts=16,
        n_shared_experts=1,
        moe_intermediate_size=256,
        moe_shared_expert_intermediate_size=256,
        num_experts_per_tok=4,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    ),
    "qwen3_5_moe": Qwen3_5MoeTextConfig(
        vocab_size=248320,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=65536,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_conv_kernel_dim=4,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
        num_experts=16,
        num_experts_per_tok=4,
    ),
    "qwen4_exp": Qwen3_8FlashNextTextConfig(
        vocab_size=248320,
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=16,
        num_key_value_heads=1,
        head_dim=256,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=256,
        indexer_compress_ratio=4,
        hc_count=4,
        hc_lowrank=64,
        ple_layer_ids=[2],
        ple_embed_dim=512,
        heads_per_ngram=8,
        ngram_vocab_size_base=10000,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=4,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
        num_experts=16,
        num_experts_per_tok=4,
    ),
}


def _fill_hash_tables(model: torch.nn.Module) -> None:
    """Give every DSv4 hash router a `tid2eid` table with distinct experts per token, as a real one has."""
    for router in model.modules():
        if isinstance(router, DeepseekV4HashRouter):
            rows = [torch.randperm(router.num_experts)[: router.top_k] for _ in range(router.tid2eid.shape[0])]
            router.tid2eid.copy_(torch.stack(rows))


def _init_moe_weights(model: torch.nn.Module, init_std: float) -> None:
    """Initialize every MoE layer's experts and router, which `_from_config` leaves as uninitialized memory."""
    for moe in model.modules():
        if isinstance(moe, MoE):
            moe.init_weights(init_std, torch.device("cpu"))


def write_tiny_checkpoint(arch: str, out_dir: Path) -> Path:
    """Write a seeded tiny HF safetensors checkpoint for `arch` and verify its HF -> PrimeRL roundtrip."""
    config = TINY_CONFIGS[arch]
    model_cls = get_custom_causal_lm_cls(config)
    torch.manual_seed(SEED)
    with torch.device("cpu"), default_dtype(torch.bfloat16), torch.no_grad():
        model = model_cls._from_config(config)
        model.init_buffers_post_meta()
        _init_moe_weights(model, config.initializer_range)
        _fill_hash_tables(model)

    prime_state_dict = {key: value.clone() for key, value in model.state_dict().items()}
    hf_state_dict = model.convert_to_hf(dict(model.state_dict()))
    save_state_dict(hf_state_dict, out_dir)
    config.save_pretrained(out_dir)

    reloaded: dict[str, Tensor] = load_state_dict(out_dir)
    model.convert_to_prime(reloaded)
    assert reloaded.keys() == prime_state_dict.keys(), reloaded.keys() ^ prime_state_dict.keys()
    for key, value in prime_state_dict.items():
        assert torch.equal(reloaded[key], value), f"{arch}: roundtrip mismatch at {key}"
    return out_dir
