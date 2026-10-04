"""Tiny per-architecture configs and deterministic weights shared by the multi-GPU model tests."""

import zlib

import torch

from prime_rl.trainer.models.base import PrimeModel
from prime_rl.trainer.models.registry import build_model_config, get_model_cls
from prime_rl.utils.utils import default_dtype

_ATTN = dict(num_attention_heads=4, num_key_value_heads=2, head_dim=64)
_MOE = dict(num_experts=8, num_experts_per_tok=2, moe_intermediate_size=128)
_QWEN3_5_TEXT = dict(
    vocab_size=512,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=4,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    max_position_embeddings=1024,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_num_key_heads=4,
    linear_num_value_heads=8,
    **_ATTN,
)

# Tiny `config.json`s, one per architecture. Shapes respect each family's kernel constraints
# (e.g. DSA sparse MLA needs kv_lora_rank + qk_rope_head_dim == 576 and index_topk % 64 == 0;
# the DeepSeek V4 attention kernel needs >= 32 heads).
ARCH_CONFIGS: dict[str, dict] = {
    "llama": dict(
        model_type="llama",
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        max_position_embeddings=1024,
        rope_theta=500000.0,
        **_ATTN,
    ),
    "qwen3": dict(
        model_type="qwen3",
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        max_position_embeddings=1024,
        **_ATTN,
    ),
    "qwen3_moe": dict(
        model_type="qwen3_moe",
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        max_position_embeddings=1024,
        norm_topk_prob=True,
        **_ATTN,
        **_MOE,
    ),
    "afmoe": dict(
        model_type="afmoe",
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_dense_layers=1,
        max_position_embeddings=1024,
        global_attn_every_n_layers=2,
        sliding_window=64,
        num_shared_experts=1,
        **_ATTN,
        **_MOE,
    ),
    "minimax_m2": dict(
        model_type="minimax_m2",
        vocab_size=512,
        hidden_size=256,
        intermediate_size=128,
        num_hidden_layers=2,
        max_position_embeddings=1024,
        num_local_experts=8,
        num_experts_per_tok=2,
        rotary_dim=32,
        use_qk_norm=True,
        **_ATTN,
    ),
    "glm4_moe": dict(
        model_type="glm4_moe",
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=3,
        first_k_dense_replace=1,
        max_position_embeddings=1024,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        use_qk_norm=True,
        **_ATTN,
    ),
    "glm_moe_dsa": dict(
        model_type="glm_moe_dsa",
        vocab_size=512,
        pad_token_id=0,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=3,
        first_k_dense_replace=1,
        max_position_embeddings=1024,
        # The sparse-MLA backward kernel tiles heads in blocks of 64.
        num_attention_heads=64,
        num_key_value_heads=64,
        q_lora_rank=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=64,
        v_head_dim=64,
        index_n_heads=8,
        index_head_dim=128,
        index_topk=64,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
    ),
    "gpt_oss": dict(
        model_type="gpt_oss",
        vocab_size=512,
        hidden_size=256,
        intermediate_size=128,
        num_hidden_layers=2,
        max_position_embeddings=1024,
        num_local_experts=8,
        num_experts_per_tok=2,
        sliding_window=64,
        **_ATTN,
    ),
    "laguna": dict(
        model_type="laguna",
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_attention_heads_per_layer=[4] * 4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=1024,
        layer_types=["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
        sliding_window=64,
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        rope_parameters={
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500000.0,
                "factor": 4.0,
                "original_max_position_embeddings": 256,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
    ),
    "nemotron_h": dict(
        model_type="nemotron_h",
        vocab_size=512,
        hidden_size=256,
        hybrid_override_pattern="ME*E",
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=1024,
        intermediate_size=512,
        expand=2,
        mamba_num_heads=8,
        mamba_head_dim=64,
        ssm_state_size=64,
        # Mamba context parallelism splits the SSM groups across CP ranks.
        n_groups=2,
        conv_kernel=4,
        chunk_size=64,
        n_routed_experts=8,
        n_shared_experts=1,
        moe_intermediate_size=128,
        moe_shared_expert_intermediate_size=128,
        num_experts_per_tok=2,
        norm_topk_prob=True,
    ),
    "qwen3_5": dict(model_type="qwen3_5_text", **_QWEN3_5_TEXT),
    "qwen3_5_moe": dict(model_type="qwen3_5_moe_text", shared_expert_intermediate_size=128, **_QWEN3_5_TEXT, **_MOE),
    # The VLM body on text-only data: the vision tower still runs on dummy pixels.
    "qwen3_5_vlm": dict(
        model_type="qwen3_5",
        text_config=dict(model_type="qwen3_5_text", **_QWEN3_5_TEXT),
        vision_config=dict(depth=1, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=256),
        image_token_id=500,
    ),
    "qwen3_8_flash_next": dict(
        model_type="qwen4_exp_text",
        vocab_size=512,
        eos_token_id=511,
        hidden_size=256,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        indexer_n_heads=4,
        indexer_head_dim=64,
        indexer_budget=128,
        indexer_compress_ratio=4,
        hc_count=2,
        hc_lowrank=32,
        ple_layer_ids=[2],
        ple_embed_dim=64,
        heads_per_ngram=2,
        ngram_vocab_size_base=1024,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=4,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
    ),
    "deepseek_v4": dict(
        model_type="deepseek_v4",
        vocab_size=512,
        hidden_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=5,
        num_attention_heads=32,
        num_key_value_heads=1,
        head_dim=32,
        q_lora_rank=64,
        partial_rotary_factor=0.5,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        max_position_embeddings=1024,
        sliding_window=6,
        o_groups=2,
        o_lora_rank=16,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
            "compressed_sparse_attention",
            "sliding_attention",
        ],
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
        index_n_heads=64,
        index_head_dim=128,
        index_topk=2,
        n_routed_experts=8,
        num_experts_per_tok=3,
        n_shared_experts=1,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        num_hash_layers=2,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
    ),
}


def seeded_generator(name: str) -> torch.Generator:
    return torch.Generator().manual_seed(zlib.crc32(name.encode()))


def init_tensor(name: str, tensor: torch.Tensor, config_dict: dict) -> torch.Tensor:
    """A deterministic, well-conditioned value for every parameter and persistent buffer."""
    generator = seeded_generator(name)
    if name.endswith("tid2eid"):
        # DeepSeek V4 hash routing: each token goes to `top_k` distinct experts.
        num_experts = config_dict["n_routed_experts"]
        rows = [torch.randperm(num_experts, generator=generator)[: tensor.shape[1]] for _ in range(tensor.shape[0])]
        return torch.stack(rows).to(tensor.dtype)
    if not tensor.is_floating_point():
        return tensor  # constants computed at construction (e.g. n-gram hash tables)
    if tensor.ndim >= 2:
        return (torch.randn(tensor.shape, generator=generator) * 0.02).to(tensor.dtype)
    if "norm" in name:
        return torch.ones(tensor.shape, dtype=tensor.dtype)
    return torch.zeros(tensor.shape, dtype=tensor.dtype)


def build_model(arch: str, device: torch.device | str = "cpu", dtype: torch.dtype = torch.bfloat16) -> PrimeModel:
    model_config = build_model_config(ARCH_CONFIGS[arch])
    with torch.device(device), default_dtype(dtype):
        return get_model_cls(model_config.model_type)(model_config)


def reference_state_dict(arch: str) -> dict[str, torch.Tensor]:
    """Deterministic PrimeRL-format weights (and persistent buffers) for the arch, in bf16 on CPU.

    Built on CPU rather than meta so buffers computed at construction hold real values.
    """
    model = build_model(arch)
    return {name: init_tensor(name, tensor, ARCH_CONFIGS[arch]) for name, tensor in model.state_dict().items()}
