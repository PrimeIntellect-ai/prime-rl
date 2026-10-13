"""Seeded tiny HF checkpoints for the production MoE archs, built from explicit config kwargs."""

from pathlib import Path

import torch
from torch import Tensor

from prime_rl.trainer.models import get_custom_causal_lm_cls
from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.deepseek_v4.moe import DeepseekV4HashRouter
from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaConfig
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig
from prime_rl.utils.utils import default_dtype
from prime_rl.utils.weights import load_state_dict, save_state_dict

SEED = 0

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
}


def _fill_hash_tables(model: torch.nn.Module) -> None:
    """Give every DSv4 hash router a `tid2eid` table with distinct experts per token, as a real one has."""
    for router in model.modules():
        if isinstance(router, DeepseekV4HashRouter):
            rows = [torch.randperm(router.num_experts)[: router.top_k] for _ in range(router.tid2eid.shape[0])]
            router.tid2eid.copy_(torch.stack(rows))


def write_tiny_checkpoint(arch: str, out_dir: Path) -> Path:
    """Write a seeded tiny HF safetensors checkpoint for `arch` and verify its HF -> PrimeRL roundtrip."""
    config = TINY_CONFIGS[arch]
    model_cls = get_custom_causal_lm_cls(config)
    torch.manual_seed(SEED)
    with torch.device("cpu"), default_dtype(torch.bfloat16), torch.no_grad():
        model = model_cls._from_config(config)
        _fill_hash_tables(model)

    prime_state_dict = {key: value.clone() for key, value in model.state_dict().items()}
    hf_state_dict = model.convert_to_hf(dict(model.state_dict()))
    assert model.is_hf_state_dict(hf_state_dict)
    save_state_dict(hf_state_dict, out_dir)
    config.save_pretrained(out_dir)

    reloaded: dict[str, Tensor] = load_state_dict(out_dir)
    model.convert_to_prime(reloaded)
    assert reloaded.keys() == prime_state_dict.keys(), reloaded.keys() ^ prime_state_dict.keys()
    for key, value in prime_state_dict.items():
        assert torch.equal(reloaded[key], value), f"{arch}: roundtrip mismatch at {key}"
    return out_dir
