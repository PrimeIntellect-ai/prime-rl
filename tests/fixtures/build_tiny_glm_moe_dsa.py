"""Build a tiny GLM MoE DSA checkpoint for integration tests.

Usage:
    python -m tests.fixtures.build_tiny_glm_moe_dsa /tmp/tiny_glm

The resulting directory contains a ``config.json`` + sharded safetensors that
can be loaded both into the trainer (``GlmMoeDsaForCausalLM``) and into vLLM
(as any GLM-5 / ``model_type=glm_moe_dsa`` checkpoint).

Sizes are intentionally tiny to make CI and interactive iteration fast.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from prime_rl.trainer.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from prime_rl.trainer.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM


def tiny_config() -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        kv_lora_rank=64,
        q_lora_rank=128,
        qk_rope_head_dim=32,
        v_head_dim=64,
        qk_nope_head_dim=64,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        max_position_embeddings=2048,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=64,
        pad_token_id=0,
        # Fit inside FP8 block quantization (block_size=128) without padding weirdness.
        use_grouped_mm=False,
    )


def multi_tiny_config() -> GlmMoeDsaConfig:
    """FSDP-R=2-friendly tiny config for multi-rank NIXL tests.

    Every quantized non-expert tensor's rank-local dim-0 shard is a multiple
    of ``BLOCK_SIZE=128``, so per-shard FP8 block quantization tiles cleanly
    with no cross-shard block straddles:
      - fused_qkv_a_proj: sources [256, 256] and [256, 256] → shard rows 128
      - q_b_proj [768, 256] → shard 384
      - kv_b_proj [512, 128] → shard 256
      - o_proj [256, 256] → shard 128
      - indexer.wq_b [1024, 256] → shard 512
      - indexer.wk [256, 256] → shard 128
      - shared_experts.gate_up_proj sources [256, 256] → shard 128
      - shared_experts.down_proj [256, 256] → shard 128
    """
    return GlmMoeDsaConfig(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        kv_lora_rank=128,
        q_lora_rank=256,
        qk_rope_head_dim=128,
        v_head_dim=64,
        qk_nope_head_dim=64,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        max_position_embeddings=2048,
        index_n_heads=4,
        index_head_dim=256,
        index_topk=64,
        pad_token_id=0,
        use_grouped_mm=False,
    )


def medium_config() -> GlmMoeDsaConfig:
    """Bandwidth-realistic config.

    With R=2 EP sharding this yields a ~20 GB bf16 expert shard per rank and a
    ~6.5 GB FP8 push per rank — enough to saturate a single IB rail and amortize
    post/wait overhead.

    Per-rank FP8 expert bytes ≈ num_moe_layers * num_local_experts * 3 * moe * hidden:
        17 * 16 * 3 * 2048 * 4096 ≈ 6.85 GB
    """
    return GlmMoeDsaConfig(
        vocab_size=4096,
        hidden_size=4096,
        intermediate_size=8192,
        moe_intermediate_size=2048,
        num_hidden_layers=18,
        num_attention_heads=32,
        num_key_value_heads=32,
        n_shared_experts=1,
        n_routed_experts=32,
        kv_lora_rank=512,
        q_lora_rank=1024,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=4,
        first_k_dense_replace=1,
        max_position_embeddings=4096,
        index_n_heads=16,
        index_head_dim=64,
        index_topk=256,
        pad_token_id=0,
        use_grouped_mm=False,
    )


_CONFIGS = {
    "tiny": tiny_config,
    "multi_tiny": multi_tiny_config,
    "medium": medium_config,
}


def build_tiny(out_dir: Path, seed: int = 0, size: str = "tiny") -> Path:
    torch.manual_seed(seed)
    cfg = _CONFIGS[size]()
    model = GlmMoeDsaForCausalLM(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    cfg.save_pretrained(out_dir)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", choices=tuple(_CONFIGS.keys()), default="tiny")
    args = ap.parse_args()
    path = build_tiny(args.out_dir, seed=args.seed, size=args.size)
    print(f"Wrote {args.size} GLM MoE DSA to {path}")


if __name__ == "__main__":
    main()
