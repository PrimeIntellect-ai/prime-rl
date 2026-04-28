"""Create a tiny Qwen3.5-MoE text-only model for debugging GDN CP forward.

Mirrors Qwen3.6 head structure (num_k_heads=16, num_v_heads=32, both divisible by 2 for cp=2)
but with a small hidden_size and few layers so it fits trivially on a single GPU.

Usage:
    uv run python scripts/mini_qwen3_5_moe_text.py --output-dir ./outputs/mini-qwen35-moe-text
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM


def make_config() -> Qwen3_5MoeTextConfig:
    return Qwen3_5MoeTextConfig(
        vocab_size=248320,
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        attention_bias=False,
        # GDN: keep Qwen3.6 head ratio so cp=2 works
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        # MoE
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        num_experts_per_tok=2,
        num_experts=4,
        # Force at least one linear_attention layer at index 0 so the GDN path runs first
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        use_cache=False,
        tie_word_embeddings=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    config = make_config()
    print(f"Creating tiny Qwen3.5 MoE text model: hidden={config.hidden_size} layers={config.num_hidden_layers}")
    print(f"  layer_types={config.layer_types}")
    print(
        f"  linear_num_key_heads={config.linear_num_key_heads} "
        f"linear_num_value_heads={config.linear_num_value_heads}"
    )

    with torch.device("cpu"):
        model = Qwen3_5MoeForCausalLM(config)
    n = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n/1e6:.1f}M")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    AutoTokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B", trust_remote_code=True).save_pretrained(args.output_dir)
    print(f"  Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
