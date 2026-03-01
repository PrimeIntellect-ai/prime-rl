"""Create and verify a mini Qwen3-VL model for local e2e testing.

Creates a small vision-language model with random weights, saves it with
a tokenizer/processor, and verifies the forward pass works for both
text-only and multimodal inputs.

Usage:
    # Create and verify
    uv run python scripts/mini_vlm.py --output-dir ./mini-qwen3-vl

    # Verify only (on an existing checkpoint)
    uv run python scripts/mini_vlm.py --output-dir ./mini-qwen3-vl --verify-only
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, Qwen3VLConfig

from prime_rl.utils.logger import setup_logger

setup_logger("info")

TOKENIZER_SOURCE = "Qwen/Qwen3-VL-4B-Instruct"

VLM_CONFIG = Qwen3VLConfig(
    text_config=dict(
        vocab_size=151936,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [12, 10, 10],
            "rope_type": "default",
            "rope_theta": 5000000,
        },
        tie_word_embeddings=True,
    ),
    vision_config=dict(
        depth=4,
        hidden_size=128,
        intermediate_size=256,
        num_heads=4,
        out_hidden_size=256,  # must match text hidden_size
        deepstack_visual_indexes=[1, 2, 3],
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
    ),
    tie_word_embeddings=True,
)


def create(output_dir: Path) -> None:
    print("Creating mini Qwen3-VL model...")
    print(
        f"  text: hidden_size={VLM_CONFIG.text_config.hidden_size}, layers={VLM_CONFIG.text_config.num_hidden_layers}"
    )
    print(f"  vision: hidden_size={VLM_CONFIG.vision_config.hidden_size}, depth={VLM_CONFIG.vision_config.depth}")

    with torch.device("cpu"):
        model = AutoModelForImageTextToText.from_config(VLM_CONFIG)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count / 1e6:.1f}M")

    print(f"  Copying tokenizer & processor from {TOKENIZER_SOURCE}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SOURCE, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(TOKENIZER_SOURCE, trust_remote_code=True, use_fast=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"  Saved to {output_dir}")


def verify(model_dir: Path) -> None:
    print(f"Verifying forward pass for {model_dir}...")

    config = Qwen3VLConfig.from_pretrained(str(model_dir))
    config._attn_implementation = "sdpa"

    with torch.device("cpu"):
        model = AutoModelForImageTextToText.from_pretrained(str(model_dir), config=config)
    model.eval()

    # Text-only forward
    input_ids = torch.randint(0, config.text_config.vocab_size, (1, 32))
    position_ids = torch.arange(32).unsqueeze(0)

    with torch.no_grad():
        out = model(input_ids=input_ids, position_ids=position_ids)
    print(f"  Text-only logits shape: {out.logits.shape}")

    # Multimodal forward
    image_token_id = config.image_token_id
    spatial_merge = config.vision_config.spatial_merge_size
    grid_thw = torch.tensor([[1, 2 * spatial_merge, 2 * spatial_merge]])
    num_merged_tokens = int(grid_thw[0, 0] * (grid_thw[0, 1] // spatial_merge) * (grid_thw[0, 2] // spatial_merge))
    num_patches = int(grid_thw[0, 0] * grid_thw[0, 1] * grid_thw[0, 2])
    patch_dim = 3 * config.vision_config.temporal_patch_size * config.vision_config.patch_size**2

    mm_input_ids = torch.tensor([[1, 2] + [image_token_id] * num_merged_tokens + [3, 4]])
    pixel_values = torch.randn(num_patches, patch_dim)

    with torch.no_grad():
        out = model(input_ids=mm_input_ids, pixel_values=pixel_values, image_grid_thw=grid_thw)
    print(f"  Multimodal logits shape: {out.logits.shape}")

    print("  Verification passed.")


def main():
    parser = argparse.ArgumentParser(description="Create and verify a mini Qwen3-VL model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true", help="Skip creation, only verify an existing model")
    args = parser.parse_args()

    if not args.verify_only:
        create(args.output_dir)

    verify(args.output_dir)


if __name__ == "__main__":
    main()
