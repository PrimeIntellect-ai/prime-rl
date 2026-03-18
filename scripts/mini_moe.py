"""Create and verify a mini MoE model for testing.

Creates a small MoE model with random weights, saves it with a tokenizer,
and verifies the HF <-> PrimeRL weight conversion roundtrip.

Usage:
    # Create and verify
    uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./mini-glm-moe

    # Verify only (on an existing checkpoint)
    uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./mini-glm-moe --verify-only
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import Glm4MoeForCausalLM as HFGlm4MoeForCausalLM
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration as HFQwen3_5MoeVLM,
)

from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig
from prime_rl.trainer.models.glm4_moe import Glm4MoeForCausalLM as PrimeRLGlm4MoeForCausalLM
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.minimax_m2 import MiniMaxM2Config
from prime_rl.trainer.models.minimax_m2 import MiniMaxM2ForCausalLM as PrimeRLMiniMaxM2ForCausalLM
from prime_rl.trainer.models.qwen3_5_moe import Qwen3_5MoeForCausalLM as PrimeRLQwen3_5MoeVLM
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.utils import default_dtype

setup_logger("info")


def _qwen3_5_moe_vlm_config():
    """Build a tiny composite VLM config for Qwen3.5 MoE."""
    config = AutoConfig.from_pretrained("Qwen/Qwen3.5-35B-A3B", trust_remote_code=True, attn_implementation="sdpa")
    config.use_cache = False

    tc = config.text_config
    tc.vocab_size = 256
    tc.hidden_size = 256
    tc.num_hidden_layers = 2
    tc.num_attention_heads = 4
    tc.num_key_value_heads = 2
    tc.head_dim = 64
    tc.moe_intermediate_size = 128
    tc.shared_expert_intermediate_size = 128
    tc.num_experts = 4
    tc.num_experts_per_tok = 2
    tc.max_position_embeddings = 512
    tc.linear_key_head_dim = 32
    tc.linear_value_head_dim = 32
    tc.linear_num_key_heads = 4
    tc.linear_num_value_heads = 8
    tc.layer_types = ["full_attention", "linear_attention"]
    tc.use_cache = False

    vc = config.vision_config
    vc.depth = 2
    vc.hidden_size = 128
    vc.intermediate_size = 256
    vc.num_heads = 4
    vc.out_hidden_size = tc.hidden_size

    config.image_token_id = 250
    config.video_token_id = 251
    config.vision_start_token_id = 252
    config.vision_end_token_id = 253
    return config


ARCH_PRESETS = {
    "glm4_moe": {
        "config_class": Glm4MoeConfig,
        "config_kwargs": dict(
            vocab_size=151552,
            hidden_size=1024,
            intermediate_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
            hidden_act="silu",
            max_position_embeddings=131072,
            rms_norm_eps=1e-5,
            rope_theta=1000000,
            attention_bias=True,
            partial_rotary_factor=0.5,
            moe_intermediate_size=256,
            n_routed_experts=8,
            num_experts_per_tok=4,
            n_shared_experts=1,
            first_k_dense_replace=1,
            norm_topk_prob=True,
            use_qk_norm=False,
            use_grouped_mm=False,
            pad_token_id=151329,
            eos_token_id=[151329, 151336, 151338],
        ),
        "hf_model_class": HFGlm4MoeForCausalLM,
        "prime_model_class": PrimeRLGlm4MoeForCausalLM,
        "tokenizer_source": "THUDM/GLM-4-9B-0414",
    },
    "minimax_m2": {
        "config_class": MiniMaxM2Config,
        "config_kwargs": dict(
            vocab_size=200064,
            hidden_size=512,
            intermediate_size=256,
            num_hidden_layers=12,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=64,
            hidden_act="silu",
            max_position_embeddings=4096,
            rms_norm_eps=1e-6,
            rope_theta=5000000,
            rotary_dim=32,
            num_local_experts=8,
            num_experts_per_tok=4,
            scoring_func="sigmoid",
            use_routing_bias=True,
            use_qk_norm=True,
            qk_norm_type="per_layer",
            use_grouped_mm=False,
            auto_map={"AutoModelForCausalLM": "MiniMaxAI/MiniMax-M2.1--modeling_minimax_m2.MiniMaxM2ForCausalLM"},
        ),
        "hf_model_class": None,  # uses AutoModelForCausalLM with trust_remote_code
        "prime_model_class": PrimeRLMiniMaxM2ForCausalLM,
        "tokenizer_source": "MiniMaxAI/MiniMax-M2.1",
    },
    "qwen3_5_moe_vlm": {
        "config_fn": _qwen3_5_moe_vlm_config,
        "hf_model_class": HFQwen3_5MoeVLM,
        "prime_model_class": PrimeRLQwen3_5MoeVLM,
        "tokenizer_source": "Qwen/Qwen3.5-35B-A3B",
        "is_vlm": True,
    },
    # glm_moe_dsa: HF implementation is incorrect, not supported here
}


def _create_hf_model(preset, config):
    """Create an HF model from a preset and config."""
    hf_cls = preset["hf_model_class"]
    if hf_cls is not None:
        return hf_cls(config)
    return AutoModelForCausalLM.from_config(config, trust_remote_code=True)


def _load_hf_model(preset, model_dir, config):
    """Load an HF model from a preset and directory."""
    hf_cls = preset["hf_model_class"]
    if hf_cls is not None:
        return hf_cls.from_pretrained(str(model_dir), config=config)
    return AutoModelForCausalLM.from_pretrained(str(model_dir), config=config, trust_remote_code=True)


def _create_hf_model_from_config(preset, config):
    """Create an empty HF model from config (for roundtrip verification)."""
    hf_cls = preset["hf_model_class"]
    if hf_cls is not None:
        return hf_cls._from_config(config)
    return AutoModelForCausalLM.from_config(config, trust_remote_code=True)


def _build_config(preset):
    """Build model config from preset (handles both config_class and config_fn styles)."""
    if "config_fn" in preset:
        return preset["config_fn"]()
    return preset["config_class"](**preset["config_kwargs"])


def create(arch: str, output_dir: Path) -> None:
    preset = ARCH_PRESETS[arch]
    config = _build_config(preset)

    text_config = getattr(config, "text_config", config)
    print(f"Creating mini {arch} model...")
    print(f"  hidden_size={text_config.hidden_size}, layers={text_config.num_hidden_layers}")

    with torch.device("cpu"):
        model = _create_hf_model(preset, config)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count / 1e6:.1f}M")

    print(f"  Copying tokenizer from {preset['tokenizer_source']}...")
    tokenizer = AutoTokenizer.from_pretrained(preset["tokenizer_source"], trust_remote_code=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Saved to {output_dir}")


def verify(arch: str, model_dir: Path) -> None:
    preset = ARCH_PRESETS[arch]
    is_vlm = preset.get("is_vlm", False)
    print(f"Verifying HF <-> PrimeRL roundtrip for {model_dir}...")

    trust_remote_code = preset["hf_model_class"] is None
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
    config._attn_implementation = "sdpa"
    if hasattr(config, "text_config"):
        config.text_config._attn_implementation = "sdpa"

    text_config = getattr(config, "text_config", config)
    vocab_size = text_config.vocab_size

    hf_model = _load_hf_model(preset, model_dir, config).to(device="cuda", dtype=torch.float32)
    with torch.device("cuda"), default_dtype(torch.float32):
        prime_model = preset["prime_model_class"]._from_config(config)

    with torch.no_grad():
        state_dict = hf_model.state_dict()
        prime_model.convert_to_prime(state_dict)
        prime_model.load_state_dict(state_dict)

    inject_prime_lm_head(prime_model, chunk_size=None)

    # Use tokens in safe range (avoid special VLM token IDs)
    max_token = min(vocab_size, 200) if is_vlm else vocab_size
    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, max_token, (1, 64))
        position_ids = torch.arange(1, 65).unsqueeze(0)

    hf_output = hf_model(input_ids=input_ids, position_ids=position_ids)
    prime_output = prime_model(input_ids, position_ids)

    if is_vlm:
        # HF GatedDeltaNet has a dtype bug in float32 mode; just verify non-NaN output
        assert not torch.isnan(prime_output["logits"]).any(), "PrimeRL VLM output contains NaN"
        assert prime_output["logits"].shape == hf_output.logits.shape
        print("  VLM forward pass verified (shape match, no NaN)")
    else:
        logits_diff = prime_output["logits"] - hf_output.logits
        max_diff = logits_diff.abs().max().item()
        print(f"  HF vs PrimeRL max logits diff: {max_diff:.6f}")
        assert max_diff < 0.1, f"HF vs PrimeRL logits mismatch: max diff {max_diff}"

    # Roundtrip weight conversion: HF -> PrimeRL -> HF
    # Normalize both through the same roundtrip to handle expert format differences
    prime_cls = preset["prime_model_class"]
    with torch.no_grad():
        roundtrip_sd = prime_cls.convert_to_hf(dict(prime_model.state_dict()))
        orig_sd = dict(hf_model.state_dict())
        prime_cls.convert_to_prime(orig_sd)
        prime_cls.convert_to_hf(orig_sd)

    for key in orig_sd:
        assert key in roundtrip_sd, f"Missing key after roundtrip: {key}"
        assert torch.equal(orig_sd[key], roundtrip_sd[key]), f"Roundtrip mismatch at {key}"
    print("  HF -> PrimeRL -> HF weight roundtrip verified")

    print("  Verification passed.")


def main():
    parser = argparse.ArgumentParser(description="Create and verify a mini MoE model")
    parser.add_argument("--arch", choices=list(ARCH_PRESETS.keys()), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true", help="Skip creation, only verify an existing model")
    args = parser.parse_args()

    if not args.verify_only:
        create(args.arch, args.output_dir)

    verify(args.arch, args.output_dir)


if __name__ == "__main__":
    main()
