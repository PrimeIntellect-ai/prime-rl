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
from transformers import AutoConfig, AutoTokenizer
from transformers import Glm4MoeForCausalLM as HFGlm4MoeForCausalLM
from transformers import GlmMoeDsaForCausalLM as HFGlmMoeDsaForCausalLM

from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig
from prime_rl.trainer.models.glm4_moe import Glm4MoeForCausalLM as PrimeRLGlm4MoeForCausalLM
from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaConfig
from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaForCausalLM as PrimeRLGlmMoeDsaForCausalLM
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.utils import default_dtype

setup_logger("info")

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
    "glm_moe_dsa": {
        "config_class": GlmMoeDsaConfig,
        "config_kwargs": dict(
            vocab_size=154880,
            hidden_size=1024,
            intermediate_size=2048,
            moe_intermediate_size=256,
            num_hidden_layers=12,
            num_attention_heads=8,
            num_key_value_heads=8,
            kv_lora_rank=128,
            q_lora_rank=256,
            qk_rope_head_dim=32,
            v_head_dim=64,
            qk_nope_head_dim=32,
            hidden_act="silu",
            max_position_embeddings=4096,
            rms_norm_eps=1e-5,
            rope_interleave=True,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            first_k_dense_replace=2,
            norm_topk_prob=True,
            use_grouped_mm=False,
            pad_token_id=151329,
            eos_token_id=[151329, 151336, 151338],
        ),
        "hf_model_class": HFGlmMoeDsaForCausalLM,
        "prime_model_class": PrimeRLGlmMoeDsaForCausalLM,
        "tokenizer_source": "zai-org/GLM-5",
    },
}


def create(arch: str, output_dir: Path) -> None:
    preset = ARCH_PRESETS[arch]
    config = preset["config_class"](**preset["config_kwargs"])

    print(f"Creating mini {arch} model...")
    print(
        f"  hidden_size={config.hidden_size}, layers={config.num_hidden_layers}, "
        f"experts={config.n_routed_experts}, moe_intermediate_size={config.moe_intermediate_size}"
    )

    with torch.device("cpu"):
        model = preset["hf_model_class"](config)

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
    print(f"Verifying HF <-> PrimeRL roundtrip for {model_dir}...")

    config = AutoConfig.from_pretrained(str(model_dir))
    config._attn_implementation = "sdpa"

    with torch.device("cuda"), default_dtype(torch.float32):
        hf_model = preset["hf_model_class"].from_pretrained(str(model_dir), config=config)
        prime_model = preset["prime_model_class"]._from_config(config)

    with torch.no_grad():
        state_dict = hf_model.state_dict()
        prime_model.convert_to_prime(state_dict)
        prime_model.load_state_dict(state_dict)

    inject_prime_lm_head(prime_model, chunk_size=None)

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, config.vocab_size, (1, 64))
        position_ids = torch.arange(1, 65).unsqueeze(0)

    hf_output = hf_model(input_ids=input_ids, position_ids=position_ids)
    prime_output = prime_model(input_ids, position_ids)

    logits_diff = prime_output["logits"] - hf_output.logits
    max_diff = logits_diff.abs().max().item()
    print(f"  HF vs PrimeRL max logits diff: {max_diff:.6f}")
    assert max_diff < 0.1, f"HF vs PrimeRL logits mismatch: max diff {max_diff}"

    with torch.no_grad():
        roundtrip_state_dict = prime_model.convert_to_hf(prime_model.state_dict())
    with torch.device("cuda"), default_dtype(torch.float32):
        hf_roundtrip = preset["hf_model_class"]._from_config(config)
        hf_roundtrip.load_state_dict(roundtrip_state_dict)

    hf_roundtrip_output = hf_roundtrip(input_ids=input_ids, position_ids=position_ids)
    roundtrip_diff = hf_roundtrip_output.logits - hf_output.logits
    max_roundtrip_diff = roundtrip_diff.abs().max().item()
    print(f"  HF -> PrimeRL -> HF roundtrip max logits diff: {max_roundtrip_diff:.6f}")
    assert max_roundtrip_diff < 0.1, f"Roundtrip logits mismatch: max diff {max_roundtrip_diff}"

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
