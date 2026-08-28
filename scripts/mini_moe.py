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
from transformers import DeepseekV4ForCausalLM as HFDeepseekV4ForCausalLM
from transformers import Glm4MoeForCausalLM as HFGlm4MoeForCausalLM
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration as HFQwen3_5MoeVLM,
)

from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.deepseek_v4 import DeepseekV4ForCausalLM as PrimeRLDeepseekV4ForCausalLM
from prime_rl.trainer.models.deepseek_v4.converting_deepseek_v4 import to_on_disk_naming
from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig
from prime_rl.trainer.models.glm4_moe import Glm4MoeForCausalLM as PrimeRLGlm4MoeForCausalLM
from prime_rl.trainer.models.laguna import LagunaConfig
from prime_rl.trainer.models.laguna import LagunaForCausalLM as PrimeRLLagunaForCausalLM
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.minimax_m2 import MiniMaxM2Config
from prime_rl.trainer.models.minimax_m2 import MiniMaxM2ForCausalLM as PrimeRLMiniMaxM2ForCausalLM
from prime_rl.trainer.models.qwen3_5_moe import Qwen3_5MoeForCausalLM as PrimeRLQwen3_5MoeVLM
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.utils import default_dtype
from prime_rl.utils.weights import load_state_dict

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
    # mrope_section must sum to rotary_dim // 2 for the shrunken head_dim
    tc.rope_parameters["mrope_section"] = [3, 3, 2]
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
    "deepseek_v4": {
        "config_class": DeepseekV4Config,
        # Real deepseek-ai/DeepSeek-V4-Flash-0731 config values throughout, only
        # `num_hidden_layers` (and the per-layer schedules that must match its length) are
        # truncated -- this keeps every kernel-relevant shape (head_dim, index_head_dim,
        # sliding_window, compress ratios, o_groups, ...) identical to the real checkpoint,
        # since arbitrary "convenient" small values can silently violate real kernel
        # constraints (e.g. vLLM's fused indexer quant+cache path only supports
        # head_dim in {128, 512}) that a real checkpoint would never actually exercise.
        # `n_routed_experts`/`num_experts_per_tok` are the one exception: expert *count* is a
        # routing cardinality, not a kernel-shape constraint, and the real value (256) makes
        # the checkpoint ~136GB (won't fit one GPU, OOMs `verify()`'s single-GPU fp32 compare)
        # -- reduced here to keep this locally runnable; every expert's own shape stays real.
        "config_kwargs": dict(
            vocab_size=129280,
            hidden_size=4096,
            moe_intermediate_size=2048,
            num_hidden_layers=5,
            num_attention_heads=64,
            num_key_value_heads=1,
            head_dim=512,
            q_lora_rank=1024,
            partial_rotary_factor=64 / 512,  # qk_rope_head_dim=64
            sliding_window=128,
            o_groups=8,
            o_lora_rank=1024,
            rope_scaling={
                "type": "yarn",
                "factor": 16,
                "beta_fast": 32,
                "beta_slow": 1,
                "original_max_position_embeddings": 65536,
            },
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
            index_topk=512,
            n_routed_experts=16,
            num_experts_per_tok=4,
            n_shared_experts=1,
            num_hash_layers=2,
            use_grouped_mm=False,
        ),
        "hf_model_class": HFDeepseekV4ForCausalLM,
        "prime_model_class": PrimeRLDeepseekV4ForCausalLM,
        "tokenizer_source": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "attn_implementation": "eager",  # HF's DeepseekV4Attention doesn't support sdpa
    },
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
            auto_map={"AutoModelForCausalLM": "MiniMaxAI/MiniMax-M2.1--modeling_minimax_m2.MiniMaxM2ForCausalLM"},
        ),
        "hf_model_class": None,  # uses AutoModelForCausalLM with trust_remote_code
        "prime_model_class": PrimeRLMiniMaxM2ForCausalLM,
        "tokenizer_source": "MiniMaxAI/MiniMax-M2.1",
    },
    "laguna": {
        "config_class": LagunaConfig,
        "config_kwargs": dict(
            vocab_size=100352,
            hidden_size=512,
            intermediate_size=2048,
            num_hidden_layers=12,
            num_attention_heads=8,
            num_attention_heads_per_layer=[8] * 12,
            num_key_value_heads=4,
            head_dim=64,
            hidden_act="silu",
            max_position_embeddings=4096,
            rms_norm_eps=1e-6,
            rope_parameters={
                "full_attention": {
                    "rope_type": "yarn",
                    "rope_theta": 500000.0,
                    "factor": 4.0,
                    "original_max_position_embeddings": 1024,
                    "beta_slow": 1.0,
                    "beta_fast": 64.0,
                    "attention_factor": 1.0,
                    "partial_rotary_factor": 0.5,
                },
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0,
                },
            },
            layer_types=["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"] * 3,
            sliding_window=512,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
            num_experts=8,
            num_experts_per_tok=4,
            mlp_layer_types=["dense"] + ["sparse"] * 11,
            moe_routed_scaling_factor=2.5,
            pad_token_id=9,
            bos_token_id=2,
            eos_token_id=[2, 24],
            auto_map={
                "AutoConfig": "poolside/Laguna-XS.2--configuration_laguna.LagunaConfig",
                "AutoModel": "poolside/Laguna-XS.2--modeling_laguna.LagunaModel",
                "AutoModelForCausalLM": "poolside/Laguna-XS.2--modeling_laguna.LagunaForCausalLM",
            },
        ),
        "hf_model_class": None,  # uses Poolside remote modeling code
        "prime_model_class": PrimeRLLagunaForCausalLM,
        "tokenizer_source": "poolside/Laguna-XS.2",
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
    if arch == "deepseek_v4":
        _fixup_deepseek_v4_checkpoint(output_dir)
    print(f"  Saved to {output_dir}")


def _fixup_deepseek_v4_checkpoint(output_dir: Path) -> None:
    """Bring a locally saved DeepSeek V4 checkpoint in line with the real on-disk format.

    Two separate gaps in what `save_pretrained` leaves behind. The key naming is corrected by
    `to_on_disk_naming` (see its docstring for what transformers gets wrong and how that was
    established); vLLM's own `hf_to_vllm_mapper` assumes the real format and fails with
    `KeyError: 'hc_head.hc_base'` otherwise, and this repo's own `conversion_chain` targets the
    real format too. The config needs a `topk_method` backfill, below.
    """
    import json

    from safetensors.torch import load_file, save_file

    path = output_dir / "model.safetensors"
    save_file(to_on_disk_naming(load_file(path)), path, metadata={"format": "pt"})

    # `deepseek-ai/DeepSeek-V4-Flash-0731` ships no chat template: no `chat_template.jinja` in
    # the repo and no `chat_template` key in its tokenizer config. prime-rl's environments call
    # the chat completions endpoint, and the router renders the template itself from the model
    # directory (it takes no template argument), so without one in the checkpoint every rollout
    # comes back 502 and the run dies reporting "10 consecutive zero-output batch equivalents".
    # Deliberately plain: this only has to make the loop runnable on an untrained checkpoint,
    # and inventing a template for a *real* checkpoint would degrade quality silently.
    tokenizer_config_path = output_dir / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text())
    tokenizer_config.setdefault(
        "chat_template",
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}",
    )
    tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2))

    # `topk_method` is a real DeepSeek V4 config field (the real checkpoint sets it to
    # "noaux_tc") that this repo's `DeepseekV4Config` doesn't model at all -- prime-rl's own
    # `DeepseekV4MoE` always behaves as if it were "noaux_tc" (always builds the aux-loss-free
    # `expert_bias` for non-hash layers), but never persists the field, so external consumers
    # like vLLM that gate `e_score_correction_bias` construction on it see it as absent and
    # skip building the parameter entirely. Backfill it so the saved config matches what the
    # real checkpoint (and vLLM) expect.
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text())
    config.setdefault("topk_method", "noaux_tc")
    config_path.write_text(json.dumps(config, indent=2))


def verify(arch: str, model_dir: Path) -> None:
    preset = ARCH_PRESETS[arch]
    is_vlm = preset.get("is_vlm", False)
    print(f"Verifying HF <-> PrimeRL roundtrip for {model_dir}...")

    trust_remote_code = preset["hf_model_class"] is None
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
    attn_implementation = preset.get("attn_implementation", "sdpa")
    config._attn_implementation = attn_implementation
    if hasattr(config, "text_config"):
        config.text_config._attn_implementation = attn_implementation

    text_config = getattr(config, "text_config", config)
    vocab_size = text_config.vocab_size

    hf_model = _load_hf_model(preset, model_dir, config).to(device="cuda", dtype=torch.float32)
    with torch.device("cuda"), default_dtype(torch.float32):
        prime_model = preset["prime_model_class"]._from_config(config)

    # Convert from the *on-disk* checkpoint, not from `hf_model.state_dict()`, because those
    # are two different key namings for any model that transformers carries a conversion
    # registry entry for (DeepSeek V4 does). The trainer reads raw on-disk state dicts in
    # `load_dcp_from_hf`, so this is the naming `conversion_chain` has to handle, and feeding
    # it the in-memory names instead hid a conversion chain that was a complete no-op.
    disk_state_dict = load_state_dict(model_dir)
    with torch.no_grad():
        state_dict = {k: v.to(device="cuda", dtype=torch.float32) for k, v in disk_state_dict.items()}
        prime_model.convert_to_prime(state_dict)
        prime_model.load_state_dict(state_dict)

    inject_prime_lm_head(prime_model, chunk_size=None)

    # Use tokens in safe range (avoid special VLM token IDs)
    max_token = min(vocab_size, 200) if is_vlm else vocab_size
    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, max_token, (1, 64))
        position_ids = torch.arange(1, 65).unsqueeze(0)

    hf_output = hf_model(input_ids=input_ids, position_ids=position_ids)
    # One unpacked document, which is what the probe above builds. Required by every model that
    # derives document boundaries from it rather than from `position_ids`.
    seq_lens = torch.tensor([input_ids.shape[1]], device=input_ids.device)
    prime_output = prime_model(input_ids, position_ids, seq_lens=seq_lens)

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

    # Roundtrip weight conversion: on-disk -> PrimeRL -> on-disk. Compared against the real
    # on-disk keys rather than against the same dict pushed through the same conversion, which
    # a no-op chain satisfies trivially.
    with torch.no_grad():
        roundtrip_sd = prime_model.convert_to_hf(dict(prime_model.state_dict()))

    extra = sorted(set(roundtrip_sd) - set(disk_state_dict))
    assert not extra, f"Roundtrip produced keys absent from the checkpoint: {extra[:5]}"
    for key, value in disk_state_dict.items():
        assert key in roundtrip_sd, f"Missing key after roundtrip: {key}"
        assert torch.equal(roundtrip_sd[key].cpu(), value), f"Roundtrip mismatch at {key}"
    print(f"  on-disk -> PrimeRL -> on-disk weight roundtrip verified ({len(disk_state_dict)} keys)")

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
