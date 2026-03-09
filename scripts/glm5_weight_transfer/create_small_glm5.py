"""Create a small GLM5 (GlmMoeDsa) checkpoint for testing weight transfer.

Creates two checkpoints:
1. bf16 checkpoint - for PrimeRL model
2. FP8 checkpoint - for vLLM model (compatible with GLM-5-FP8 format)

Architecture: 2 layers (layer 0 dense, layer 1 MoE).
Uses same head dimensions as real GLM-5 (required by vLLM MLA backends)
but with fewer layers, experts, and vocab.
"""

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

BASE_DIR = Path("/home/matej/dev/prime-rl/scripts/glm5_weight_transfer/checkpoints")
BF16_DIR = BASE_DIR / "glm5-tiny-bf16"
FP8_DIR = BASE_DIR / "glm5-tiny-fp8"

# Exact GLM-5 dimensions, just 2 layers (1 dense + 1 MoE with 256 experts)
CONFIG = {
    "architectures": ["GlmMoeDsaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "dtype": "bfloat16",
    "eos_token_id": [154820, 154827, 154829],
    "ep_size": 1,
    "first_k_dense_replace": 1,
    "hidden_act": "silu",
    "head_dim": 64,
    "hidden_size": 6144,
    "index_head_dim": 128,
    "index_n_heads": 32,
    "index_topk": 2048,
    "indexer_rope_interleave": True,
    "initializer_range": 0.02,
    "intermediate_size": 12288,
    "kv_lora_rank": 512,
    "max_position_embeddings": 4096,
    "moe_intermediate_size": 2048,
    "moe_layer_freq": 1,
    "model_type": "glm_moe_dsa",
    "n_group": 1,
    "n_routed_experts": 16,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 64,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 2,
    "num_key_value_heads": 64,
    "num_nextn_predict_layers": 0,
    "pad_token_id": 154820,
    "pretraining_tp": 1,
    "q_lora_rank": 2048,
    "qk_head_dim": 256,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "rms_norm_eps": 1e-05,
    "rope_interleave": True,
    "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "tie_word_embeddings": False,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "transformers_version": "5.0.2.dev0",
    "use_cache": True,
    "v_head_dim": 256,
    "vocab_size": 154880,
}

# Derived dimensions
H = CONFIG["hidden_size"]
V = CONFIG["vocab_size"]
I = CONFIG["intermediate_size"]
MI = CONFIG["moe_intermediate_size"]
NE = CONFIG["n_routed_experts"]
NH = CONFIG["num_attention_heads"]
KV_RANK = CONFIG["kv_lora_rank"]
Q_RANK = CONFIG["q_lora_rank"]
QK_ROPE = CONFIG["qk_rope_head_dim"]
QK_NOPE = CONFIG["qk_nope_head_dim"]
QK_HEAD = QK_NOPE + QK_ROPE
V_HEAD = CONFIG["v_head_dim"]
IDX_NH = CONFIG["index_n_heads"]
IDX_HD = CONFIG["index_head_dim"]


def rand_bf16(*shape):
    return torch.randn(*shape, dtype=torch.bfloat16) * 0.02


def create_attention_weights(layer_idx: int) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer_idx}.self_attn"
    return {
        f"{prefix}.q_a_proj.weight": rand_bf16(Q_RANK, H),
        f"{prefix}.q_a_layernorm.weight": torch.ones(Q_RANK, dtype=torch.bfloat16),
        f"{prefix}.q_b_proj.weight": rand_bf16(NH * QK_HEAD, Q_RANK),
        f"{prefix}.kv_a_proj_with_mqa.weight": rand_bf16(KV_RANK + QK_ROPE, H),
        f"{prefix}.kv_a_layernorm.weight": torch.ones(KV_RANK, dtype=torch.bfloat16),
        f"{prefix}.kv_b_proj.weight": rand_bf16(NH * (QK_NOPE + V_HEAD), KV_RANK),
        f"{prefix}.o_proj.weight": rand_bf16(H, NH * V_HEAD),
        # Indexer
        f"{prefix}.indexer.wq_b.weight": rand_bf16(IDX_NH * IDX_HD, Q_RANK),
        f"{prefix}.indexer.wk.weight": rand_bf16(IDX_HD, H),
        f"{prefix}.indexer.k_norm.weight": torch.ones(IDX_HD, dtype=torch.float32),
        f"{prefix}.indexer.k_norm.bias": torch.zeros(IDX_HD, dtype=torch.float32),
        f"{prefix}.indexer.weights_proj.weight": rand_bf16(IDX_NH, H),
    }


def create_dense_mlp_weights(layer_idx: int) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer_idx}.mlp"
    return {
        f"{prefix}.gate_proj.weight": rand_bf16(I, H),
        f"{prefix}.up_proj.weight": rand_bf16(I, H),
        f"{prefix}.down_proj.weight": rand_bf16(H, I),
    }


def create_moe_weights(layer_idx: int) -> dict[str, torch.Tensor]:
    """Create MoE weights in HF checkpoint format (per-expert)."""
    prefix = f"model.layers.{layer_idx}.mlp"
    weights = {}

    # Router
    weights[f"{prefix}.gate.weight"] = rand_bf16(NE, H)
    weights[f"{prefix}.gate.e_score_correction_bias"] = torch.zeros(NE, dtype=torch.bfloat16)

    # Routed experts (per-expert format for HF compatibility)
    for j in range(NE):
        weights[f"{prefix}.experts.{j}.gate_proj.weight"] = rand_bf16(MI, H)
        weights[f"{prefix}.experts.{j}.up_proj.weight"] = rand_bf16(MI, H)
        weights[f"{prefix}.experts.{j}.down_proj.weight"] = rand_bf16(H, MI)

    # Shared expert
    weights[f"{prefix}.shared_experts.gate_proj.weight"] = rand_bf16(MI, H)
    weights[f"{prefix}.shared_experts.up_proj.weight"] = rand_bf16(MI, H)
    weights[f"{prefix}.shared_experts.down_proj.weight"] = rand_bf16(H, MI)

    return weights


def create_norm_weights(layer_idx: int) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer_idx}"
    return {
        f"{prefix}.input_layernorm.weight": torch.ones(H, dtype=torch.bfloat16),
        f"{prefix}.post_attention_layernorm.weight": torch.ones(H, dtype=torch.bfloat16),
    }


def quantize_to_fp8_blockwise(weight: torch.Tensor, block_size: tuple[int, int] = (128, 128)):
    """Quantize a bf16 weight tensor to FP8 e4m3 with block-wise scaling."""
    assert weight.ndim == 2
    rows, cols = weight.shape
    br, bc = block_size

    pad_rows = (br - rows % br) % br
    pad_cols = (bc - cols % bc) % bc
    if pad_rows > 0 or pad_cols > 0:
        padded = torch.zeros(rows + pad_rows, cols + pad_cols, dtype=weight.dtype, device=weight.device)
        padded[:rows, :cols] = weight
    else:
        padded = weight

    pr, pc = padded.shape
    blocks = padded.reshape(pr // br, br, pc // bc, bc).permute(0, 2, 1, 3)

    max_abs = blocks.float().abs().amax(dim=(2, 3))
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = max_abs / fp8_max
    scale = scale.clamp(min=1e-12)

    blocks_scaled = blocks.float() / scale[:, :, None, None]
    blocks_fp8 = blocks_scaled.clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)

    quantized = blocks_fp8.permute(0, 2, 1, 3).reshape(pr, pc)[:rows, :cols].contiguous()
    scale_out = scale.float().contiguous()

    return quantized, scale_out


def create_bf16_checkpoint():
    """Create the bf16 checkpoint using PrimeRL model initialization (handles MoE correctly)."""
    print("Creating bf16 checkpoint...")

    from prime_rl.trainer.models import AutoModelForCausalLMPrimeRL
    from prime_rl.trainer.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
    from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import convert_tt_to_hf_moe

    if BF16_DIR.exists():
        shutil.rmtree(BF16_DIR)
    BF16_DIR.mkdir(parents=True)

    # Save config
    with open(BF16_DIR / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    # Use PrimeRL's config (handles mlp_layer_types correctly)
    config = GlmMoeDsaConfig(**CONFIG)
    print(f"  mlp_layer_types: {config.mlp_layer_types}")

    # Create model with PrimeRL implementation (handles MoE)
    model = AutoModelForCausalLMPrimeRL.from_config(config, dtype=torch.bfloat16)

    # Save in PrimeRL format directly (w1/w2/w3, router.gate, shared_expert)
    # This is what PrimeRL's from_pretrained expects
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    # Remove non-persistent buffers that shouldn't be in checkpoint
    state_dict = {k: v for k, v in state_dict.items() if "tokens_per_expert" not in k}
    save_file(state_dict, BF16_DIR / "model.safetensors")

    # Also save HF-format version for vLLM checkpoint loading
    from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import convert_tt_to_hf_moe
    hf_state_dict = {k: v.clone() for k, v in state_dict.items()}
    convert_tt_to_hf_moe(hf_state_dict)

    hf_dir = BASE_DIR / "glm5-tiny-bf16-hf"
    if hf_dir.exists():
        shutil.rmtree(hf_dir)
    hf_dir.mkdir(parents=True)
    save_file(hf_state_dict, hf_dir / "model.safetensors")
    with open(hf_dir / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tok.save_pretrained(BF16_DIR)

    from transformers import AutoTokenizer as AT2
    AT2.from_pretrained("Qwen/Qwen3-0.6B").save_pretrained(hf_dir)

    print(f"  Saved bf16 checkpoint to {BF16_DIR} (PrimeRL format)")
    print(f"  Saved HF-format copy to {hf_dir}")
    print(f"  Total parameters: {sum(p.numel() for p in state_dict.values()):,}")
    return hf_state_dict  # Return HF format for FP8 conversion


def create_fp8_checkpoint(bf16_state_dict: dict[str, torch.Tensor]):
    """Create the FP8 checkpoint from the bf16 checkpoint."""
    print("Creating FP8 checkpoint...")

    fp8_config = dict(CONFIG)

    modules_to_not_convert = [
        "lm_head",
        "model.embed_tokens",
    ]
    for layer_idx in range(CONFIG["num_hidden_layers"]):
        prefix = f"model.layers.{layer_idx}"
        modules_to_not_convert.extend([
            f"{prefix}.input_layernorm",
            f"{prefix}.post_attention_layernorm",
            f"{prefix}.self_attn.indexer",
            f"{prefix}.self_attn.kv_a_layernorm",
            f"{prefix}.self_attn.q_a_layernorm",
        ])
        if layer_idx >= CONFIG["first_k_dense_replace"]:
            modules_to_not_convert.extend([
                f"{prefix}.mlp.gate",
            ])

    fp8_config["quantization_config"] = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": modules_to_not_convert,
    }

    fp8_state_dict = {}
    for name, tensor in bf16_state_dict.items():
        should_convert = True
        for exclude in modules_to_not_convert:
            if name.startswith(exclude):
                should_convert = False
                break

        if should_convert and tensor.ndim == 2 and name.endswith(".weight"):
            quantized, scale = quantize_to_fp8_blockwise(tensor)
            fp8_state_dict[name] = quantized
            # Replace only the trailing .weight with .weight_scale_inv
            scale_name = name[:-len(".weight")] + ".weight_scale_inv"
            fp8_state_dict[scale_name] = scale
        else:
            fp8_state_dict[name] = tensor.clone()

    if FP8_DIR.exists():
        shutil.rmtree(FP8_DIR)
    FP8_DIR.mkdir(parents=True)
    save_file(fp8_state_dict, FP8_DIR / "model.safetensors")

    with open(FP8_DIR / "config.json", "w") as f:
        json.dump(fp8_config, f, indent=2)

    # Copy tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tok.save_pretrained(FP8_DIR)

    print(f"  Saved FP8 checkpoint to {FP8_DIR}")
    return fp8_state_dict


if __name__ == "__main__":
    torch.manual_seed(42)
    bf16_sd = create_bf16_checkpoint()
    create_fp8_checkpoint(bf16_sd)
    print("Done!")
