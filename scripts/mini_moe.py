"""Create and verify a mini MoE model for testing.

Creates a small MoE model with random weights, saves it with a tokenizer, and verifies the HF <->
PrimeRL weight conversion roundtrip. Compares KL divergences between prime-rl and HF
implementations, and top-1 agreement (a noisy metric for random-init models).

How this mirrors production:
  1. bf16 weights, fp32 buffers (inv_freq, expert_bias, ...): the prod forward under FSDP mixed precision
  2. fp32 MoE router gate, per the moe_router_dtype="float32" default
  3. Grouped-mm experts, per the use_grouped_mm=True default (needs a recent GPU arch)
  4. flash_attention_2 on both models, so a logits gap means the port, not two kernels
  5. Prime LM head injected, as setup_model does
  6. seq_lens passed explicitly, but as one document, so packed boundaries stay untested

NOTE: should be taken as a very coarse sanity check against catastrophic incorrectness. Thresholds
are loose and may pass even with moderate correctness bugs. Not a replacement for robust unit
testing.

Usage:
    # Create and verify
    uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./mini-glm-moe

    # Verify only (on an existing checkpoint)
    uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./mini-glm-moe --verify-only
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers.utils.generic
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import Glm4MoeForCausalLM as HFGlm4MoeForCausalLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.minimax_m2.modeling_minimax_m2 import (
    MiniMaxM2RotaryEmbedding as NativeMiniMaxM2RotaryEmbedding,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration as HFQwen3_5MoeVLM,
)
from transformers.utils.output_capturing import OutputRecorder

from prime_rl.trainer.model import apply_fp32_moe_router
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

setup_logger("info")

MINIMAX_M2_REPO = "MiniMaxAI/MiniMax-M2.1"


def _patch_minimax_m2_hub_code() -> None:
    """Backport three transformers refactors into the MiniMax-M2.1 hub modeling code.

    The hub code has not changed since 2026-02-13 and predates transformers moving OutputRecorder
    out of `utils.generic`, turning `compute_default_rope_parameters` into a per-class method on the
    rotary embedding, and respelling `_tied_weights_keys` as a tied-key -> source mapping. Without
    these the module fails to import, then to initialize its weights, then to save. The two
    modernized attributes are borrowed from the native transformers and PrimeRL MiniMax-M2 classes.

    The hub code, not transformers' native MiniMaxM2ForCausalLM, is the reference we want: real
    MiniMax-M2.1 checkpoints use `block_sparse_moe.experts.{j}.w1/w2/w3`, which
    `converting_minimax_m2.py` targets, whereas the native class uses fused
    `mlp.experts.gate_up_proj`.
    """
    transformers.utils.generic.OutputRecorder = OutputRecorder

    def hub_class(name: str) -> type:
        return get_class_from_dynamic_module(f"{MINIMAX_M2_REPO}--modeling_minimax_m2.{name}", MINIMAX_M2_REPO)

    hub_class("MiniMaxM2RotaryEmbedding").compute_default_rope_parameters = staticmethod(
        NativeMiniMaxM2RotaryEmbedding.compute_default_rope_parameters
    )
    hub_class("MiniMaxM2ForCausalLM")._tied_weights_keys = PrimeRLMiniMaxM2ForCausalLM._tied_weights_keys


def _qwen3_5_moe_vlm_config():
    """Build a tiny composite VLM config for Qwen3.5 MoE."""
    config = AutoConfig.from_pretrained(
        "Qwen/Qwen3.5-35B-A3B", trust_remote_code=True, attn_implementation="flash_attention_2"
    )
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
            auto_map={"AutoModelForCausalLM": f"{MINIMAX_M2_REPO}--modeling_minimax_m2.MiniMaxM2ForCausalLM"},
        ),
        "hf_model_class": None,  # uses AutoModelForCausalLM with trust_remote_code
        "prime_model_class": PrimeRLMiniMaxM2ForCausalLM,
        "tokenizer_source": MINIMAX_M2_REPO,
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


def _create_hf_model_from_config(preset, config):
    """Create an empty HF model from config (for roundtrip verification)."""
    hf_cls = preset["hf_model_class"]
    if hf_cls is not None:
        return hf_cls._from_config(config)
    return AutoModelForCausalLM.from_config(config, trust_remote_code=True)


def _compare_distributions(hf_logits: torch.Tensor, prime_logits: torch.Tensor) -> tuple[float, float, float]:
    "Report and return full-vocab KL mean, KL max, and top-1 agreement."
    hf_logprobs = hf_logits.float().log_softmax(-1)
    prime_logprobs = prime_logits.float().log_softmax(-1)
    # F.kl_div computes KL(target || input), so the arguments read in the opposite order to the name
    kl = F.kl_div(input=prime_logprobs, target=hf_logprobs, reduction="none", log_target=True).sum(-1)
    top1 = (hf_logits.argmax(-1) == prime_logits.argmax(-1)).float().mean()
    print(f"  HF vs PrimeRL KL(HF || PrimeRL) per token: mean {kl.mean().item():.3e}, max {kl.max().item():.3e}")
    print(f"  HF vs PrimeRL top-1 agreement: {top1.item():.1%}")
    return kl.mean().item(), kl.max().item(), top1.item()


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
    config._attn_implementation = "flash_attention_2"
    if hasattr(config, "text_config"):
        config.text_config._attn_implementation = "flash_attention_2"
    # The checkpoint is saved in float32, but flash attention only accepts fp16/bf16. `_from_config`
    # reads `config.dtype`, which would otherwise override the ambient default dtype below.
    config.dtype = torch.bfloat16

    text_config = getattr(config, "text_config", config)
    vocab_size = text_config.vocab_size

    # `config.dtype` above already builds bf16 weights. Casting with `.to(dtype=...)` would also cast
    # buffers, quantizing inv_freq and the routing bias that PrimeRL keeps in fp32.
    hf_model = _load_hf_model(preset, model_dir, config).to(device="cuda")
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        prime_model = preset["prime_model_class"]._from_config(config)

    with torch.no_grad():
        state_dict = hf_model.state_dict()
        prime_model.convert_to_prime(state_dict)
        prime_model.load_state_dict(state_dict)

    inject_prime_lm_head(prime_model, chunk_size=None)
    apply_fp32_moe_router(prime_model)

    # Use tokens in safe range (avoid special VLM token IDs)
    max_token = min(vocab_size, 200) if is_vlm else vocab_size
    input_ids = torch.randint(0, max_token, (1, 64), device="cuda")
    position_ids = torch.arange(1, 65, device="cuda").unsqueeze(0)

    seq_lens = torch.tensor([input_ids.shape[1]], device=input_ids.device)
    hf_output = hf_model(input_ids=input_ids, position_ids=position_ids)
    prime_output = prime_model(input_ids, position_ids, seq_lens=seq_lens)

    if is_vlm:
        assert not torch.isnan(prime_output["logits"]).any(), "PrimeRL VLM output contains NaN"
        assert prime_output["logits"].shape == hf_output.logits.shape

    logits_diff = prime_output["logits"] - hf_output.logits
    max_diff = logits_diff.abs().max().item()
    print(f"  HF vs PrimeRL max logits diff: {max_diff:.6f}")
    mean_kl, max_kl, top1 = _compare_distributions(hf_output.logits, prime_output["logits"])
    assert mean_kl < 1e-3, f"HF vs PrimeRL distribution mismatch: mean KL {mean_kl}"
    assert max_kl < 1e-2, f"HF vs PrimeRL distribution mismatch on one token: max KL {max_kl}"
    assert top1 >= 0.75, f"HF vs PrimeRL top-1 agreement too low: {top1}"
    assert max_diff < 1.0, f"HF vs PrimeRL logits mismatch: max diff {max_diff}"

    # Roundtrip weight conversion: HF -> PrimeRL -> HF
    # Normalize both through the same roundtrip to handle expert format differences
    with torch.no_grad():
        roundtrip_sd = prime_model.convert_to_hf(dict(prime_model.state_dict()))
        orig_sd = dict(hf_model.state_dict())
        prime_model.convert_to_prime(orig_sd)
        prime_model.convert_to_hf(orig_sd)

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

    if args.arch == "minimax_m2":
        _patch_minimax_m2_hub_code()

    if not args.verify_only:
        create(args.arch, args.output_dir)

    verify(args.arch, args.output_dir)


if __name__ == "__main__":
    main()
