"""Create and verify a mini MoE model for testing.

Creates a small MoE model with random weights, saves it with a tokenizer, and verifies the HF <->
PrimeRL weight conversion roundtrip. Compares KL divergences between prime-rl and HF
implementations, and top-1 agreement (a noisy metric for random-init models). Also checks that
packing a document behind another one leaves its logprobs unchanged, and with --check-vllm-kl
scores the model with vLLM and checks the mismatch KL the RL trainer would see.

How this mirrors production:
  1. bf16 weights, fp32 buffers (inv_freq, expert_bias, ...): the prod forward under FSDP mixed precision
  2. fp32 MoE router gate, per the moe_router_dtype="float32" default
  3. Grouped-mm experts, per the use_grouped_mm=True default (needs a recent GPU arch)
  4. flash_attention_2 on both models for the HF comparison, so a logits gap means the port, not two
     kernels; the packing and vLLM checks have no HF side and use the kernel the trainer resolves to
  5. Prime LM head injected, as setup_model does
  6. seq_lens passed as one document for the HF comparison; the packing check passes two

NOTE: should be taken as a very coarse sanity check against catastrophic incorrectness. Thresholds
are loose and may pass even with moderate correctness bugs. Not a replacement for robust unit
testing. The packing check is the exception: its correct value is exactly zero, so its bar sits far
below any real boundary bug.

Usage:
    # Create and verify
    uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./mini-glm-moe

    # Verify only (on an existing checkpoint)
    uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./mini-glm-moe --verify-only

    # Also compare against vLLM
    uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./mini-glm-moe \
        --verify-only --check-vllm-kl
"""

import argparse
import gc
import shutil
import tempfile
from dataclasses import dataclass
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

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import apply_fp32_moe_router, resolve_auto_attn
from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig
from prime_rl.trainer.models.glm4_moe import Glm4MoeForCausalLM as PrimeRLGlm4MoeForCausalLM
from prime_rl.trainer.models.laguna import LagunaConfig
from prime_rl.trainer.models.laguna import LagunaForCausalLM as PrimeRLLagunaForCausalLM
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.minimax_m2 import MiniMaxM2Config
from prime_rl.trainer.models.minimax_m2 import MiniMaxM2ForCausalLM as PrimeRLMiniMaxM2ForCausalLM
from prime_rl.trainer.models.qwen3_5_moe import Qwen3_5MoeForCausalLM as PrimeRLQwen3_5MoeVLM
from prime_rl.trainer.rl.loss import (
    compute_importance_ratio_and_mismatch_kl,
    selective_log_softmax,
    shift_tensor_left,
    shift_tensor_right,
)
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.utils import default_dtype
from prime_rl.utils.weights import save_state_dict

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


DOC_LEN = 512
FILLER_LEN = 128
VLLM_MAX_MODEL_LEN = 2048
VLLM_GPU_MEMORY_UTILIZATION = 0.6

# Observed across seeds 0-3 on the production-mirroring forward. Packing spans all three text
# presets; the vLLM rows span glm4_moe and minimax_m2 only, since vLLM cannot load the laguna
# checkpoint (its hub code writes `mlp.shared_experts`, vLLM expects `mlp.shared_expert`).
#   packed vs unpacked, max logprob diff   0.020 to 0.022 minimax, 0.031 to 0.043 glm4_moe,
#                                          0.097 to 0.127 laguna
#   mismatch KL vs vLLM, mean              1.1e-5 to 3.6e-5 minimax/glm4_moe, 1.2e-4 to 1.9e-4 laguna
#   mismatch KL vs vLLM, max               2.8e-4 to 8.3e-4 minimax/glm4_moe, 5.4e-3 to 1.2e-2 laguna
#   per-token |packed - unpacked| KL       1.0e-5 to 4.0e-5 mean, 2.1e-4 to 5.7e-4 max
#                                          minimax/glm4_moe; laguna 1.6e-4 mean, 4.6e-3 to 1.3e-2 max
# Laguna diverges from vLLM about 20x more than the others on the max, and is likewise the outlier
# in the HF comparison, so its delta rows inherit that rather than showing a boundary problem: its
# packed and unpacked means track each other, and the engine-free boundary gate sits at 0.12.
# For scale, dropping the document boundary measures 3.14 on the packing row and 0.44 mean / 15.4
# max on the packed KL rows.
PACKED_LOGPROB_DIFF_THRESHOLD = 0.5
# 0.015 is the KL-mismatch merge bar from docs/development.md. That number is measured on a trained
# model over 20 steps of math, not on random weights, so it is borrowed as a sanity bound rather
# than an equivalent measurement: a model that cannot clear it here certainly cannot clear it there.
KL_MEAN_THRESHOLD = 0.015
KL_MAX_THRESHOLD = 0.1
KL_DELTA_MEAN_THRESHOLD = 0.001
KL_DELTA_MAX_THRESHOLD = 0.05


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
            # Matches what HF and prime-rl derive anyway, but vLLM reads config.head_dim directly.
            head_dim=64,
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


def _load_config(model_dir: Path, trust_remote_code: bool, attn_implementation: str):
    """Load a saved config, pinning the attention implementation on the text config too."""
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
    config._attn_implementation = attn_implementation
    if hasattr(config, "text_config"):
        config.text_config._attn_implementation = attn_implementation
    return config


def _build_config(preset):
    """Build model config from preset (handles both config_class and config_fn styles)."""
    if "config_fn" in preset:
        return preset["config_fn"]()
    return preset["config_class"](**preset["config_kwargs"])


def create(arch: str, output_dir: Path, seed: int) -> None:
    torch.manual_seed(seed)
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


@dataclass
class Check:
    """One verification result and the bar its value must not exceed."""

    name: str
    value: float
    threshold: float
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.value <= self.threshold

    @property
    def ratio(self) -> float | None:
        """value / threshold, the pass margin. None when the bar is exactly zero."""
        return None if self.threshold == 0 else self.value / self.threshold


def report(checks: list[Check]) -> None:
    width = max(len(c.name) for c in checks)
    print()
    print(f"  {'check':<{width}}  {'value':>12}  {'threshold':>12}  {'ratio':>12}  status")
    for c in checks:
        detail = f"  {c.detail}" if c.detail else ""
        ratio = "N/A" if c.ratio is None else f"{c.ratio:.6f}"
        print(
            f"  {c.name:<{width}}  {c.value:>12.6f}  {c.threshold:>12.6f}  {ratio:>12}  "
            f"{'ok' if c.ok else 'FAIL'}{detail}"
        )
    print()


def _resolve_attn(attn: str) -> str:
    """Resolve attn='auto' the same way the trainer does, by GPU architecture."""
    # resolve_auto_attn only resolves; setup_model separately validates that FA3/FA4 are installed,
    # so an unusable choice surfaces here as the underlying import error.
    config = ModelConfig(attn=attn)
    resolve_auto_attn(config)
    return config.attn


def verify_hf(arch: str, model_dir: Path, seed: int) -> list[Check]:
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
    # Seeded here, not at function entry, so the tokens do not depend on how much RNG the model
    # construction above happened to consume.
    torch.manual_seed(seed)
    input_ids = torch.randint(0, max_token, (1, 64), device="cuda")
    position_ids = torch.arange(1, 65, device="cuda").unsqueeze(0)

    seq_lens = torch.tensor([input_ids.shape[1]], device=input_ids.device)
    hf_output = hf_model(input_ids=input_ids, position_ids=position_ids)
    prime_output = prime_model(input_ids, position_ids, seq_lens=seq_lens)

    checks: list[Check] = []
    if is_vlm:
        num_nans = int(torch.isnan(prime_output["logits"]).sum().item())
        shape_matches = prime_output["logits"].shape == hf_output.logits.shape
        checks.append(Check("vlm_logits_nans", num_nans, 0))
        checks.append(
            Check(
                "vlm_logits_shape_mismatch",
                0 if shape_matches else 1,
                0,
                detail=""
                if shape_matches
                else f"{tuple(prime_output['logits'].shape)} vs {tuple(hf_output.logits.shape)}",
            )
        )

    max_diff = (prime_output["logits"] - hf_output.logits).abs().max().item()
    mean_kl, max_kl, top1 = _compare_distributions(hf_output.logits, prime_output["logits"])
    checks.append(Check("hf_vs_prime_kl_mean", mean_kl, 1e-3))
    checks.append(Check("hf_vs_prime_kl_max", max_kl, 1e-2))
    # Check.ok compares upward, so record the complement of the top-1 lower bound.
    checks.append(Check("hf_vs_prime_top1_disagreement", 1 - top1, 0.25))
    checks.append(Check("hf_vs_prime_max_logits_diff", max_diff, 1.0))

    # Roundtrip weight conversion: HF -> PrimeRL -> HF
    # Normalize both through the same roundtrip to handle expert format differences
    with torch.no_grad():
        roundtrip_sd = prime_model.convert_to_hf(dict(prime_model.state_dict()))
        orig_sd = dict(hf_model.state_dict())
        prime_model.convert_to_prime(orig_sd)
        prime_model.convert_to_hf(orig_sd)

    bad_keys = [key for key in orig_sd if key not in roundtrip_sd or not torch.equal(orig_sd[key], roundtrip_sd[key])]
    detail = ", ".join(bad_keys[:3]) + (", ..." if len(bad_keys) > 3 else "")
    checks.append(Check("weight_roundtrip_mismatches", len(bad_keys), threshold=0, detail=detail))

    return checks


def _trainer_logprobs(
    prime_model, input_ids: torch.Tensor, position_ids: torch.Tensor, seq_lens: torch.Tensor, doc_start: int
) -> torch.Tensor:
    """Per-token logprobs of the scored document, which is always the suffix of the packed row.

    Returns probability-of-current-token logprobs with position 0 dropped, so the returned tokens
    are exactly those vLLM reports a prompt logprob for.
    """
    with torch.no_grad():
        logits = prime_model(input_ids, position_ids, seq_lens=seq_lens)["logits"][:, doc_start:].float()
    doc_ids = input_ids[:, doc_start:]
    logprobs = selective_log_softmax(logits, shift_tensor_left(doc_ids))
    return shift_tensor_right(logprobs)[:, 1:]


def _export_for_vllm(prime_model, model_dir: Path, out_dir: Path) -> None:
    """Write a vLLM-loadable copy of the checkpoint using prime-rl's own HF conversion.

    Poolside's hub code names laguna's shared expert `mlp.shared_experts` and hangs the router bias
    off `mlp.gate`; neither matches their released weights or what vLLM looks for. `convert_to_hf`
    emits the released spellings, and it is what both weight transports send during an RL run, so
    the engine ends up loading what production would. A key the conversion chain drops will surface
    here as a vLLM loading error rather than as a conversion bug.
    """
    with torch.no_grad():
        state_dict = prime_model.convert_to_hf(dict(prime_model.state_dict()))
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in model_dir.iterdir():  # config.json, tokenizer, any trust_remote_code modules
        if path.is_file() and ".safetensors" not in path.name:
            shutil.copyfile(path, out_dir / path.name)
    save_state_dict({key: value.detach().cpu() for key, value in state_dict.items()}, out_dir)


def _score_packed_and_unpacked(
    arch: str, model_dir: Path, seed: int, attn: str, export_dir: Path | None = None
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Score one document alone, then again packed behind a filler document.

    Returns both logprob rows and the document's token ids. The scored document is the suffix of
    the row in both cases, so one scoring path serves both. Frees the model before returning:
    vLLM profiles its KV cache against the whole free GPU, so nothing large may still be resident
    when the caller builds an engine.
    """
    preset = ARCH_PRESETS[arch]
    trust_remote_code = preset["hf_model_class"] is None
    config = _load_config(model_dir, trust_remote_code, attn)
    config.dtype = torch.bfloat16

    hf_model = _load_hf_model(preset, model_dir, config)
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        prime_model = preset["prime_model_class"]._from_config(config)

    with torch.no_grad():
        state_dict = hf_model.state_dict()
        prime_model.convert_to_prime(state_dict)
        prime_model.load_state_dict(state_dict)
    del hf_model, state_dict

    inject_prime_lm_head(prime_model, chunk_size=None)
    apply_fp32_moe_router(prime_model)

    torch.manual_seed(seed)
    doc = torch.randint(0, config.vocab_size, (1, DOC_LEN), device="cuda")
    filler = torch.randint(0, config.vocab_size, (1, FILLER_LEN), device="cuda")

    unpacked_logprobs = _trainer_logprobs(
        prime_model,
        doc,
        torch.arange(DOC_LEN, device="cuda").unsqueeze(0),
        torch.tensor([DOC_LEN], dtype=torch.long, device="cuda"),
        doc_start=0,
    )
    packed_logprobs = _trainer_logprobs(
        prime_model,
        torch.cat([filler, doc], dim=1),
        torch.cat([torch.arange(FILLER_LEN, device="cuda"), torch.arange(DOC_LEN, device="cuda")]).unsqueeze(0),
        torch.tensor([FILLER_LEN, DOC_LEN], dtype=torch.long, device="cuda"),
        doc_start=FILLER_LEN,
    )

    if export_dir is not None:
        _export_for_vllm(prime_model, model_dir, export_dir)

    del prime_model
    gc.collect()
    torch.cuda.empty_cache()

    return unpacked_logprobs, packed_logprobs, doc.squeeze(0).tolist()


def verify_packed(arch: str, model_dir: Path, seed: int, attn: str) -> list[Check]:
    """Check that packing a document behind another one does not change its logprobs.

    Honouring the document boundaries makes the two rows identical up to kernel nondeterminism, so
    this is the sharpest boundary test available, and it needs no inference engine.
    """
    print(f"Verifying packed vs unpacked logprobs for {model_dir}...")
    unpacked_logprobs, packed_logprobs, _ = _score_packed_and_unpacked(arch, model_dir, seed, attn)
    delta = (packed_logprobs - unpacked_logprobs).abs()
    return [
        Check(
            "packed_vs_unpacked_max_logprob_diff",
            delta.max().item(),
            PACKED_LOGPROB_DIFF_THRESHOLD,
        )
    ]


def verify_vllm(arch: str, model_dir: Path, seed: int, attn: str, moe_backend: str | None = None) -> list[Check]:
    """Compare PrimeRL logprobs against vLLM's, unpacked and packed behind a filler document.

    Text-only presets; see the guard in main().

    vLLM serves one document per request, so the engine side is identical in both cases and the
    packed-minus-unpacked delta isolates document-boundary handling from whatever constant
    divergence the two implementations carry.
    """
    print(f"Comparing PrimeRL vs vLLM mismatch KL for {model_dir}...")
    trust_remote_code = ARCH_PRESETS[arch]["hf_model_class"] is None
    export_dir = tempfile.TemporaryDirectory()
    unpacked_logprobs, packed_logprobs, doc_ids = _score_packed_and_unpacked(
        arch, model_dir, seed, attn, export_dir=Path(export_dir.name)
    )

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=export_dir.name,
        enforce_eager=True,
        dtype="bfloat16",
        max_model_len=VLLM_MAX_MODEL_LEN,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        trust_remote_code=trust_remote_code,
        **({"kernel_config": {"moe_backend": moe_backend}} if moe_backend else {}),
    )
    outputs = llm.generate(
        TokensPrompt(prompt_token_ids=doc_ids),
        SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0),
    )
    # prompt_logprobs[i] is the logprob of token i given everything before it; entry 0 is None.
    prompt_logprobs = outputs[0].prompt_logprobs
    inference_logprobs = torch.tensor(
        [logprobs[token_id].logprob for logprobs, token_id in zip(prompt_logprobs[1:], doc_ids[1:])],
        device="cuda",
    ).unsqueeze(0)

    export_dir.cleanup()

    _, _, kl_unpacked = compute_importance_ratio_and_mismatch_kl(unpacked_logprobs, inference_logprobs)
    _, _, kl_packed = compute_importance_ratio_and_mismatch_kl(packed_logprobs, inference_logprobs)
    # Per-token, so a packed and an unpacked outlier on different tokens cannot cancel.
    delta = (kl_packed - kl_unpacked).abs()

    return [
        Check("kl_unpacked_mean", kl_unpacked.mean().item(), KL_MEAN_THRESHOLD),
        Check("kl_unpacked_max", kl_unpacked.max().item(), KL_MAX_THRESHOLD),
        Check("kl_packed_mean", kl_packed.mean().item(), KL_MEAN_THRESHOLD),
        Check("kl_packed_max", kl_packed.max().item(), KL_MAX_THRESHOLD),
        Check("kl_packed_minus_unpacked_mean", delta.mean().item(), KL_DELTA_MEAN_THRESHOLD),
        Check("kl_packed_minus_unpacked_max", delta.max().item(), KL_DELTA_MAX_THRESHOLD),
    ]


def main():
    parser = argparse.ArgumentParser(description="Create and verify a mini MoE model")
    parser.add_argument("--arch", choices=list(ARCH_PRESETS.keys()), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true", help="Skip creation, only verify an existing model")
    parser.add_argument(
        "--check-vllm-kl",
        action="store_true",
        help="Also score the model with vLLM and check the packed-vs-unpacked mismatch KL (starts an engine)",
    )
    parser.add_argument(
        "--vllm-moe-backend",
        default=None,
        help="Override vLLM's MoE kernel backend, e.g. 'triton' on GPUs whose auto-selected FlashInfer kernels do not build",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seeds the random weights and the random token ids")
    parser.add_argument(
        "--attn",
        default="auto",
        help="PrimeRL attention implementation; 'auto' resolves by GPU architecture as the trainer does",
    )
    args = parser.parse_args()

    if args.arch == "minimax_m2":
        _patch_minimax_m2_hub_code()

    is_vlm = ARCH_PRESETS[args.arch].get("is_vlm", False)
    if args.check_vllm_kl and is_vlm:
        parser.error(
            f"--check-vllm-kl does not support the VLM preset {args.arch!r}: driving prompt "
            "logprobs for a multimodal model through vLLM is a separate path."
        )

    if not args.verify_only:
        create(args.arch, args.output_dir, args.seed)

    attn = _resolve_attn(args.attn)
    checks = verify_hf(args.arch, args.output_dir, args.seed)
    if not is_vlm:
        # Packing a VLM row needs MRoPE 3D positions and modality-aware token ids, which this
        # script does not build.
        checks += verify_packed(args.arch, args.output_dir, args.seed, attn)
    if args.check_vllm_kl:
        checks += verify_vllm(args.arch, args.output_dir, args.seed, attn, args.vllm_moe_backend)

    report(checks)
    failed = [c for c in checks if not c.ok]
    if failed:
        raise SystemExit(f"{len(failed)} of {len(checks)} checks failed: " + ", ".join(c.name for c in failed))
    print(f"All {len(checks)} checks passed.")


if __name__ == "__main__":
    main()
