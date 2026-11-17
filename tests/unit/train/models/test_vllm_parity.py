"""Forward parity of the trainer's models with vLLM's.

Question: for the same weights, does vLLM's model code give the same per-token logprobs as prime-rl's? RL samples with
vLLM and trains with prime-rl, so any difference shows up as trainer/inference mismatch KL and skews the training
signal, even when each side looks fine on its own. No reference implementation or weight file is involved.

How it works:

- No weight files. Each case writes only a tiny config.json (small widths, few layers and experts, every layer kind
  of the arch), with the architecture vLLM serves it as.
- Trainer side. The trainer builds its model from that config through its own path: `get_model`, the default
  fusions, and the fused LM head. It then draws seeded random weights (`randomize`). Matrices get std 0.02. Norm
  gains, biases and other vectors are drawn between 0.5 and 1.5, away from their usual 1 or 0, so a dropped or
  misnamed one shows in the logprobs. Router selection biases are drawn too. A case's `post_init` adjusts weights
  that need a structure or scale (DeepSeek-V4's hash routing table, for example).
- vLLM side. A vLLM engine starts inside the test process with `load_format="dummy"` (eager mode, small KV cache).
  The trainer's weights then go in through prime-rl's weight-update path: `convert_state_dict_to_hf` in their wire
  dtypes, as the weight broadcast sends them, then the inference worker's layerwise load. So a wrong weight
  conversion fails the test too. The config is parsed with the classes transformers ships, as a vLLM server does.
- Same tokens on both sides. There are 2 rows of real Reverse-Text-SFT conversations (201 and 175 tokens), longer
  than the sliding windows and long enough for DeepSeek-V4's heavily compressed attention. vLLM returns prompt
  logprobs; the trainer computes its logprobs for the same tokens in bf16, packed into one micro batch.
- Metrics: the largest |logprob difference| and the mean mismatch KL, which is the estimator the RL trainer logs as
  `mismatch_kl`.

MoE routers pick their top-k experts from near-tied scores, so bf16-level differences between the two forwards send a
few tokens to other experts. MoE cases therefore require that at least 85% of (token, MoE layer) pairs pick the same
experts (near-tie flips cost up to 6%). They then compare logprobs with the trainer replaying vLLM's routing, as RL's
router replay does, so the result does not depend on the GPU. A pure expert-selection bug still fails the agreement
check.

Tolerances: the runs are deterministic, so each arch has one clean value. Its `max_abs` and `mismatch_kl` are 2.4-3.5x
that value. Plain transformers allow 0.02 and 1e-5; clean qwen3 measures 0.008 and 3.5e-6. DeepSeek-V4, GLM-5.3,
Qwen3.8 Flash Next and gpt-oss were measured on Hopper, as their kernels need. The GLM-5 tolerances are provisional:
they were measured with substitute kernels.

What it catches, from bugs injected by hand:

- rope theta x2, attention scale x1.02, GELU instead of SiLU, a 1% final-norm scale (qwen3)
- swapped gate/up and permuted query heads in the weight conversion, a dropped k_norm
- router renormalization and sigmoid bugs, swapped experts, a selection-bias-only bug (through the agreement check)
- gated delta net `dt_bias` and conv bugs

It misses single ~1% norm scales and a 10x eps change, which sit at the bf16 noise floor.

Cases with `known_bug` run as strict xfails, so a fix shows up as a failure until the flag is removed:

- Laguna: vLLM 0.33 never loads its `e_score_correction_bias` on weight updates, so 75% of routes differ.
- MiniMax-M2: vLLM 0.33's layerwise reload crashes on its `functools.partial` QK-norm weight loaders.

Cases with `tiny=True` (qwen3, qwen3_moe, DeepSeek-V4, GLM-5) also run in the tiny GPU subset.

Adding an arch is one `Case`, with these fields:

- the tiny config, with `architectures` set to the class vLLM serves it as
- `max_abs` and `mismatch_kl`: at least 2x the clean values you measure
- `model` and `engine` overrides if needed
- `post_init` for weights whose random values need a structure or scale
- `tiny=True` to add it to the tiny subset
"""

import gc
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaConfig, PretrainedConfig, Qwen3Config
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from vllm import LLM, ModelRegistry, SamplingParams
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
from vllm.inputs import TokensPrompt

from prime_rl.configs.trainer import ModelConfig
from prime_rl.inference.vllm.worker.weight_transfer import get_weight_model, load_weights_checkpoint_layerwise
from prime_rl.trainer.model import forward, get_model, resolve_auto_attn
from prime_rl.trainer.models.afmoe import AfmoeConfig
from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.deepseek_v4.moe import DeepseekV4HashRouter
from prime_rl.trainer.models.fusions import apply_model_fusions
from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig
from prime_rl.trainer.models.glm5_next import Glm5NextTextConfig
from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaConfig
from prime_rl.trainer.models.gpt_oss import GptOssConfig
from prime_rl.trainer.models.laguna import LagunaConfig
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.layers.moe import TokenChoiceTopKRouter
from prime_rl.trainer.models.minimax_m2 import MiniMaxM2Config
from prime_rl.trainer.models.muse_glimmer import MuseGlimmerTextConfig
from prime_rl.trainer.models.nemotron_h import NemotronHConfig
from prime_rl.trainer.models.qwen3_5 import Qwen3_5MoeTextConfig, Qwen3_5TextConfig
from prime_rl.trainer.models.qwen3_8_flash_next import Qwen3_8FlashNextTextConfig
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig
from prime_rl.utils.weights import convert_state_dict_to_hf, resolve_wire_dtype

pytestmark = [pytest.mark.gpu]

TOKENIZER = "Qwen/Qwen3-0.6B"
NUM_ROWS = 2
MIN_ROUTE_AGREEMENT = 0.85
"""Share of (token, MoE layer) pairs whose top-k expert sets agree. Near-tie flips cost up to 6%."""

# Every model reads the same rows, tokenized with TOKENIZER
TINY_DIMS = dict(
    vocab_size=151936,
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    max_position_embeddings=4096,
)
# vLLM's gated delta net kernel on Hopper takes head_dim 128 only
LINEAR_ATTENTION_DIMS = dict(
    layer_types=["linear_attention", "full_attention"],
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
)

ENGINE_ARGS = dict(
    load_format="dummy",
    skip_tokenizer_init=True,
    enforce_eager=True,
    dtype="bfloat16",
    max_model_len=512,
    max_num_batched_tokens=1024,
    max_num_seqs=NUM_ROWS,
    kv_cache_memory_bytes=256 * 2**20,
    gpu_memory_utilization=0.05,
    enable_flashinfer_autotune=False,
)


@dataclass(frozen=True)
class Case:
    name: str
    config: PretrainedConfig
    """The tiny config, with the architecture vLLM serves it as."""
    max_abs: float
    """Largest per-token logprob difference allowed, at least 2x the largest measured."""
    mismatch_kl: float
    """Mean per-token mismatch KL allowed (the trainer's `mismatch_kl` metric), at least 2x the largest measured."""
    model: dict = field(default_factory=dict)
    """ModelConfig fields on top of the trainer defaults."""
    engine: dict = field(default_factory=dict)
    """vLLM engine arguments on top of ENGINE_ARGS."""
    post_init: Callable[[torch.nn.Module], None] | None = None
    """Runs after `randomize`, for weights and buffers whose random values need a structure or scale."""
    tiny: bool = False
    known_bug: str | None = None
    """Why the case fails today. It runs as a strict xfail, so a fix shows up as a failure until this is removed."""


@torch.no_grad()
def hash_routing_table(model: torch.nn.Module) -> None:
    """DeepSeek-V4's hash-routed layers read every token's experts from `tid2eid`: distinct experts per token."""
    for module in model.modules():
        if isinstance(module, DeepseekV4HashRouter):
            vocab_size, top_k = module.tid2eid.shape
            module.tid2eid.copy_(torch.rand(vocab_size, module.num_experts, device="cuda").argsort(dim=-1)[:, :top_k])


@torch.no_grad()
def scale_latent_projections(model: torch.nn.Module) -> None:
    """At std 0.02, Nemotron-H's latent MoE projections shrink the routed experts' output far below the shared
    expert's, hiding the routed experts. Draw them variance-preserving."""
    for name, param in model.named_parameters():
        if name.endswith(("fc1_latent_proj.weight", "fc2_latent_proj.weight")):
            param.normal_(0.0, param.shape[1] ** -0.5)


@torch.no_grad()
def weigh_sinks(model: torch.nn.Module) -> None:
    """At the generic init, gpt-oss' query and key biases swamp its attention sinks. Draw the biases like the weights
    and put the sinks near log(sliding_window), so they take a sizable share of the attention."""
    for name, param in model.named_parameters():
        if name.endswith(("qkv_proj.bias", "q_proj.bias", "k_proj.bias", "v_proj.bias")):
            param.normal_(0.0, 0.02)
        elif name.endswith("self_attn.sinks"):
            param.uniform_(2.0, 4.0)


def qwen3_8_flash_next_config() -> Qwen3_8FlashNextTextConfig:
    """Layers 0 and 1 are GatedDeltaNet with n-gram per-layer embeddings, before and after a pending hyper-connection
    combine, layer 2 indexed attention with its indexer. Every layer is MoE inside the hyper-connections. vLLM's
    kernels fix head_dim 256 with a 64-wide RoPE part, the indexer's head_dim 128 and its budget, and 24 query heads
    over 2 KV heads keep the trainer's indexed-attention planning small."""
    config = Qwen3_8FlashNextTextConfig(
        vocab_size=151936,
        hidden_size=256,
        num_hidden_layers=3,
        layer_types=["linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
        max_position_embeddings=4096,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        indexer_n_heads=4,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        hc_count=4,
        hc_lowrank=32,
        ple_layer_ids=[1, 2],
        ple_embed_dim=128,
        heads_per_ngram=2,
        ngram_vocab_size_base=1000,
        split_ngram_parts=3,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        num_experts=8,
        num_experts_per_tok=2,
        bos_token_id=None,
        # The n-gram hashes reset at EOS, which the rows contain
        eos_token_id=151645,
        architectures=["Qwen4ExpForCausalLM"],
    )
    # The trainer's config class drops output_gate_type, which vLLM's GatedDeltaNet reads; the trainer gates with sigmoid
    config.output_gate_type = "sigmoid"
    # It also keeps a top-level mrope_section, from which vLLM would serve the text model with M-RoPE positions
    del config.mrope_section
    return config


CASES = [
    Case(
        "qwen3",
        Qwen3Config(**TINY_DIMS, intermediate_size=512, tie_word_embeddings=True, architectures=["Qwen3ForCausalLM"]),
        max_abs=0.02,
        mismatch_kl=1e-5,
        tiny=True,
    ),
    Case(
        "qwen3_moe",
        Qwen3MoeConfig(
            **TINY_DIMS,
            moe_intermediate_size=128,
            num_experts=8,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            architectures=["Qwen3MoeForCausalLM"],
        ),
        max_abs=0.02,
        mismatch_kl=1e-5,
        tiny=True,
    ),
    Case(
        "llama",
        LlamaConfig(**TINY_DIMS, intermediate_size=512, tie_word_embeddings=False, architectures=["LlamaForCausalLM"]),
        max_abs=0.02,
        mismatch_kl=1e-5,
    ),
    Case(
        "glm4_moe",
        Glm4MoeConfig(
            **TINY_DIMS,
            intermediate_size=512,
            moe_intermediate_size=128,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            first_k_dense_replace=1,
            routed_scaling_factor=2.5,
            attention_bias=True,
            use_qk_norm=True,
            architectures=["Glm4MoeForCausalLM"],
        ),
        max_abs=0.02,
        mismatch_kl=1e-5,
    ),
    Case(
        "qwen3_5",
        Qwen3_5TextConfig(
            **TINY_DIMS, **LINEAR_ATTENTION_DIMS, intermediate_size=512, architectures=["Qwen3_5ForCausalLM"]
        ),
        max_abs=0.12,
        mismatch_kl=2e-4,
    ),
    Case(
        "qwen3_5_moe",
        Qwen3_5MoeTextConfig(
            **TINY_DIMS,
            **LINEAR_ATTENTION_DIMS,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
            num_experts=8,
            num_experts_per_tok=2,
            architectures=["Qwen3_5MoeForCausalLM"],
        ),
        max_abs=0.18,
        mismatch_kl=3e-4,
    ),
    Case(
        "afmoe",
        # Layer 0 is sliding-window attention with a dense MLP, layer 1 full attention (no RoPE) with MoE. vLLM reads
        # n_group and topk_group, which the Trinity checkpoints carry.
        AfmoeConfig(
            **TINY_DIMS,
            intermediate_size=512,
            moe_intermediate_size=128,
            num_dense_layers=1,
            num_experts=8,
            num_experts_per_tok=2,
            num_shared_experts=1,
            score_func="sigmoid",
            route_norm=True,
            route_scale=2.5,
            global_attn_every_n_layers=2,
            sliding_window=32,
            mup_enabled=True,
            n_group=1,
            topk_group=1,
            architectures=["AfmoeForCausalLM"],
        ),
        max_abs=0.03,
        mismatch_kl=1.5e-5,
    ),
    Case(
        "laguna",
        # Layer 0 is full attention (YaRN, partial RoPE) with a dense MLP, layer 1 sliding-window attention with more
        # heads and MoE
        LagunaConfig(
            **TINY_DIMS,
            intermediate_size=512,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
            num_experts=8,
            num_experts_per_tok=2,
            layer_types=["full_attention", "sliding_attention"],
            mlp_layer_types=["dense", "sparse"],
            num_attention_heads_per_layer=[4, 8],
            sliding_window=32,
            gating="per-head",
            moe_routed_scaling_factor=2.5,
            rope_parameters={
                "full_attention": {
                    "rope_type": "yarn",
                    "rope_theta": 500000.0,
                    "factor": 8.0,
                    "original_max_position_embeddings": 512,
                    "beta_fast": 64.0,
                    "beta_slow": 1.0,
                    "partial_rotary_factor": 0.5,
                },
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
            },
            architectures=["LagunaForCausalLM"],
        ),
        max_abs=0.03,
        mismatch_kl=2e-5,
        known_bug="vLLM 0.33 never loads Laguna's mlp.experts.e_score_correction_bias, so it routes without it",
    ),
    Case(
        "minimax_m2",
        MiniMaxM2Config(
            **TINY_DIMS,
            intermediate_size=128,
            num_local_experts=8,
            num_experts_per_tok=2,
            rotary_dim=32,
            scoring_func="sigmoid",
            use_routing_bias=True,
            use_qk_norm=True,
            qk_norm_type="per_layer",
            architectures=["MiniMaxM2ForCausalLM"],
        ),
        max_abs=0.03,
        mismatch_kl=2e-5,
        known_bug="vLLM 0.33's layerwise reload reads __name__ of MiniMax-M2's functools.partial QK-norm weight loaders",
    ),
    Case(
        "deepseek_v4",
        # One layer of each attention type: sliding, compressed sparse (with the indexer) and heavily compressed. Layer 0
        # is hash-routed. Kernels fix head_dim 512 with a 64-wide RoPE part, 64 heads, and the compress rates 4 and
        # 128. index_topk is below the compressed entries a row has, so the indexer selects.
        DeepseekV4Config(
            vocab_size=151936,
            hidden_size=256,
            moe_intermediate_size=128,
            num_hidden_layers=3,
            num_attention_heads=64,
            num_key_value_heads=1,
            head_dim=512,
            partial_rotary_factor=64 / 512,
            q_lora_rank=128,
            o_groups=8,
            o_lora_rank=128,
            max_position_embeddings=512,
            sliding_window=64,
            layer_types=["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"],
            compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
            index_n_heads=64,
            index_head_dim=128,
            index_topk=16,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            num_hash_layers=1,
            architectures=["DeepseekV4ForCausalLM"],
        ),
        max_abs=0.06,
        mismatch_kl=7e-5,
        post_init=hash_routing_table,
        tiny=True,
    ),
    Case(
        "glm_moe_dsa",
        # GLM-5: sparse MLA with the lightning indexer, a dense layer, then MoE layers; layer 2 reuses layer 1's
        # selection (IndexShare). Kernels fix kv_lora_rank 512 with a 64-wide RoPE part and 64 heads, and vLLM's sparse
        # MLA a multiple of 128 for index_topk. It is below the row lengths, so the indexer selects for later tokens.
        GlmMoeDsaConfig(
            vocab_size=151936,
            hidden_size=256,
            intermediate_size=512,
            moe_intermediate_size=128,
            num_hidden_layers=3,
            first_k_dense_replace=1,
            num_attention_heads=64,
            num_key_value_heads=64,
            q_lora_rank=256,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            qk_nope_head_dim=64,
            v_head_dim=64,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            routed_scaling_factor=2.5,
            n_group=1,
            topk_group=1,
            index_n_heads=32,
            index_head_dim=128,
            index_topk=128,
            indexer_types=["full", "full", "shared"],
            index_topk_freq=2,
            index_skip_topk_offset=2,
            max_position_embeddings=4096,
            rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
            pad_token_id=151643,
            architectures=["GlmMoeDsaForCausalLM"],
        ),
        max_abs=0.1,
        mismatch_kl=1.5e-4,
        tiny=True,
    ),
    Case(
        "glm5_next",
        # GLM-5.3 Flash: layer 0 is Kimi Delta Attention with a dense MLP, layer 1 sparse MLA with the k-pool indexer and
        # MoE, both inside hyper-connections. Kernels fix the KDA head_dim 128, 64 MLA heads with kv_lora_rank 512 and
        # no RoPE, and the indexer's 32 heads of 128. swiglu_limit is low enough for the clamp to act. mhc, mla_use_nope,
        # topk_method, moe_router_dtype and indexer_rope_interleave come from the real config.json, for vLLM.
        Glm5NextTextConfig(
            vocab_size=151936,
            hidden_size=256,
            intermediate_size=512,
            moe_intermediate_size=128,
            num_hidden_layers=2,
            layer_types=["linear_attention", "deepseek_sparse_attention"],
            mlp_layer_types=["dense", "sparse"],
            linear_attn_config={"num_heads": 2, "head_dim": 128, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0},
            num_attention_heads=64,
            num_key_value_heads=64,
            q_lora_rank=128,
            kv_lora_rank=512,
            qk_nope_head_dim=256,
            qk_rope_head_dim=0,
            v_head_dim=256,
            index_n_heads=32,
            index_head_dim=128,
            index_topk=2048,
            index_kpool=4,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            routed_scaling_factor=2.5,
            swiglu_limit=0.5,
            hc_mult=4,
            max_position_embeddings=4096,
            num_nextn_predict_layers=0,
            pad_token_id=None,
            mhc=True,
            mla_use_nope=True,
            topk_method="noaux_tc",
            moe_router_dtype="float32",
            indexer_rope_interleave=True,
            architectures=["Glm5NextForCausalLM"],
        ),
        max_abs=0.03,
        mismatch_kl=1e-5,
        # Prompts shorter than index_topk otherwise take a dense prefill on Hopper, away from the sparse MLA kernel
        engine=dict(attention_config={"sparse_mla_force_mqa": True}),
    ),
    Case(
        "qwen3_8_flash_next",
        qwen3_8_flash_next_config(),
        max_abs=0.04,
        mismatch_kl=3e-5,
    ),
    Case(
        "nemotron_h",
        # Mamba-2, latent MoE with a shared expert, and attention layers
        NemotronHConfig(
            vocab_size=151936,
            hidden_size=256,
            hybrid_override_pattern="ME*E",
            max_position_embeddings=4096,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            mamba_num_heads=8,
            mamba_head_dim=64,
            ssm_state_size=64,
            n_groups=2,
            conv_kernel=4,
            chunk_size=64,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            moe_intermediate_size=128,
            moe_shared_expert_intermediate_size=256,
            moe_latent_size=128,
            routed_scaling_factor=2.5,
            architectures=["NemotronHForCausalLM"],
        ),
        max_abs=0.02,
        mismatch_kl=1e-5,
        post_init=scale_latent_projections,
    ),
    Case(
        "muse_glimmer",
        # Layer 0 is sliding-window attention with RoPE, layer 1 full attention without
        MuseGlimmerTextConfig(
            **TINY_DIMS,
            intermediate_size=512,
            sliding_window=32,
            rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
            layer_rope_theta=[500000.0, 0],
            architectures=["MuseGlimmerForCausalLM"],
        ),
        max_abs=0.02,
        mismatch_kl=1e-5,
    ),
    Case(
        "gpt_oss",
        # Layer 0 is sliding-window attention, layer 1 full attention, both with sinks
        GptOssConfig(
            **TINY_DIMS,
            intermediate_size=128,
            num_local_experts=8,
            num_experts_per_tok=2,
            sliding_window=32,
            architectures=["GptOssForCausalLM"],
        ),
        max_abs=0.02,
        mismatch_kl=1e-5,
        post_init=weigh_sinks,
    ),
]


@torch.no_grad()
def randomize(model: torch.nn.Module) -> None:
    """Seeded random weights. Norm gains, biases and other vectors are drawn away from their usual 1 or 0, so that a
    dropped or misnamed one shows in the logprobs."""
    torch.manual_seed(0)
    for param in model.parameters():
        if param.ndim < 2:
            param.uniform_(0.5, 1.5)
        else:
            param.normal_(0.0, 0.02)
    for name, buffer in model.named_buffers():
        # The routers' selection-only bias is a buffer, which the weight broadcast sends too
        if name.endswith("router.selection_bias"):
            buffer.normal_(0.0, 0.05)


def build_trainer_model(model_dir: Path, case: Case) -> torch.nn.Module:
    """The trainer's model, unsharded on one GPU, with fp32 master weights."""
    config = ModelConfig(name=model_dir.as_posix(), compile=None, ac=None, **case.model)
    resolve_auto_attn(config)
    model = get_model(config, device=torch.device("meta"), dtype=torch.float32)
    apply_model_fusions(model, config.fusions.enabled)
    inject_prime_lm_head(model, chunk_size=config.fused_lm_head_token_chunk_size)
    model.to_empty(device="cuda")
    model.tie_weights()
    model.init_buffers_post_meta()
    randomize(model)
    if case.post_init is not None:
        case.post_init(model)
    return model


def broadcast_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The weights the trainer broadcasts: parameters in their wire dtype, buffers as they are, converted to the
    checkpoint format vLLM loads."""
    keep_in_fp32 = model.keep_in_fp32_for_weight_transfer
    params = {
        name: param.detach().to(resolve_wire_dtype(keep_in_fp32, name, torch.bfloat16))
        for name, param in model.named_parameters(remove_duplicate=False)
    }
    state_dict = {name: params.get(name, tensor) for name, tensor in model.state_dict().items()}
    return convert_state_dict_to_hf(model, state_dict)


@torch.no_grad()
def trainer_logprobs(
    model: torch.nn.Module, rows: list[list[int]], routed_experts: torch.Tensor | None = None
) -> list[torch.Tensor]:
    """The trainer's per-token logprobs of the rows, packed into one micro batch."""
    seq_lens = [len(row) for row in rows]
    input_ids = torch.tensor([sum(rows, [])], device="cuda")
    position_ids = torch.cat([torch.arange(seq_len) for seq_len in seq_lens]).unsqueeze(0).cuda()
    labels = torch.cat([torch.tensor(row[1:] + [0]) for row in rows]).unsqueeze(0).cuda()
    out = forward(
        model,
        input_ids,
        position_ids,
        seq_lens=torch.tensor(seq_lens),
        labels=labels,
        temperature=torch.ones_like(input_ids, dtype=torch.float32),
        routed_experts=routed_experts,
    )
    # The last position of each row predicts the next row's first token
    return [row_logprobs[:-1] for row_logprobs in out["logprobs"][0].cpu().split(seq_lens)]


def moe_routers(model: torch.nn.Module) -> dict[int, TokenChoiceTopKRouter]:
    return {
        int(re.search(r"layers\.(\d+)\.", name).group(1)): module
        for name, module in model.named_modules()
        if isinstance(module, TokenChoiceTopKRouter)
    }


def trainer_routing(model: torch.nn.Module, rows: list[list[int]]) -> dict[int, torch.Tensor]:
    """The experts the trainer's own routers pick for every token, by layer."""
    routes = {}

    def record(layer: int):
        def hook(module, args, output):
            routes[layer] = output[1].cpu()

        return hook

    hooks = [router.register_forward_hook(record(layer)) for layer, router in moe_routers(model).items()]
    trainer_logprobs(model, rows)
    for hook in hooks:
        hook.remove()
    return routes


def route_agreement(routes: dict[int, torch.Tensor], vllm_routes: torch.Tensor) -> float:
    """Share of (token, MoE layer) pairs for which both forwards pick the same top-k experts."""
    same = [
        (experts.sort(dim=-1).values == vllm_routes[:, layer].sort(dim=-1).values).all(dim=-1)
        for layer, experts in routes.items()
    ]
    return torch.cat(same).float().mean().item()


@contextmanager
def stock_configs() -> Iterator[None]:
    """Parse configs with the classes transformers ships, as a vLLM server does. Importing the trainer's models
    registers its own config classes over them, and those lack attributes that vLLM reads."""
    trainer_configs = {
        model_type: config_class
        for model_type, config_class in CONFIG_MAPPING._extra_content.items()
        if config_class.__module__.startswith("prime_rl.")
    }
    for model_type in trainer_configs:
        del CONFIG_MAPPING._extra_content[model_type]
    try:
        yield
    finally:
        CONFIG_MAPPING._extra_content.update(trainer_configs)


def load_weights(worker, state_dict: dict[str, torch.Tensor]) -> None:
    """The inference worker's weight update, fed from memory instead of a broadcast."""
    model = get_weight_model(worker.model_runner, "model")
    load_weights_checkpoint_layerwise(
        model, iter(state_dict.items()), worker.model_runner.model_config, worker.vllm_config
    )


def vllm_logprobs(
    model_dir: Path, case: Case, state_dict: dict[str, torch.Tensor], rows: list[list[int]], moe: bool
) -> tuple[list[torch.Tensor], torch.Tensor | None]:
    """vLLM's prompt logprobs of the rows from the trainer's weights, and for MoE models the experts it routed every
    token to, as [tokens, layers, top_k]."""
    architecture = case.config.architectures[0]
    # Registering the class in this process skips vLLM's inspection of the architecture in a subprocess (about 10 s
    # without a warm vLLM cache)
    ModelRegistry.register_model(architecture, ModelRegistry.models[architecture].load_model_cls())
    # vLLM runs a dozen full garbage collections while it starts, each about 0.6 s over this process's heap. Frozen
    # objects are skipped; cleanup_dist_env_and_memory unfreezes them.
    gc.freeze()
    with stock_configs():
        llm = LLM(model=model_dir.as_posix(), **{**ENGINE_ARGS, "enable_return_routed_experts": moe, **case.engine})
    try:
        llm.collective_rpc(load_weights, args=(state_dict,))
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=row) for row in rows],
            SamplingParams(max_tokens=1, prompt_logprobs=0, detokenize=False),
            use_tqdm=False,
        )
    finally:
        llm.llm_engine.engine_core.shutdown()
        cleanup_dist_env_and_memory()
    # The first token has no logprob
    logprobs = [
        torch.tensor([output.prompt_logprobs[i][token].logprob for i, token in enumerate(row) if i > 0])
        for row, output in zip(rows, outputs)
    ]
    if not moe:
        return logprobs, None
    routes = np.concatenate([output.outputs[0].routed_experts for output in outputs])
    return logprobs, torch.from_numpy(routes).long()


@pytest.fixture(scope="module")
def rows() -> list[list[int]]:
    """Two pairs of Reverse-Text-SFT conversations, 201 and 175 tokens: longer than the sliding windows, and long
    enough for DeepSeek-V4's heavily compressed attention (128 tokens per entry)."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    dataset = load_dataset("PrimeIntellect/Reverse-Text-SFT", split=f"train[:{2 * NUM_ROWS}]")
    conversations = [
        tokenizer.apply_chat_template(row["prompt"] + row["completion"], return_dict=False) for row in dataset
    ]
    return [conversations[i] + conversations[i + 1] for i in range(0, len(conversations), 2)]


def case_param(case: Case):
    marks = [pytest.mark.tiny] if case.tiny else []
    if case.known_bug is not None:
        marks.append(pytest.mark.xfail(reason=case.known_bug, strict=True))
    return pytest.param(case, id=case.name, marks=marks)


@pytest.mark.parametrize("case", [case_param(case) for case in CASES])
def test_matches_vllm(case: Case, rows: list[list[int]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Keep the engine in this process, so the trainer's weights reach it through a function call
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_NO_USAGE_STATS", "1")
    case.config.save_pretrained(tmp_path)
    model = build_trainer_model(tmp_path, case)
    state_dict = broadcast_state_dict(model)
    # FSDP computes in bf16 from the fp32 master weights
    for param in model.parameters():
        param.data = param.data.bfloat16()
    moe = bool(moe_routers(model))
    vllm, vllm_routes = vllm_logprobs(tmp_path, case, state_dict, rows, moe)

    if moe:
        agreement = route_agreement(trainer_routing(model, rows), vllm_routes)
        assert agreement >= MIN_ROUTE_AGREEMENT, f"top-k experts agree on {agreement:.0%} of tokens"
    trainer = trainer_logprobs(model, rows, vllm_routes.unsqueeze(0).cuda() if moe else None)

    log_ratio = torch.cat(trainer) - torch.cat(vllm)
    max_abs = log_ratio.abs().max().item()
    mismatch_kl = (torch.expm1(log_ratio) - log_ratio).mean().item()
    assert max_abs <= case.max_abs, f"max |logprob diff| {max_abs:.2e} > {case.max_abs:.0e}"
    assert mismatch_kl <= case.mismatch_kl, f"mismatch KL {mismatch_kl:.2e} > {case.mismatch_kl:.0e}"
