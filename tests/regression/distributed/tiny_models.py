"""Seeded tiny HF checkpoints of trainer archs, built from explicit config kwargs.

Each config is the arch's published HF config with `num_hidden_layers` cut to the shortest prefix that covers every
layer kind, and its per-layer lists cut to match. Quantization and MTP fields are left out because the checkpoints are
bf16 and train without MTP.

Notes:

- MoE archs use a smaller hidden size, smaller intermediate sizes and fewer experts than published. Layer kinds,
  vocab, head counts and head dims, and the attention and indexer shapes stay published. At published widths the MoE
  checkpoints reach 26 GB, take up to 77 GB of host memory to build and double the launch time of the largest archs.
- qwen4_exp `ngram_vocab_size_base` is 200000, not 20000000: the real per-layer n-gram table alone is about 51B params.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from transformers import LlamaConfig, PretrainedConfig, Qwen3Config

from prime_rl.trainer.models import get_custom_causal_lm_cls
from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.deepseek_v4.moe import DeepseekV4HashRouter
from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig
from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaConfig
from prime_rl.trainer.models.gpt_oss import GptOssConfig
from prime_rl.trainer.models.laguna import LagunaConfig
from prime_rl.trainer.models.layers.moe import MoE
from prime_rl.trainer.models.minimax_m2 import MiniMaxM2Config
from prime_rl.trainer.models.nemotron_h import NemotronHConfig
from prime_rl.trainer.models.qwen3_5 import Qwen3_5MoeTextConfig, Qwen3_5TextConfig
from prime_rl.trainer.models.qwen3_8_flash_next import Qwen3_8FlashNextTextConfig
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig
from prime_rl.utils.utils import default_dtype
from prime_rl.utils.weights import load_state_dict, save_state_dict

SEED = 0


@dataclass(frozen=True, eq=False)
class TinyModel:
    """A tiny HF config, the tokenizer matching its vocab, and its renderer when the tokenizer can't select one."""

    name: str
    config: PretrainedConfig
    tokenizer: str
    renderer: str | None = None


qwen3_moe = TinyModel(
    "qwen3_moe",
    Qwen3MoeConfig(
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=151643,
        decoder_sparse_step=1,
        eos_token_id=151645,
        head_dim=128,
        hidden_act="silu",
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=3072,
        max_position_embeddings=40960,
        max_window_layers=48,
        mlp_only_layers=[],
        moe_intermediate_size=384,
        norm_topk_prob=True,
        num_attention_heads=32,
        num_experts=32,
        num_experts_per_tok=8,
        num_hidden_layers=2,
        num_key_value_heads=4,
        output_router_logits=False,
        rms_norm_eps=1e-06,
        rope_scaling=None,
        rope_theta=1000000.0,
        router_aux_loss_coef=0.001,
        sliding_window=None,
        tie_word_embeddings=False,
        use_sliding_window=False,
        vocab_size=151936,
    ),
    tokenizer="Qwen/Qwen3-30B-A3B",
)

glm_moe_dsa = TinyModel(
    "glm_moe_dsa",
    GlmMoeDsaConfig(
        attention_bias=False,
        attention_dropout=0.0,
        eos_token_id=[154820, 154827, 154829],
        ep_size=1,
        first_k_dense_replace=3,
        hidden_act="silu",
        head_dim=64,
        hidden_size=2048,
        index_head_dim=128,
        index_n_heads=32,
        index_topk=2048,
        indexer_rope_interleave=True,
        initializer_range=0.02,
        intermediate_size=4096,
        kv_lora_rank=512,
        max_position_embeddings=202752,
        moe_intermediate_size=512,
        moe_layer_freq=1,
        n_group=1,
        n_routed_experts=32,
        n_shared_experts=1,
        norm_topk_prob=True,
        num_attention_heads=64,
        num_experts_per_tok=8,
        num_hidden_layers=4,
        num_key_value_heads=64,
        pad_token_id=154820,
        pretraining_tp=1,
        q_lora_rank=2048,
        qk_head_dim=256,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        rms_norm_eps=1e-05,
        rope_interleave=True,
        rope_parameters={"rope_theta": 1000000, "rope_type": "default"},
        routed_scaling_factor=2.5,
        scoring_func="sigmoid",
        tie_word_embeddings=False,
        topk_group=1,
        topk_method="noaux_tc",
        v_head_dim=256,
        vocab_size=154880,
    ),
    tokenizer="zai-org/GLM-5",
)

deepseek_v4 = TinyModel(
    "deepseek_v4",
    DeepseekV4Config(
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=0,
        eos_token_id=1,
        hc_eps=1e-06,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        head_dim=512,
        hidden_act="silu",
        hidden_size=1024,
        index_head_dim=128,
        index_n_heads=64,
        index_topk=512,
        initializer_range=0.02,
        max_position_embeddings=1048576,
        moe_intermediate_size=512,
        n_routed_experts=16,
        n_shared_experts=1,
        norm_topk_prob=True,
        num_attention_heads=64,
        num_experts_per_tok=6,
        num_hidden_layers=4,
        num_hash_layers=3,
        num_key_value_heads=1,
        o_groups=8,
        o_lora_rank=1024,
        q_lora_rank=1024,
        qk_rope_head_dim=64,
        rms_norm_eps=1e-06,
        rope_scaling={
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "type": "yarn",
        },
        rope_theta=10000,
        routed_scaling_factor=1.5,
        scoring_func="sqrtsoftplus",
        sliding_window=128,
        swiglu_limit=10.0,
        tie_word_embeddings=False,
        topk_method="noaux_tc",
        vocab_size=129280,
        compress_rope_theta=160000,
        compress_ratios=[0, 0, 4, 128],
    ),
    tokenizer="deepseek-ai/DeepSeek-V4-Flash-0731",
)

llama = TinyModel(
    "llama",
    LlamaConfig(
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=128000,
        eos_token_id=128009,
        head_dim=64,
        hidden_act="silu",
        hidden_size=2048,
        initializer_range=0.02,
        intermediate_size=8192,
        max_position_embeddings=131072,
        mlp_bias=False,
        num_attention_heads=32,
        num_hidden_layers=2,
        num_key_value_heads=8,
        pad_token_id=128004,
        pretraining_tp=1,
        rms_norm_eps=1e-05,
        rope_scaling={
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        rope_theta=500000.0,
        tie_word_embeddings=True,
        vocab_size=128256,
    ),
    tokenizer="unsloth/Llama-3.2-1B-Instruct",
    renderer="llama-3",
)

qwen3_dense = TinyModel(
    "qwen3_dense",
    Qwen3Config(
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=151643,
        eos_token_id=151645,
        head_dim=128,
        hidden_act="silu",
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=3072,
        max_position_embeddings=40960,
        max_window_layers=28,
        num_attention_heads=16,
        num_hidden_layers=2,
        num_key_value_heads=8,
        rms_norm_eps=1e-06,
        rope_scaling=None,
        rope_theta=1000000,
        sliding_window=None,
        tie_word_embeddings=True,
        use_sliding_window=False,
        vocab_size=151936,
    ),
    tokenizer="Qwen/Qwen3-0.6B",
)

qwen3_5_dense = TinyModel(
    "qwen3_5_dense",
    Qwen3_5TextConfig(
        attention_bias=False,
        attention_dropout=0.0,
        attn_output_gate=True,
        eos_token_id=248044,
        full_attention_interval=4,
        head_dim=256,
        hidden_act="silu",
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=3584,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_value_head_dim=128,
        max_position_embeddings=262144,
        mlp_only_layers=[],
        num_attention_heads=8,
        num_hidden_layers=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-06,
        tie_word_embeddings=True,
        vocab_size=248320,
        mamba_ssm_dtype="float32",
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
    ),
    tokenizer="Qwen/Qwen3.5-0.8B",
)

glm4_moe = TinyModel(
    "glm4_moe",
    Glm4MoeConfig(
        attention_bias=True,
        attention_dropout=0.0,
        pad_token_id=151329,
        eos_token_id=[151329, 151336, 151338],
        head_dim=128,
        hidden_act="silu",
        hidden_size=1024,
        partial_rotary_factor=0.5,
        initializer_range=0.02,
        intermediate_size=2816,
        max_position_embeddings=131072,
        moe_intermediate_size=384,
        norm_topk_prob=True,
        num_attention_heads=96,
        n_group=1,
        topk_group=1,
        n_routed_experts=32,
        n_shared_experts=1,
        routed_scaling_factor=1.0,
        num_experts_per_tok=8,
        first_k_dense_replace=1,
        num_hidden_layers=2,
        num_key_value_heads=8,
        rms_norm_eps=1e-05,
        rope_scaling=None,
        rope_theta=1000000,
        tie_word_embeddings=False,
        use_qk_norm=False,
        vocab_size=151552,
    ),
    tokenizer="zai-org/GLM-4.5-Air",
)

gpt_oss = TinyModel(
    "gpt_oss",
    GptOssConfig(
        attention_bias=True,
        attention_dropout=0.0,
        eos_token_id=200002,
        experts_per_token=4,
        head_dim=64,
        hidden_act="silu",
        hidden_size=1024,
        initial_context_length=4096,
        initializer_range=0.02,
        intermediate_size=1024,
        layer_types=["sliding_attention", "full_attention"],
        max_position_embeddings=131072,
        num_attention_heads=64,
        num_experts_per_tok=4,
        num_hidden_layers=2,
        num_key_value_heads=8,
        num_local_experts=16,
        output_router_logits=False,
        pad_token_id=199999,
        rms_norm_eps=1e-05,
        rope_scaling={
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "rope_type": "yarn",
            "truncate": False,
        },
        rope_theta=150000,
        router_aux_loss_coef=0.9,
        sliding_window=128,
        swiglu_limit=7.0,
        tie_word_embeddings=False,
        vocab_size=201088,
    ),
    tokenizer="openai/gpt-oss-20b",
)

laguna = TinyModel(
    "laguna",
    LagunaConfig(
        vocab_size=100352,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=2,
        num_attention_heads=48,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=262144,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-06,
        num_experts=32,
        num_experts_per_tok=8,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
        router_aux_loss_coef=0.0,
        bos_token_id=2,
        eos_token_id=[2, 24],
        pad_token_id=9,
        tie_word_embeddings=False,
        gating=True,
        sliding_window=512,
        rope_parameters={
            "full_attention": {
                "rope_theta": 500000.0,
                "rope_type": "yarn",
                "factor": 64.0,
                "original_max_position_embeddings": 4096,
                "beta_slow": 1.0,
                "beta_fast": 64.0,
                "attention_factor": 1.4158883083359672,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
        },
        layer_types=["full_attention", "sliding_attention"],
        moe_apply_router_weight_on_input=False,
        partial_rotary_factor=0.5,
        mlp_layer_types=["dense", "sparse"],
        moe_routed_scaling_factor=2.5,
        num_attention_heads_per_layer=[48, 64],
    ),
    tokenizer="poolside/Laguna-XS.2",
)

minimax_m2 = TinyModel(
    "minimax_m2",
    MiniMaxM2Config(
        attn_type_list=[1, 1],
        head_dim=128,
        hidden_act="silu",
        hidden_size=1024,
        intermediate_size=512,
        max_position_embeddings=196608,
        num_attention_heads=48,
        num_experts_per_tok=8,
        num_hidden_layers=2,
        num_key_value_heads=8,
        num_local_experts=32,
        qk_norm_type="per_layer",
        rms_norm_eps=1e-06,
        rope_theta=5000000,
        rotary_dim=64,
        scoring_func="sigmoid",
        shared_intermediate_size=0,
        tie_word_embeddings=False,
        use_qk_norm=True,
        use_routing_bias=True,
        vocab_size=200064,
    ),
    tokenizer="MiniMaxAI/MiniMax-M2.5",
)

nemotron_h = TinyModel(
    "nemotron_h",
    NemotronHConfig(
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=1,
        chunk_size=128,
        conv_kernel=4,
        eos_token_id=2,
        expand=2,
        head_dim=128,
        hidden_dropout=0.0,
        hidden_size=1024,
        hybrid_override_pattern="MEMEM*",
        initializer_range=0.02,
        intermediate_size=512,
        layer_norm_epsilon=1e-05,
        mamba_head_dim=64,
        mamba_hidden_act="silu",
        mamba_num_heads=64,
        mamba_proj_bias=False,
        mamba_ssm_cache_dtype="float32",
        max_position_embeddings=262144,
        mlp_bias=False,
        mlp_hidden_act="relu2",
        moe_intermediate_size=512,
        moe_shared_expert_intermediate_size=1024,
        n_group=1,
        n_groups=8,
        n_routed_experts=32,
        n_shared_experts=1,
        norm_eps=1e-05,
        norm_topk_prob=True,
        num_attention_heads=32,
        num_experts_per_tok=6,
        num_hidden_layers=6,
        num_key_value_heads=2,
        num_logits_to_keep=1,
        pad_token_id=0,
        partial_rotary_factor=1.0,
        rescale_prenorm_residual=True,
        residual_in_fp32=False,
        rope_theta=10000,
        routed_scaling_factor=2.5,
        sliding_window=None,
        ssm_state_size=128,
        tie_word_embeddings=False,
        time_step_floor=0.0001,
        time_step_max=0.1,
        time_step_min=0.001,
        topk_group=1,
        use_bias=False,
        use_conv_bias=True,
        use_mamba_kernels=True,
        vocab_size=131072,
    ),
    tokenizer="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
)

qwen3_5_moe = TinyModel(
    "qwen3_5_moe",
    Qwen3_5MoeTextConfig(
        attention_bias=False,
        attention_dropout=0.0,
        attn_output_gate=True,
        eos_token_id=248044,
        full_attention_interval=4,
        head_dim=256,
        hidden_act="silu",
        hidden_size=1024,
        initializer_range=0.02,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_value_head_dim=128,
        max_position_embeddings=262144,
        mlp_only_layers=[],
        moe_intermediate_size=256,
        num_attention_heads=16,
        num_experts=32,
        num_experts_per_tok=8,
        num_hidden_layers=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-06,
        router_aux_loss_coef=0.001,
        shared_expert_intermediate_size=256,
        vocab_size=248320,
        mamba_ssm_dtype="float32",
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
    ),
    tokenizer="Qwen/Qwen3.5-35B-A3B",
)

qwen4_exp = TinyModel(
    "qwen4_exp",
    Qwen3_8FlashNextTextConfig(
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=248044,
        eos_token_id=248044,
        full_attention_interval=4,
        hc_count=4,
        hc_lowrank=320,
        head_dim=256,
        heads_per_ngram=8,
        hidden_act="silu",
        hidden_size=1024,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        indexer_head_dim=128,
        indexer_kv_heads=1,
        indexer_n_heads=4,
        initializer_range=0.02,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_value_head_dim=128,
        make_ngram_vocab_size_divisible_by=128,
        mamba_ssm_dtype="float32",
        max_position_embeddings=262144,
        moe_intermediate_size=256,
        ngram_size=3,
        ngram_vocab_size_base=200000,
        num_attention_heads=24,
        num_experts=64,
        num_experts_per_tok=10,
        num_hidden_layers=4,
        num_key_value_heads=2,
        output_gate_type="sigmoid",
        output_router_logits=False,
        pad_token_id=None,
        partial_rotary_factor=0.25,
        ple_conv_kernel_size=4,
        ple_embed_dim=1024,
        ple_layer_ids=[2],
        rms_norm_eps=1e-06,
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        router_aux_loss_coef=0.001,
        shared_expert_intermediate_size=256,
        split_ngram_parts=128,
        tie_word_embeddings=False,
        vocab_size=248320,
    ),
    tokenizer="Qwen/Qwen3.8-Flash-Next",
)


# TODO: build DSv4's `tid2eid` in the model when it is created from a config, then drop this.
def _fill_hash_tables(model: torch.nn.Module) -> None:
    """Give every DSv4 hash router a `tid2eid` table with distinct experts per token, as a real one has."""
    for router in model.modules():
        if isinstance(router, DeepseekV4HashRouter):
            rows = [torch.randperm(router.num_experts)[: router.top_k] for _ in range(router.tid2eid.shape[0])]
            router.tid2eid.copy_(torch.stack(rows))


# TODO: initialize MoE weights in `_from_config` (not on the `from_pretrained` path, which loads them), then drop this.
def _init_moe_weights(model: torch.nn.Module, init_std: float) -> None:
    """Initialize every MoE layer's experts and router, which `_from_config` leaves as uninitialized memory."""
    for moe in model.modules():
        if isinstance(moe, MoE):
            moe.init_weights(init_std, torch.device("cpu"))


def write_tiny_checkpoint(tiny_model: TinyModel, out_dir: Path) -> Path:
    """Write a seeded tiny HF safetensors checkpoint for `tiny_model` and verify its HF -> PrimeRL roundtrip."""
    config = tiny_model.config
    model_cls = get_custom_causal_lm_cls(config)
    torch.manual_seed(SEED)
    with torch.device("cpu"), default_dtype(torch.bfloat16), torch.no_grad():
        model = model_cls._from_config(config)
        model.init_buffers_post_meta()
        _init_moe_weights(model, config.initializer_range)
        _fill_hash_tables(model)

    state_dict = dict(model.state_dict())
    if config.tie_word_embeddings:
        state_dict.pop("lm_head.weight")
    prime_state_dict = {key: value.clone() for key, value in state_dict.items()}
    hf_state_dict = model.convert_to_hf(state_dict)
    save_state_dict(hf_state_dict, out_dir)
    config.save_pretrained(out_dir)

    reloaded: dict[str, Tensor] = load_state_dict(out_dir)
    model.convert_to_prime(reloaded)
    assert reloaded.keys() == prime_state_dict.keys(), reloaded.keys() ^ prime_state_dict.keys()
    for key, value in prime_state_dict.items():
        assert torch.equal(reloaded[key], value), f"{tiny_model.name}: roundtrip mismatch at {key}"
    return out_dir
