"""Tiny seeded models and packed Reverse-Text-SFT rows, shared by the trainer's 2-GPU tests."""

from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

import torch
from renderers import AutoRendererConfig
from transformers import AutoModelForCausalLM, Glm4MoeConfig, LlamaConfig, PretrainedConfig, Qwen3Config, Qwen3MoeConfig

from prime_rl.configs.sft import SFTDataConfig
from prime_rl.configs.trainer import CompileConfig, ModelConfig
from prime_rl.trainer.data import setup_dataloader
from prime_rl.trainer.model import forward, resolve_auto_attn
from prime_rl.trainer.models import get_custom_causal_lm_cls
from prime_rl.trainer.models.afmoe import AfmoeConfig
from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.glm5_next import Glm5NextTextConfig
from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaConfig
from prime_rl.trainer.models.gpt_oss import GptOssConfig
from prime_rl.trainer.models.inkling2 import Inkling2TextConfig
from prime_rl.trainer.models.inkling2_loop import Inkling2LoopTextConfig
from prime_rl.trainer.models.laguna import LagunaConfig
from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX
from prime_rl.trainer.models.minimax_m2 import MiniMaxM2Config
from prime_rl.trainer.models.muse_glimmer import MuseGlimmerTextConfig
from prime_rl.trainer.models.nemotron_h import NemotronHConfig
from prime_rl.trainer.models.qwen3_5 import Qwen3_5MoeTextConfig, Qwen3_5TextConfig
from prime_rl.trainer.models.qwen3_8_flash_next import Qwen3_8FlashNextTextConfig
from prime_rl.trainer.sft.data import setup_sft_dataset
from prime_rl.trainer.world import get_world
from prime_rl.utils.cp import setup_cp_params, shard_for_cp
from prime_rl.utils.weights import save_state_dict

TOKENIZER = "Qwen/Qwen3-0.6B"
NUM_ROWS = 4
ROW_LEN = 512

TINY_DIMS = dict(
    # Every model reads the same rows, tokenized with TOKENIZER
    vocab_size=151936,
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    max_position_embeddings=4096,
)
MOE_DIMS = dict(moe_intermediate_size=128, num_experts_per_tok=2, tie_word_embeddings=False)
TINY_MODELS: dict[str, PretrainedConfig] = {
    "qwen3": Qwen3Config(**TINY_DIMS, intermediate_size=512, tie_word_embeddings=True),
    "llama": LlamaConfig(**TINY_DIMS, intermediate_size=512, tie_word_embeddings=False),
    "qwen3_moe": Qwen3MoeConfig(**TINY_DIMS, **MOE_DIMS, num_experts=8),
    "glm4_moe": Glm4MoeConfig(
        **TINY_DIMS, **MOE_DIMS, n_routed_experts=8, n_shared_experts=1, first_k_dense_replace=1, intermediate_size=512
    ),
}
# Every layer kind of each arch with small widths and 8 experts. Head counts and head dims stay at the published
# values where kernels require them: FlashMLA serves 64 or 128 query heads of dim 512, and Qwen's mrope sections cover
# a quarter of a 256-dim head. DeepSeek-V4 and GLM-5 build with the TileLang backend, whose constructor does not check
# the GPU; the trainer picks `model.dsa_backend` itself.
QWEN_MROPE = {
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "rope_type": "default",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
}
QWEN_NEXT_DIMS = dict(
    vocab_size=151936,
    hidden_size=256,
    num_hidden_layers=2,
    layer_types=["linear_attention", "full_attention"],
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=256,
    linear_num_key_heads=4,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=128,
    shared_expert_intermediate_size=128,
    rope_parameters=QWEN_MROPE,
    max_position_embeddings=4096,
    bos_token_id=151643,
    eos_token_id=151645,
)
TINY_PRIME_RL_MODELS: dict[str, PretrainedConfig] = {
    "qwen3_5_moe": Qwen3_5MoeTextConfig(**QWEN_NEXT_DIMS, linear_num_value_heads=8, tie_word_embeddings=False),
    "qwen3_8_flash_next": Qwen3_8FlashNextTextConfig(
        **QWEN_NEXT_DIMS,
        linear_num_value_heads=12,
        # A budget below the documents' lengths, so the indexer drops keys
        indexer_budget=64,
        hc_lowrank=64,
        ple_layer_ids=[1],
        ple_embed_dim=256,
        ngram_vocab_size_base=4096,
        split_ngram_parts=4,
    ),
    "deepseek_v4": DeepseekV4Config(
        vocab_size=151936,
        hidden_size=256,
        num_hidden_layers=5,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        # The packed Reverse-Text-SFT documents are ~90-100 tokens, and a document shorter than the compress rate gets
        # no compressed entry, so the published rate of 128 would leave the heavily compressed attention unused
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 16},
        q_lora_rank=256,
        o_lora_rank=128,
        index_topk=64,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        num_hash_layers=3,
        max_position_embeddings=4096,
        dsa_backend="tilelang",
    ),
    "glm_moe_dsa": GlmMoeDsaConfig(
        vocab_size=151936,
        hidden_size=256,
        num_hidden_layers=2,
        first_k_dense_replace=1,
        intermediate_size=512,
        moe_intermediate_size=128,
        q_lora_rank=256,
        index_topk=128,
        n_routed_experts=8,
        num_experts_per_tok=2,
        pad_token_id=None,
        max_position_embeddings=4096,
        dsa_backend="tilelang",
    ),
    "qwen3_5": Qwen3_5TextConfig(
        **TINY_DIMS,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        intermediate_size=512,
    ),
    "afmoe": AfmoeConfig(
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
    ),
    "laguna": LagunaConfig(
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
    ),
    "minimax_m2": MiniMaxM2Config(
        **TINY_DIMS,
        intermediate_size=128,
        num_local_experts=8,
        num_experts_per_tok=2,
        rotary_dim=32,
        scoring_func="sigmoid",
        use_routing_bias=True,
        use_qk_norm=True,
        qk_norm_type="per_layer",
    ),
    "glm5_next": Glm5NextTextConfig(
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
    ),
    "nemotron_h": NemotronHConfig(
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
    ),
    "muse_glimmer": MuseGlimmerTextConfig(
        **TINY_DIMS,
        intermediate_size=512,
        sliding_window=32,
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
        layer_rope_theta=[500000.0, 0],
    ),
    "gpt_oss": GptOssConfig(
        **TINY_DIMS,
        intermediate_size=128,
        num_local_experts=8,
        num_experts_per_tok=2,
        sliding_window=32,
    ),
    # Two blocks of two layers: sliding-window layers that start a block (0, 2) and one that does not (1), and a
    # global layer (3)
    "inkling2": Inkling2TextConfig(
        vocab_size=151936,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=64,
        d_rel=16,
        rel_extent=128,
        sliding_window_size=128,
        local_layer_ids=[0, 1, 2],
        n_layers_per_attn_res_block=2,
        log_scaling_n_floor=64,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
    ),
    # Blocks of two layers, a sliding then a full one, with layers 2..5 looped three times. The window, the relative
    # bias extent and the log-scaling floor are all below the ~90-100 tokens of a packed document. The clip thresholds
    # and softcaps are sized for these widths, so that the weights of `init_inkling2_loop_weights` reach them
    "inkling2_loop": Inkling2LoopTextConfig(
        vocab_size=151936,
        hidden_size=256,
        num_hidden_layers=8,
        n_layers_per_attn_res_block=2,
        loop_config={"core_start": 2, "core_end": 6, "num_passes": 3, "gate_rank": 16, "is_varloop": False},
        local_layer_ids=[0, 2, 4, 6],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=64,
        sliding_window_size=32,
        rel_extent=48,
        log_scaling_n_floor=32,
        qk_l2_clip_threshold=8.0,
        rel_l2_clip_threshold=4.0,
        n_routed_experts=8,
        num_experts_per_tok=2,
        intermediate_size=128,
        route_scale=2.0,
        mlp_gate_softcap=0.5,
        mlp_up_softcap=0.75,
        final_logit_softcapping=4.0,
        eos_token_id=151645,
    ),
}
# Second names of TINY_PRIME_RL_MODELS entries, under which `save_tiny_model` builds the entry's config again.
# `inkling2_loop_generic` is `inkling2_loop` on the generic seeding alone, for the compiled case (see the tolerances
# in test_distributed_invariance.py). An entry of TINY_PRIME_RL_MODELS gets a compile case in test_compile.py, a
# second name does not.
TINY_MODEL_ALIASES = {"inkling2_loop_generic": "inkling2_loop"}


@dataclass(frozen=True)
class Case:
    name: str
    model: str
    cp: int = 1
    ep: int = 1
    compile: bool = False
    """torch.compile costs seconds per case, so few cases turn it on."""
    fullgraph: bool = False
    overrides: dict = field(default_factory=dict)
    """ModelConfig fields on top of the trainer defaults (full AC with activation offloading, fusions)."""
    tolerances: dict = field(default_factory=dict)
    """Tolerances on top of the invariance test's defaults, each measured for this case."""


def model_config(case: Case, models_dir: Path) -> ModelConfig:
    # SFT does not freeze the router, and the ModelConfig default "auto" is resolved by the trainer config
    config = ModelConfig(
        name=(models_dir / case.model).as_posix(),
        cp=case.cp,
        ep=case.ep,
        compile=CompileConfig(fullgraph=case.fullgraph) if case.compile else None,
        freeze_moe_router=False,
        **case.overrides,
    )
    resolve_auto_attn(config)
    return config


def save_tiny_model(name: str, models_dir: Path) -> dict:
    """An HF checkpoint, which the trainer converts to its own format on load. Every rank builds the model, so that a
    build that fails on this GPU fails on all ranks alike, and the master saves it."""
    torch.manual_seed(0)
    if name in TINY_MODELS:
        model = AutoModelForCausalLM.from_config(TINY_MODELS[name])
        if get_world().is_master:
            model.save_pretrained(models_dir / name)
    else:
        config = TINY_PRIME_RL_MODELS[TINY_MODEL_ALIASES.get(name, name)]
        model = get_custom_causal_lm_cls(config).from_config(config)
        init_weights(model)
        if name == "inkling2_loop":
            init_inkling2_loop_weights(model)
        if get_world().is_master:
            save_state_dict(model.convert_to_hf(model.state_dict()), models_dir / name)
            config.save_pretrained(models_dir / name)
    return {}


@torch.no_grad()
def init_weights(model: torch.nn.Module) -> None:
    """Seeded values for every floating parameter, since `from_config` leaves some of them uninitialized, and a
    DeepSeek-V4 hash routing table with distinct experts per token."""
    for name, param in model.named_parameters():
        if param.ndim == 1 and name.endswith("norm.weight"):
            param.fill_(1.0)
        else:
            param.normal_(0.0, 0.02)
    for module in model.modules():
        if hasattr(module, "tid2eid"):
            experts = torch.rand(module.tid2eid.shape[0], module.num_experts).argsort(dim=1)
            module.tid2eid.copy_(experts[:, : module.top_k])


# Std of the Inkling 2 Mini projections in units of `1 / sqrt(fan_in)`, by name fragment; 1 for the others. The key
# convolutions halve the keys, a sink's logit is compared with the scores on those keys, and the experts and message
# gates compound over the 16 layer visits.
INKLING2_LOOP_PROJECTION_SCALES = {
    ".k_proj.": 2.0,
    ".sink_proj.": 2.0,
    "weight_proj": 0.5,
    ".experts.": 0.5,
    ".message_gates.": 0.5,
}


@torch.no_grad()
def init_inkling2_loop_weights(model: torch.nn.Module) -> None:
    """Inkling 2 Mini weights that make its attention matter. At the generic 0.02 the clip scales and the depth-mix and
    message queries are near zero: attention and depth mixes are uniform, no clip binds, the softcaps stay linear,
    and the gradients of the query, key and relative-bias parameters fall far below the invariance test's
    `MIN_GRAD_SHARE`. Here the clip scales are around 1, the queries and convolution taps are O(1), and the projections
    are large enough for about half of the query, key and relative-state vectors to be clipped and for the softcaps to
    bend."""
    for name, param in model.named_parameters():
        if name.endswith(("q_scale", "rel_scale")):
            param.normal_(1.0, 0.1)
        elif name.endswith(("query", "message_queries")):
            param.normal_(0.0, 0.5)
        elif name.endswith(("conv.weight", "static_weight")):
            param.normal_(0.0, 0.3)
        elif param.ndim > 1:
            scale = next(
                (scale for fragment, scale in INKLING2_LOOP_PROJECTION_SCALES.items() if fragment in name), 1.0
            )
            param.normal_(0.0, scale / param.shape[-1] ** 0.5)


def load_rows(tokenizer, num_rows: int = NUM_ROWS) -> list[dict]:
    """The first rows of Reverse-Text-SFT, packed as the SFT trainer packs them, identical on every rank. Rows of one
    length keep torch.compile from recompiling for a second shape."""
    config = SFTDataConfig(
        source=[{"name": "reverse-text", "dataset": "PrimeIntellect/Reverse-Text-SFT"}],
        seq_len=ROW_LEN,
        shuffle=False,
        pad_to_seq_len=True,
    )
    dataset = setup_sft_dataset(tokenizer, config, dp_rank=0, dp_size=1, renderer_config=AutoRendererConfig())
    return list(islice(setup_dataloader(dataset, config, cp=2), num_rows))


def compute_loss(
    model, row: dict, cp_rank: int, cp_size: int, routed_experts: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """The SFT trainer's `compute_loss`: summed cross-entropy and token count of this rank's share of a row.
    `routed_experts` replays an MoE routing, as RL's router replay does."""
    input_ids, position_ids, target_ids, loss_mask = (
        row[key].cuda() for key in ("input_ids", "position_ids", "target_ids", "loss_mask")
    )
    if routed_experts is not None:
        routed_experts = routed_experts.cuda()
    if cp_size > 1:
        input_ids, position_ids = setup_cp_params(input_ids, position_ids, cp_rank, cp_size)
        target_ids = shard_for_cp(target_ids, cp_rank=cp_rank, cp_world_size=cp_size)
        loss_mask = shard_for_cp(loss_mask, cp_rank=cp_rank, cp_world_size=cp_size)
        if routed_experts is not None:
            routed_experts = shard_for_cp(routed_experts, cp_rank=cp_rank, cp_world_size=cp_size)
    labels = target_ids.masked_fill(~loss_mask, IGNORE_INDEX)
    out = forward(
        model,
        input_ids,
        position_ids,
        seq_lens=row["seq_lens"].cuda(),
        labels=labels,
        routed_experts=routed_experts,
        seq_lens_are_pre_shard=cp_size > 1,
    )
    return out["loss"], loss_mask.sum()
