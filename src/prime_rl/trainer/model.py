import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

# Disable transformers hub kernel interception. Installed hub kernels can otherwise replace
# modules with implementations that have incompatible CUDA requirements.
os.environ.setdefault("USE_HUB_KERNELS", "NO")

import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
from huggingface_hub import snapshot_download
from jaxtyping import Int
from torch import Tensor
from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, OffloadPolicy, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_common import FSDPMeshInfo, ShardPlacementResult
from torch.distributed.fsdp._fully_shard._fsdp_init import _get_mesh_info
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import parallelize_module
from transformers import AutoConfig, AutoTokenizer, GenerationConfig, PretrainedConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.import_utils import is_flash_attn_3_available

from prime_rl.configs.trainer import (
    ActivationCheckpointConfig,
    CompileConfig,
    FP8Config,
    ModelConfig,
    MXFP8Config,
    TokenizerConfig,
)
from prime_rl.multimodal import ForwardPolicy
from prime_rl.trainer.activation_checkpointing import get_activation_checkpoint_wrapper
from prime_rl.trainer.distributed.embedding_parallel import EmbeddingParallel
from prime_rl.trainer.lora import apply_lora_to_model, freeze_all_except_lora_and_specified, strip_lora_from_state_dict
from prime_rl.trainer.models import (
    AutoModelForCausalLMPrimeRL,
    PrimeLmOutput,
    cast_float_and_contiguous,
    get_custom_causal_lm_cls,
    get_custom_vlm_cls,
    supports_custom_impl,
)
from prime_rl.trainer.models.deepseek_v4.attention import DeepseekV4Indexer
from prime_rl.trainer.models.deepseek_v4.moe import DeepseekV4HashRouter
from prime_rl.trainer.models.fusions import (
    apply_model_fusions,
    get_fsdp_shard_placement_fn,
    write_back_loaded_packed_parameters,
)
from prime_rl.trainer.models.glm_moe_dsa.sparse_mla_attention import Indexer
from prime_rl.trainer.models.layers.fp8_linear import replace_linear_with_fp8_blockwise_linear
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.layers.moe import MoE, TokenChoiceTopKRouter
from prime_rl.trainer.models.layers.mxfp8_linear import replace_linear_with_mxfp8_linear
from prime_rl.trainer.models.qwen3_8_flash_next.indexer import SparseAttentionIndexer
from prime_rl.trainer.models.qwen3_8_flash_next.ngram_embedding import NGramEmbedding
from prime_rl.trainer.moe_runtime import configure_moe_runtime
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import format_time
from prime_rl.utils.vlm import get_language_model, get_vision_encoder, is_vlm_architecture
from prime_rl.utils.weights import (
    load_state_dict,
    load_state_dict_keys,
    save_state_dict,
)


def pre_download_model(model_name: str, *, skip_weights: bool = False) -> None:
    """Pre-download model from HuggingFace Hub so all nodes have cached weights before training.

    With ``skip_weights`` (random-init debug runs), only config and tokenizer files are fetched.
    """
    if Path(model_name).exists():
        get_logger().info(f"Model {model_name} found at local path, skipping download")
        return
    t0 = time.perf_counter()
    if skip_weights:
        get_logger().info(f"Pre-downloading config and tokenizer for {model_name} (random init, skipping weights)")
        path = snapshot_download(
            repo_id=model_name, repo_type="model", allow_patterns=["*.json", "*.txt", "tokenizer*", "*.jinja"]
        )
    else:
        get_logger().info(f"Pre-downloading model {model_name}")
        path = snapshot_download(repo_id=model_name, repo_type="model")
    get_logger().debug(
        f"Finished pre-downloading model {model_name} to {path} in {format_time(time.perf_counter() - t0)}"
    )


# Add filter to the standard logging module for transformers.modeling_utils to supress the
# flash attention dtype warnings since FSDP is used to handle mixed precision.
transformers_modeling_utils_logger = logging.getLogger("transformers.modeling_utils")
transformers_modeling_utils_logger.addFilter(
    lambda record: "Flash Attention 2 only supports torch.float16 and torch.bfloat16 dtypes" not in record.getMessage()
)

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# We increase the torch.compile recompile limit and cache size as we found this
# necessary for training INTELLECT-3 with Muon.
torch._dynamo.config.recompile_limit = 16  # default: 8
torch._dynamo.config.cache_size_limit = 64  # default: 8


def freeze_vision_encoder(model: nn.Module, override_attr: str | None = None) -> None:
    logger = get_logger()
    vision_encoder = get_vision_encoder(model, override=override_attr)
    if vision_encoder is None:
        raise ValueError("Could not find vision encoder to freeze")
    num_frozen = 0
    for param in vision_encoder.parameters():
        param.requires_grad = False
        num_frozen += 1
    logger.info(f"Froze {num_frozen} parameters in vision encoder")


def freeze_moe_router(model: nn.Module) -> None:
    """Freeze MoE router parameters to maintain stable routing during training."""
    logger = get_logger()
    language_model = get_language_model(model)
    num_frozen = 0

    for layer in language_model.layers:
        mlp = layer.mlp if hasattr(layer, "mlp") else layer.feed_forward if hasattr(layer, "feed_forward") else None
        if isinstance(mlp, MoE):
            for param in mlp.router.parameters():
                param.requires_grad = False
                num_frozen += 1

    if num_frozen == 0:
        raise ValueError("No MoE router parameters found to freeze. Is this a MoE model?")

    logger.info(f"Froze {num_frozen} MoE router parameters")


def apply_fp32_moe_router(model: nn.Module) -> None:
    """Cast MoE router gates to fp32 so routing runs in fp32 in forward and backward.

    The FSDP bf16 cast exemption is applied separately in `setup_fsdp`.
    """
    logger = get_logger()
    language_model = get_language_model(model)
    num_routers = 0

    for layer in language_model.layers:
        mlp = layer.mlp if hasattr(layer, "mlp") else layer.feed_forward if hasattr(layer, "feed_forward") else None
        if isinstance(mlp, MoE):
            mlp.router.to(torch.float32)
            if isinstance(mlp.router, TokenChoiceTopKRouter):
                mlp.router.fp32_gate = True
            num_routers += 1

    # No-op for non-MoE models: moe_router_dtype='float32' is the default,
    # so absence of MoE routers is the common case, not an error.
    if num_routers > 0:
        logger.info(f"Running {num_routers} MoE router gates in fp32")


def get_full_offload_dtype_policy(
    model: nn.Module,
    config: ModelConfig,
) -> dict[int, tuple[torch.dtype, torch.dtype]]:
    """Return persistent parameter and reduced-gradient dtypes for full offload."""
    # FSDP casts reduced gradients back to the persistent sharded-parameter dtype.
    policy = {id(param): (torch.bfloat16, torch.bfloat16) for param in model.parameters() if param.is_floating_point()}
    if config.moe_router_dtype != "float32":
        return policy

    language_model = get_language_model(model)
    for layer in language_model.layers:
        mlp = layer.mlp if hasattr(layer, "mlp") else layer.feed_forward if hasattr(layer, "feed_forward") else None
        if isinstance(mlp, MoE):
            for param in mlp.router.parameters():
                if param.is_floating_point():
                    policy[id(param)] = (torch.float32, torch.float32)
    return policy


def freeze_sparse_indexer(model: nn.Module) -> None:
    """Freeze sparse-attention indexer parameters.

    An indexer forward runs under `torch.no_grad()`, so its params never receive a gradient
    and cannot be trained. Left with requires_grad=True they stay stateless in the optimizer,
    which breaks strict checkpoint resume: DCP materializes optimizer state for every
    requires_grad param at load time, but the stateless params were never saved -> "Missing
    key in checkpoint state_dict". Freezing them keeps the saved and loaded optimizer state
    symmetric.
    """
    # TODO: no model here trains its indexer. DeepSeek's auxiliary KL objective, which supervises
    # the top-k selection, is unimplemented, so these params are frozen rather than learned.
    logger = get_logger()
    num_frozen = 0

    for module in model.modules():
        if isinstance(module, (Indexer, DeepseekV4Indexer, SparseAttentionIndexer)):
            for param in module.parameters():
                param.requires_grad = False
                num_frozen += 1

    if num_frozen > 0:
        logger.info(f"Froze {num_frozen} sparse indexer parameters")


def apply_force_balanced_routing(model: nn.Module) -> None:
    """Force MoE token-choice routers into round-robin assignment for fake-data smoke tests."""
    logger = get_logger()
    language_model = get_language_model(model)
    num_routers = 0

    for layer in language_model.layers:
        mlp = layer.mlp if hasattr(layer, "mlp") else layer.feed_forward if hasattr(layer, "feed_forward") else None
        if isinstance(mlp, MoE):
            mlp.router.force_balanced = True
            num_routers += 1

    if num_routers == 0:
        raise ValueError("No MoE routers found to force-balance. Is this a MoE model?")

    logger.warning(
        f"Forced balanced routing on {num_routers} MoE layers (debug.force_balanced_routing=True). "
        "Expert assignment is round-robin; gradient flow through the router is broken."
    )


def apply_router_replay_filter(model: nn.Module, min_score_ratio: float) -> None:
    """Filter implausible replayed experts in every learned MoE router (see `filter_replayed_experts`)."""
    num_routers = 0
    for layer in get_language_model(model).layers:
        mlp = layer.mlp if hasattr(layer, "mlp") else layer.feed_forward if hasattr(layer, "feed_forward") else None
        # Hash routers select by token id, not by score, so there is nothing to filter.
        if isinstance(mlp, MoE) and not isinstance(mlp.router, DeepseekV4HashRouter):
            mlp.router.replay_min_score_ratio = min_score_ratio
            num_routers += 1

    if num_routers == 0:
        raise ValueError("No MoE routers found for router_replay_min_score_ratio. Is this a custom-impl MoE model?")
    get_logger().info(
        f"Filtering replayed experts below {min_score_ratio} x the router's own top-k on {num_routers} MoE layers"
    )


def is_tt_moe_model(model: nn.Module) -> bool:
    config = getattr(model.config, "text_config", model.config)
    return hasattr(config, "num_experts") or hasattr(config, "n_routed_experts")


def get_load_balance_stats(
    model: nn.Module,
    group: dist.ProcessGroup | None = None,
) -> dict[str, Tensor | None]:
    """Compute routing stats after summing raw counts across the group, if given.

    Also returns this rank's raw per-layer expert counts (`[num_moe_layers, num_experts]`) for step-level stats.
    """
    per_layer_max_vio = []
    per_layer_routing_confidence = []
    language_model = get_language_model(model)
    block_mlps = []
    for transformer_block in language_model.layers:
        # This is necessary for models that have mixed dense layers
        block_mlp = getattr(transformer_block, "mlp", None)
        if block_mlp is not None and hasattr(block_mlp, "tokens_per_expert"):
            block_mlps.append(block_mlp)
    if not block_mlps:
        return {"max_vio": None, "routing_confidence": None, "tokens_per_expert": torch.empty(0, 0)}

    local_tokens_per_expert = torch.stack([block_mlp.tokens_per_expert for block_mlp in block_mlps])
    layer_stats = [(block_mlp.tokens_per_expert, block_mlp.routing_confidence_sum) for block_mlp in block_mlps]
    if group is not None:
        sizes = [tokens_per_expert.numel() + 1 for tokens_per_expert, _ in layer_stats]
        packed_stats = torch.cat(
            [torch.cat((tokens_per_expert, confidence.reshape(1))) for tokens_per_expert, confidence in layer_stats]
        )
        dist.all_reduce(packed_stats, op=dist.ReduceOp.SUM, group=group)
        layer_stats = [(stats[:-1], stats[-1]) for stats in packed_stats.split(sizes)]

    for block_mlp, (tokens_per_expert, routing_confidence_sum) in zip(block_mlps, layer_stats):
        num_routed_tokens = tokens_per_expert.sum() / block_mlp.router.top_k
        tokens_per_expert = tokens_per_expert.sort(dim=0, descending=True).values[block_mlp.router.top_k :]
        balanced_load = tokens_per_expert.mean()
        max_vio = (tokens_per_expert.max() - balanced_load) / balanced_load
        per_layer_max_vio.append(max_vio.detach())

        routing_confidence = routing_confidence_sum / num_routed_tokens
        per_layer_routing_confidence.append(routing_confidence.detach())

        block_mlp.tokens_per_expert.zero_()
        block_mlp.routing_confidence_sum.zero_()
    return {
        "max_vio": torch.stack(per_layer_max_vio),
        "routing_confidence": torch.stack(per_layer_routing_confidence),
        "tokens_per_expert": local_tokens_per_expert,
    }


def get_global_moe_stats(
    model: nn.Module,
    ep_group: dist.ProcessGroup | None,
    dp_cp_group: dist.ProcessGroup,
) -> tuple[dict[str, Tensor], Tensor]:
    """Reduce one microstep's routing stats across EP, then DP and CP ranks.

    Also returns this rank's unreduced per-layer expert counts, to accumulate over the step for `get_expert_load_stats`.
    """
    stats = {}
    load_balance_stats = get_load_balance_stats(model, group=ep_group)
    tokens_per_expert = load_balance_stats.pop("tokens_per_expert")
    for name, values in load_balance_stats.items():
        if values is None:
            continue
        value = values.max() if name == "max_vio" else values.mean()
        if name == "max_vio":
            dist.all_reduce(value, op=dist.ReduceOp.MAX, group=dp_cp_group)
        else:
            dist.all_reduce(value, op=dist.ReduceOp.SUM, group=dp_cp_group)
            value /= dist.get_world_size(dp_cp_group)
        stats[name] = value.to("cpu")
    return stats, tokens_per_expert


def compute_expert_load_stats(tokens_per_expert: Tensor) -> dict[str, Tensor]:
    """Per-layer expert-load balance from `[num_moe_layers, num_experts]` token counts, as mean and max over layers.

    `cv` is std/mean of the expert loads, `max_mean` the busiest expert's load over the mean load, and `cold_frac`
    the fraction of experts receiving under 0.1x the mean load (the MiMo-V2.6 definition).
    """
    mean_load = tokens_per_expert.mean(dim=1)
    per_layer = {
        "cv": tokens_per_expert.std(dim=1, correction=0) / mean_load,
        "max_mean": tokens_per_expert.amax(dim=1) / mean_load,
        "cold_frac": (tokens_per_expert < 0.1 * mean_load[:, None]).float().mean(dim=1),
    }
    stats = {}
    for name, values in per_layer.items():
        stats[f"expert_load/{name}/mean"] = values.mean()
        stats[f"expert_load/{name}/max"] = values.max()
    return stats


def get_expert_load_stats(tokens_per_expert: Tensor, group: dist.ProcessGroup) -> dict[str, float]:
    """Sum a step's per-layer expert counts across the group (every rank routes distinct tokens) and compute load stats."""
    if tokens_per_expert.numel() == 0:  # MoE model without prime-rl MoE layers (HF impl)
        return {}
    dist.all_reduce(tokens_per_expert, op=dist.ReduceOp.SUM, group=group)
    return {name: value.item() for name, value in compute_expert_load_stats(tokens_per_expert).items()}


def get_model(
    config: ModelConfig, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.bfloat16
) -> nn.Module:
    logger = get_logger()
    logger.debug(
        f"Loading model config (name={config.name}, attn={config.attn}, trust_remote_code={config.trust_remote_code})"
    )

    is_vlm_training = config.vlm is not None

    model_config = cast(
        PretrainedConfig,
        AutoConfig.from_pretrained(
            config.name, attn_implementation=config.attn, trust_remote_code=config.trust_remote_code
        ),
    )
    model_config.use_cache = False
    is_vlm_arch = is_vlm_architecture(model_config)

    if is_vlm_training:
        logger.info(f"Detected vision-language model: {config.name}")

    for subconfig_key in getattr(model_config, "sub_configs", {}):
        subconfig = getattr(model_config, subconfig_key, None)
        if subconfig is not None and hasattr(subconfig, "use_cache"):
            subconfig.use_cache = False
    if config.index_cache is not None:
        model_config.use_index_cache = True
        model_config.index_topk_freq = config.index_cache.topk_freq
        model_config.index_topk_pattern = config.index_cache.topk_pattern
        # Explicit override supersedes the model's native IndexShare schedule.
        model_config.indexer_types = None
    else:
        # Auto-enable IndexShare from the model's own indexer schedule (e.g. GLM-5.2). The model
        # reads `indexer_types` directly: shared layers reuse cached indices and carry no indexer weights.
        indexer_types = getattr(model_config, "indexer_types", None)
        if indexer_types and any(t == "shared" for t in indexer_types):
            model_config.use_index_cache = True
            logger.info(
                f"Auto-enabled IndexShare from indexer_types schedule "
                f"({sum(t == 'full' for t in indexer_types)}/{len(indexer_types)} full layers)"
            )

    # Ensure pad_token_id is set (some models like Qwen3MoE don't have it).
    # In transformers v5, token IDs moved from PretrainedConfig to GenerationConfig.
    if not hasattr(model_config, "pad_token_id") or model_config.pad_token_id is None:
        gen_config = GenerationConfig.from_model_config(model_config)
        # Use `is not None` instead of truthiness: token ID 0 is valid.
        pad_token_id = next(
            (
                v
                for v in [gen_config.pad_token_id, gen_config.eos_token_id, getattr(model_config, "eos_token_id", None)]
                if v is not None
            ),
            None,
        )
        # Some HF configs (e.g. Llama 3.2) set pad_token_id to a list, which
        # crashes both huggingface_hub's strict setter and transformers'
        # GenerationConfig.validate(). Unwrap before assigning.
        if isinstance(pad_token_id, list):
            pad_token_id = pad_token_id[0]
        model_config.pad_token_id = pad_token_id

    # Handle list pad_token_id that was already set on the config (not from our
    # fallback above, but directly in the model's config.json).
    if isinstance(getattr(model_config, "pad_token_id", None), list):
        model_config.pad_token_id = model_config.pad_token_id[0]

    # NOTE: For VLM models, we do NOT propagate dtype to sub_configs.
    # The model should load in its default dtype (bf16) to match vLLM inference.
    # The FSDP MixedPrecisionPolicy handles compute dtype separately.

    logger.debug(f"Loaded model config ({model_config.to_dict()})")

    if config.debug.num_layers is not None:
        # VLM configs nest num_hidden_layers under text_config
        target_config = getattr(model_config, "text_config", model_config)
        num_hidden_layers = min(config.debug.num_layers, target_config.num_hidden_layers)
        logger.warning(
            f"Setting the number of layers to {config.debug.num_layers} in the model config. This means {target_config.num_hidden_layers - num_hidden_layers} layers will not be loaded."
        )
        target_config.num_hidden_layers = num_hidden_layers

    custom_vlm_cls = get_custom_vlm_cls(model_config) if is_vlm_arch else None
    if custom_vlm_cls is None and not supports_custom_impl(model_config):
        raise ValueError(
            f"{model_config.model_type!r} has no PrimeRL model implementation. "
            "The trainer only supports the architectures in prime_rl.trainer.models."
        )

    # Queried here so a misconfigured job dies at setup rather than at the first forward.
    if config.cp > 1:
        cp_model_cls = custom_vlm_cls or get_custom_causal_lm_cls(model_config)
        support = cp_model_cls.cp_support(model_config)
        if config.cp_style not in support.styles:
            supported = f"supported styles: {sorted(support.styles)}" if support.styles else "set cp=1"
            raise ValueError(
                f"{model_config.model_type!r} does not support cp_style={config.cp_style!r} "
                f"({support.reason}); {supported}."
            )

    if config.vlm is not None and custom_vlm_cls is None:
        raise ValueError(
            f"VLM training requires a registered PrimeRL VLM implementation; {model_config.model_type!r} has none."
        )

    with device:
        model_cls = custom_vlm_cls or AutoModelForCausalLMPrimeRL

        load_model_start_time = time.perf_counter()
        if device == torch.device("meta"):
            logger.info(f"Loading model {config.name} using {model_cls.__name__} to meta device")
            model = model_cls.from_config(model_config, trust_remote_code=config.trust_remote_code, dtype=dtype)
        else:
            logger.info(f"Loading model {config.name} using {model_cls.__name__} to CPU")
            model = model_cls.from_pretrained(
                pretrained_model_name_or_path=config.name,
                config=model_config,
                trust_remote_code=config.trust_remote_code,
                dtype=dtype,
            )
        logger.debug(f"Loaded model {config.name} in {format_time(time.perf_counter() - load_model_start_time)}")

    assert model.lm_head.weight.dtype == dtype, (
        f"LM head dtype wasnt loaded correctly {model.lm_head.weight.dtype} != {dtype}"
    )
    return model


def setup_tokenizer(config: TokenizerConfig) -> PreTrainedTokenizer:
    logger = get_logger()
    tokenizer = AutoTokenizer.from_pretrained(config.name, trust_remote_code=config.trust_remote_code)
    if config.chat_template is not None:
        path = Path(config.chat_template)
        if path.is_file():
            logger.info(f"Loading custom chat template from file: {path}")
            tokenizer.chat_template = path.read_text()
            logger.debug(f"Chat template content:\n{tokenizer.chat_template}")
        else:
            logger.info("Using inline custom chat template")
            tokenizer.chat_template = config.chat_template
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def setup_processor(config: ModelConfig):
    """Load an ``AutoProcessor`` for VLM models. Returns ``None`` for text-only models."""
    from transformers import AutoProcessor

    logger = get_logger()
    try:
        processor = AutoProcessor.from_pretrained(config.name, trust_remote_code=config.trust_remote_code)
    except (ValueError, OSError, KeyError) as e:
        logger.debug(f"No AutoProcessor available for {config.name} ({type(e).__name__}); treating as text-only.")
        return None
    if not (getattr(processor, "image_processor", None) or getattr(processor, "video_processor", None)):
        logger.debug(f"AutoProcessor for {config.name} has no image/video processor; treating as text-only.")
        return None
    logger.info(f"Loaded multimodal processor: {type(processor).__name__}")
    return processor


def _expert_shard_placement_fn(
    experts: nn.Module,
    expert_mesh_info: FSDPMeshInfo,
    shard_placement_fn: Callable[[nn.Parameter], Shard | None],
) -> Callable[[nn.Parameter], ShardPlacementResult | Shard | None]:
    """Shards the expert parameters over the EP-complement mesh and everything else over the block's mesh."""
    expert_params = set(experts.parameters())

    def placement_fn(parameter: nn.Parameter) -> ShardPlacementResult | Shard | None:
        placement = shard_placement_fn(parameter)
        if parameter in expert_params:
            return ShardPlacementResult(placement=placement, mesh_info=expert_mesh_info)
        return placement

    return placement_fn


def setup_fsdp(model: nn.Module, config: ModelConfig, parallel_dims: ParallelDims):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=DTYPE_MAP[config.reduce_dtype])
    offload_policy: OffloadPolicy = CPUOffloadPolicy(pin_memory=True) if config.fsdp_cpu_offload else OffloadPolicy()

    fused_shard_placement_fn = get_fsdp_shard_placement_fn(model) if config.fusions.shard_fused_on_dim1 else None
    hsdp_mesh = parallel_dims.get_mesh("hsdp")
    shard_size = parallel_dims.get_mesh("dp_shard_cp").size()

    def shard_placement_fn(parameter: nn.Parameter) -> Shard | None:
        # Qwen's single-row shared-expert gates need equal shards for Muon's all-to-all.
        if parameter.ndim == 2 and parameter.shape[0] == 1 and parameter.shape[1] % shard_size == 0:
            return Shard(1)
        return fused_shard_placement_fn(parameter) if fused_shard_placement_fn is not None else None

    fsdp_config = {
        "mp_policy": mp_policy,
        "offload_policy": offload_policy,
        "reshard_after_forward": config.reshard_after_forward,
        "shard_placement_fn": shard_placement_fn,
    }

    expert_mesh_info: FSDPMeshInfo | None = None
    if parallel_dims.ep_enabled:
        dp_mod_ep_mesh_dim_names = []
        if parallel_dims.dp_replicate_enabled:
            dp_mod_ep_mesh_dim_names.append("dp_replicate")
        dp_mod_ep_mesh_dim_names.append("dp_shard_mod_ep")

        dp_mod_ep_mesh = parallel_dims.world_mesh[tuple(dp_mod_ep_mesh_dim_names)]
        expert_mesh_info = _get_mesh_info(dp_mod_ep_mesh)
        assert isinstance(expert_mesh_info, FSDPMeshInfo)

    is_vlm_training = config.vlm is not None
    if is_vlm_training:
        vision_encoder = get_vision_encoder(model, override=config.vlm.vision_encoder_attr)
        if vision_encoder is None:
            raise ValueError(f"VLM model {config.name} has no recognized vision encoder")

        fully_shard(vision_encoder, mesh=hsdp_mesh, **fsdp_config)
        get_logger().info(f"Applied FSDP to vision encoder (frozen={config.vlm.freeze_vision_encoder})")

    language_model = get_language_model(model, override=config.vlm.language_model_attr if is_vlm_training else None)
    transformer_layers = language_model.layers

    fullgraph = config.compile is not None and config.compile.fullgraph
    for transformer_block in transformer_layers:
        for module in transformer_block.modules():
            if isinstance(module, NGramEmbedding) and parallel_dims.get_mesh("head").size() > 1:
                embedding = module.ngram_embedding
                dp_mod_head_mesh = (
                    parallel_dims.world_mesh["dp_replicate", "dp_shard_mod_head"]
                    if parallel_dims.dp_replicate_enabled
                    else parallel_dims.get_mesh("dp_shard_mod_head")
                )
                parallelize_module(embedding, parallel_dims.get_mesh("head"), EmbeddingParallel())
                fully_shard(embedding, mesh=dp_mod_head_mesh, **fsdp_config)
                embedding.set_gradient_divide_factor(parallel_dims.fsdp_gradient_divide_factor)

        block_mlp = getattr(transformer_block, "mlp", None)
        block_fsdp_config = fsdp_config
        if expert_mesh_info is not None and isinstance(block_mlp, MoE):
            # The experts shard over the EP-complement mesh but stay in the block's FSDP unit: a nested
            # unit would put dynamo-disabled FSDP hooks inside the compiled block and break fullgraph.
            block_fsdp_config = {
                **fsdp_config,
                "shard_placement_fn": _expert_shard_placement_fn(
                    block_mlp.experts, expert_mesh_info, shard_placement_fn
                ),
            }

        if config.moe_router_dtype == "float32" and isinstance(block_mlp, MoE):
            if fullgraph:
                raise ValueError(
                    "model.compile.fullgraph=true requires model.moe_router_dtype='bfloat16': the fp32 router is "
                    "its own FSDP unit inside the compiled block, and dynamo cannot trace FSDP hooks."
                )
            # Own FSDP unit with an fp32 policy so the gate weight is not cast to
            # bf16 for forward and its gradients reduce in fp32.
            fully_shard(
                block_mlp.router,
                mesh=hsdp_mesh,
                mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32),
                offload_policy=offload_policy,
                reshard_after_forward=config.reshard_after_forward,
                shard_placement_fn=shard_placement_fn,
            )
            # Keep the router reduction from waiting for the expert reduction's input buffer.
            block_mlp.router.set_reduce_scatter_max_input_buffers(2)

        fully_shard(
            transformer_block,
            mesh=hsdp_mesh,
            **block_fsdp_config,
        )
        if expert_mesh_info is not None and isinstance(block_mlp, MoE):
            # Expert gradients reduce over the EP-complement mesh only, so they divide by the full
            # data-parallel size. That is already the dense parameters' default divisor.
            transformer_block.set_gradient_divide_factor(parallel_dims.fsdp_gradient_divide_factor)

    shard_norm_and_lm_head = hasattr(model, "config") and not model.config.tie_word_embeddings
    final_module = (
        getattr(language_model, "norm", None)
        or getattr(language_model, "norm_f", None)
        or getattr(language_model, "hyper_connection_mixer", None)
    )

    if shard_norm_and_lm_head:
        # This optimization breaks weight tying
        embed_module = getattr(language_model, "embed_tokens", None) or getattr(language_model, "embeddings", None)
        fully_shard(
            embed_module,
            mesh=hsdp_mesh,
            **fsdp_config,
        )
        fully_shard(
            [model.lm_head, final_module],
            mesh=hsdp_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=False,
            shard_placement_fn=shard_placement_fn,
        )
    else:
        get_logger().warning("Model uses tied word embeddings, so skipping the last-layer no-reshard optimization.")

    fully_shard(
        model,
        mesh=hsdp_mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        reshard_after_forward=config.reshard_after_forward,
        shard_placement_fn=shard_placement_fn,
    )

    if not parallel_dims.ep_enabled:
        return

    # if EP is enabled, d2h syncs in the dispatch/combine can interfere with FSDP prefetch, that's why we set it below manually
    # the rest of the function handles only that

    transformer_blocks = list(language_model.layers)
    next_transformer_blocks = transformer_blocks[1:] + [None]

    embed_module = getattr(language_model, "embed_tokens", None) or getattr(language_model, "embeddings", None)
    if embed_module is not None and len(language_model.layers) > 0:
        if shard_norm_and_lm_head:
            embed_module.set_modules_to_forward_prefetch([transformer_blocks[0]])

    for transformer_block, next_transformer_block in zip(transformer_blocks, next_transformer_blocks):
        if next_transformer_block is not None:
            next_mlp = getattr(next_transformer_block, "mlp", None)
            prefetch_modules = [next_transformer_block]
            if isinstance(next_mlp, MoE) and isinstance(next_mlp.router, FSDPModule):
                prefetch_modules.append(next_mlp.router)
            transformer_block.set_modules_to_forward_prefetch(prefetch_modules)
        elif final_module is not None and model.lm_head is not None:
            if shard_norm_and_lm_head:
                transformer_block.set_modules_to_forward_prefetch([final_module, model.lm_head])

    # backward
    reversed_transformer_blocks = list(reversed(language_model.layers))
    prev_transformer_blocks = reversed_transformer_blocks[1:] + [None]

    if final_module is not None and model.lm_head is not None and len(language_model.layers) > 0:
        last_transformer_block = reversed_transformer_blocks[0]
        prefetch_modules = [last_transformer_block]
        last_mlp = getattr(last_transformer_block, "mlp", None)
        if isinstance(last_mlp, MoE) and isinstance(last_mlp.router, FSDPModule):
            prefetch_modules.append(last_mlp.router)

        if shard_norm_and_lm_head:
            model.lm_head.set_modules_to_backward_prefetch(prefetch_modules)
        else:
            model.set_modules_to_backward_prefetch(prefetch_modules)

    for transformer_block, prev_transformer_block in zip(reversed_transformer_blocks, prev_transformer_blocks):
        if prev_transformer_block is not None:
            prev_mlp = getattr(prev_transformer_block, "mlp", None)
            prefetch_modules = [prev_transformer_block]
            if isinstance(prev_mlp, MoE) and isinstance(prev_mlp.router, FSDPModule):
                prefetch_modules.append(prev_mlp.router)
            transformer_block.set_modules_to_backward_prefetch(prefetch_modules)
        elif embed_module is not None:
            if shard_norm_and_lm_head:
                transformer_block.set_modules_to_backward_prefetch([embed_module])


def load_dcp_from_hf(model: nn.Module, config: ModelConfig, parallel_dims: ParallelDims):
    device = "cpu" if config.fsdp_cpu_offload else "cuda"
    model.to_empty(device=device)
    torch.distributed.barrier()

    # Must run before any weight loading: reinit can zero persistent buffers that ship in checkpoints
    model.init_buffers_post_meta()

    logger = get_logger()
    if config.debug.random_init:
        logger.warning("Randomly initializing model. Skipping loading weights from HF.")
        _move_buffers_to_cuda(model, config)
        return

    if not Path(config.name).exists():
        snapshot_path = Path(snapshot_download(repo_id=config.name, repo_type="model"))
    else:
        logger.info(
            f"Loading model weights from path {config.name}, skipping snapshot download. If this is not expected, please remove the directory {config.name} and run again"
        )
        snapshot_path = Path(config.name)

    # Dynamically convert between different weight formats if needed.
    # All ranks read just the key names (cheap) to determine the path independently.
    # Only master loads the full state dict when conversion is actually needed.
    source_path = snapshot_path
    convert_dir = config.conversion_dir or source_path
    snapshot_keys = dict.fromkeys(load_state_dict_keys(source_path))
    model_keys = dict.fromkeys(model.state_dict().keys())

    if source_path.name == "prime" and not (source_path / ".prime-v1").is_file():
        raise RuntimeError(f"PrimeRL conversion cache {source_path} is missing the required .prime-v1 marker")

    snapshot_is_hf = model.is_hf_state_dict(snapshot_keys)
    snapshot_is_prime = model.is_prime_state_dict(snapshot_keys)

    if snapshot_is_hf and not snapshot_is_prime and model.is_prime_state_dict(model_keys):
        logger.warning(
            "Found HF weight format in snapshot state dict and PrimeRL weight format in model state dict. Trying to auto-convert..."
        )
        snapshot_path = convert_dir / "prime"
        if not snapshot_path.exists() and get_world().is_master:
            logger.debug(
                f"Converting snapshot state dict to PrimeRL format and saving to {snapshot_path} on master rank. This is a one-time operation."
            )
            snapshot_state_dict = load_state_dict(source_path)
            model.convert_to_prime(snapshot_state_dict)
            save_state_dict(snapshot_state_dict, snapshot_path)
            (snapshot_path / ".prime-v1").touch()
            del snapshot_state_dict

    elif snapshot_is_prime and not snapshot_is_hf and model.is_hf_state_dict(model_keys):
        logger.warning(
            "Found PrimeRL weight format in snapshot state dict and HF weight format in model state dict. Trying to auto-convert..."
        )
        snapshot_path = convert_dir / "hf"
        if not snapshot_path.exists() and get_world().is_master:
            logger.debug(
                f"Converting snapshot state dict to HF format and saving to {snapshot_path} on master rank. This is a one-time operation."
            )
            snapshot_state_dict = load_state_dict(source_path)
            model.convert_to_hf(snapshot_state_dict)
            save_state_dict(snapshot_state_dict, snapshot_path)
            del snapshot_state_dict

    # All ranks wait for master rank to finish conversion
    torch.distributed.barrier()
    if snapshot_path.name == "prime" and not (snapshot_path / ".prime-v1").is_file():
        raise RuntimeError(f"PrimeRL conversion cache {snapshot_path} is missing the required .prime-v1 marker")

    logger.info(f"Loading weights using HF DCP from {snapshot_path}")
    load_dcp_start_time = time.perf_counter()
    state_dict = model.state_dict()
    state_dict = strip_lora_from_state_dict(state_dict)
    if model.config.tie_word_embeddings:
        state_dict.pop("lm_head.weight")
    dcp_load(
        state_dict,
        storage_reader=HuggingFaceStorageReader(path=snapshot_path.as_posix()),
    )
    write_back_loaded_packed_parameters(model, state_dict)
    _move_buffers_to_cuda(model, config)

    lora_modules = [m for m in model.modules() if hasattr(m, "_init_lora_parameters")]
    if lora_modules:
        generator: torch.Generator | None = None
        if parallel_dims.dp_replicate_enabled:
            # Synchronize LoRA initialization across dp_replicate ranks by broadcasting a seed
            dp_replicate_mesh = parallel_dims.world_mesh["dp_replicate"]
            seed_tensor = torch.empty(1, dtype=torch.long, device="cuda")
            if dp_replicate_mesh.get_local_rank() == 0:
                seed_tensor.random_()
            torch.distributed.broadcast(seed_tensor, src=0, group=dp_replicate_mesh.get_group())
            generator = torch.Generator(device="cuda").manual_seed(seed_tensor.item())
        for module in lora_modules:
            module._init_lora_parameters(generator)
    logger.debug(f"Loaded weights using HF DCP in {format_time(time.perf_counter() - load_dcp_start_time)}")


def reshard_module(model: nn.Module):
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def apply_ac(model: nn.Module, ac_config: ActivationCheckpointConfig):
    language_model = get_language_model(model)
    wrap_block = get_activation_checkpoint_wrapper(ac_config)
    checkpointed_layers = 0

    for layer_id, (layer_name, transformer_block) in enumerate(language_model.layers.named_children()):
        if layer_id % ac_config.freq != 0:
            continue

        language_model.layers.register_module(layer_name, wrap_block(transformer_block))
        checkpointed_layers += 1

    get_logger().info(
        f"Applied {ac_config.mode} activation checkpointing to {checkpointed_layers} layers (freq={ac_config.freq})"
    )


def apply_compile(model: nn.Module, compile_config: CompileConfig):
    torch._dynamo.config.capture_scalar_outputs = True
    language_model = get_language_model(model)
    for layer_id in range(len(language_model.layers)):
        # Doing it in-place avoids mangled fqn which can break checkpoint loading
        language_model.layers[layer_id].compile(fullgraph=compile_config.fullgraph, mode=compile_config.mode)
    get_logger().info(
        f"Compiled {len(language_model.layers)} layers (fullgraph={compile_config.fullgraph}, mode={compile_config.mode})"
    )


def apply_quantization(model: nn.Module, config: ModelConfig) -> None:
    """Swap dense linear modules to the configured low-precision path.

    Routed-expert compute is configured independently by ``model.moe.compute``.
    """
    quant = config.quantization
    if quant is None:
        return

    if isinstance(quant, FP8Config):
        replace_linear_with_fp8_blockwise_linear(model, ignore_modules=quant.ignore_patterns)
    elif isinstance(quant, MXFP8Config):
        capability = torch.cuda.get_device_capability()
        if capability[0] < 10:
            raise ValueError(
                f"MXFP8 quantization requires Blackwell (SM100+), but device is SM{capability[0]}{capability[1]}."
            )
        replace_linear_with_mxfp8_linear(model, recipe=quant.recipe, ignore_modules=quant.ignore_patterns)


def configure_trainable_parameters(model: nn.Module, config: ModelConfig) -> nn.Module | None:
    """Apply LoRA and identify any vision encoder that must remain frozen."""
    frozen_vision_encoder = None
    if config.vlm is not None and config.vlm.freeze_vision_encoder:
        frozen_vision_encoder = get_vision_encoder(model, override=config.vlm.vision_encoder_attr)
    elif config.vlm is None:
        frozen_vision_encoder = get_vision_encoder(model)
        if frozen_vision_encoder is not None:
            get_logger().info("Training a VLM checkpoint on text-only data; freezing the vision encoder")

    if config.lora is not None:
        apply_lora_to_model(model, config.lora)
    return frozen_vision_encoder


def _move_buffers_to_cuda(model: nn.Module, config: ModelConfig) -> None:
    """FSDP CPU offloading only manages parameters, not buffers. Move buffers to CUDA."""
    if not config.fsdp_cpu_offload:
        return
    for _, buffer in model.named_buffers():
        if buffer.device.type == "cpu":
            buffer.data = buffer.data.to("cuda")


def _reset_runtime_moe_buffers(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, MoE) and module.tokens_per_expert.device.type != "meta":
            module.tokens_per_expert.zero_()
            module.routing_confidence_sum.zero_()


def _validate_flash_attn_4_installed() -> None:
    """Validate that flash-attn-cute is installed and not overwritten by flash-attn.

    Both flash-attn and flash-attn-cute ship a `flash_attn.cute` sub-package.
    When both extras are installed, the older stub from flash-attn can shadow the
    real implementation.  We detect this by checking the line count of the interface
    module (the real one is >1000 lines).
    """
    import flash_attn.cute.interface as fa4_interface

    with open(fa4_interface.__file__, "r") as f:
        num_lines = sum(1 for _ in f)

    if num_lines < 1000:
        raise ValueError(
            "flash-attn-cute has probably been overwritten by flash-attn, "
            "run `scripts/fix-flash-attn-cute.sh` to fix this behaviour."
        )


def resolve_auto_attn(config: ModelConfig) -> None:
    """Resolve ``attn='auto'`` to a concrete flash attention implementation based on GPU architecture.

    FA4 on datacenter Blackwell (SM10x/SM11x, e.g. B200, B300), FA3 on Hopper (SM90), FA2
    otherwise. FA4 is built around the tcgen05/TMEM instructions of the SM10x/SM11x families.
    Workstation Blackwell (SM12x, e.g. RTX PRO 6000) lacks them, so it gets FA2.
    """
    if config.attn != "auto":
        return
    major, minor = torch.cuda.get_device_capability()
    if major in (10, 11):
        resolved = "flash_attention_4"
    elif major == 9:
        resolved = "flash_attention_3"
    else:
        resolved = "flash_attention_2"
    logger = get_logger()
    logger.info(f"Auto-resolved attn='auto' to '{resolved}' (SM{major}{minor})")
    config.attn = resolved


def setup_model(
    config: ModelConfig,
    parallel_dims: ParallelDims,
    loading_from_checkpoint_later: bool = False,
) -> nn.Module:
    resolve_auto_attn(config)

    if config.attn == "flash_attention_3" and not is_flash_attn_3_available():
        raise ValueError(
            "Flash attention 3 is only supported if the flash_attn_3 package is installed. Install with `uv pip install 'flash-attn-3 @ git+https://github.com/Dao-AILab/flash-attention.git@main#subdirectory=hopper' --no-build-isolation`"
        )

    if config.attn == "flash_attention_4":
        _validate_flash_attn_4_installed()

    logger = get_logger()

    # Build on the meta device; weights are materialized after FSDP sharding.
    model = get_model(config, device=torch.device("meta"), dtype=DTYPE_MAP[config.optimization_dtype])

    if config.fusions.enabled and config.lora is not None:
        logger.warning("Skipping runtime model fusions because LoRA targets the unfused projections")
    elif config.fusions.enabled:
        applied = apply_model_fusions(model, config.fusions.enabled)
        logger.info(f"Applied runtime model fusions: {applied}")

    lm_head_chunk_size: int | None = None
    if isinstance(config.fused_lm_head_token_chunk_size, int):
        lm_head_chunk_size = config.fused_lm_head_token_chunk_size

    inject_prime_lm_head(model, chunk_size=lm_head_chunk_size)

    apply_quantization(model, config)

    frozen_vision_encoder = configure_trainable_parameters(model, config)

    if config.freeze_moe_router:
        freeze_moe_router(model)

    if config.moe_router_dtype == "float32":
        apply_fp32_moe_router(model)

    # A sparse-attention indexer runs its forward under torch.no_grad(), so it is never
    # trainable. Freeze it so optimizer state stays symmetric across checkpoint save/resume.
    # No-op for models without a sparse indexer.
    freeze_sparse_indexer(model)

    if config.debug.force_balanced_routing:
        apply_force_balanced_routing(model)

    configure_moe_runtime(model, config, parallel_dims)
    if parallel_dims.ep_enabled:
        # EP replaces params with DTensors that default to requires_grad=True,
        # re-freeze base params that LoRA froze earlier.
        if config.lora is not None:
            freeze_all_except_lora_and_specified(model, config.lora)

    if frozen_vision_encoder is not None:
        freeze_vision_encoder(
            model,
            override_attr=config.vlm.vision_encoder_attr if config.vlm is not None else None,
        )

    # the right order is AC -> Compile -> FSDP
    if config.ac is not None:
        apply_ac(model, config.ac)
    if config.compile is not None:
        apply_compile(model, config.compile)

    setup_fsdp(model, config, parallel_dims)

    if loading_from_checkpoint_later:
        logger.warning(
            "Skipping loading weights. Initializing an empty model on device, loading from checkpoint later."
        )
        device = "cpu" if config.fsdp_cpu_offload else "cuda"
        model.to_empty(device=device)
        torch.distributed.barrier()
        model.init_buffers_post_meta()
        _move_buffers_to_cuda(model, config)
    else:
        load_dcp_from_hf(model, config, parallel_dims)

    _reset_runtime_moe_buffers(model)
    return model


def forward(
    model: nn.Module,
    input_ids: Int[Tensor, "batch seq"],
    position_ids: Int[Tensor, "batch seq"],
    *,
    seq_lens: Int[Tensor, "segments"],
    labels: Int[Tensor, "batch seq"] | None = None,
    temperature: Tensor | None = None,
    routed_experts: Int[Tensor, "batch seq layers topk"] | None = None,
    sampling_mask: Int[Tensor, "batch seq mask"] | None = None,
    mm_kwargs: dict[str, Tensor] | None = None,
    mm_forward_policy: ForwardPolicy | None = None,
    mm_token_type_ids: Int[Tensor, "batch seq"] | None = None,
    # True when seq_lens holds the full pre-CP-shard document boundaries
    # (kept global because documents can straddle the shard cut).
    seq_lens_are_pre_shard: bool = False,
) -> PrimeLmOutput:
    kwargs = {
        "input_ids": input_ids,
        "labels": labels,
        "temperature": temperature,
        "sampling_mask": sampling_mask,
    }

    if mm_kwargs:
        kwargs.update(mm_kwargs)
        if mm_token_type_ids is not None:
            kwargs["mm_token_type_ids"] = mm_token_type_ids
        # SFT still uses its existing eager processor path and does not provide
        # an adapter policy yet, so preserve its current kwargs-based behavior.
        policy = mm_forward_policy or ForwardPolicy(pass_position_ids="image_grid_thw" not in mm_kwargs)
        if policy.requires_mm_token_type_ids and mm_token_type_ids is None:
            raise ValueError("Multimodal forward policy requires mm_token_type_ids")
        if policy.pass_position_ids:
            kwargs["position_ids"] = position_ids
    else:
        kwargs["position_ids"] = position_ids

    kwargs["seq_lens"] = seq_lens
    kwargs["seq_lens_are_pre_shard"] = seq_lens_are_pre_shard

    if routed_experts is not None:
        kwargs["routed_experts"] = routed_experts

    return cast_float_and_contiguous(model(**kwargs))
