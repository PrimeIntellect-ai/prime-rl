import logging
import time
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from beartype import beartype as typechecker
from huggingface_hub import snapshot_download
from jaxtyping import Int, jaxtyped
# from liger_kernel.transformers import AutoLigerKernelForCausalLM  # Removed: incompatible with transformers 5.0
from torch import Tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, OffloadPolicy, fully_shard
from torch.distributed.tensor.parallel import parallelize_module
from torchtitan.distributed.expert_parallel import ExpertParallel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.import_utils import is_flash_attn_3_available

from prime_rl.trainer.config import ActivationCheckpointConfig, CompileConfig, ModelConfig, TokenizerConfig
from prime_rl.trainer.lora import apply_lora_to_model, strip_lora_from_state_dict
from prime_rl.trainer.models import (
    AutoModelForCausalLMPrimeRL,
    PreTrainedModelPrimeRL,
    PrimeLmOutput,
    cast_float_and_contiguous,
    supports_custom_impl,
)
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.layers.moe import MoE
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.weights import (
    load_state_dict,
    save_state_dict,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.tensor_hashing import get_module_signature

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


def is_tt_moe_model(model: nn.Module) -> bool:
    return hasattr(model.config, "num_experts") or hasattr(model.config, "n_routed_experts")


def _is_nemotron_h(model: nn.Module) -> bool:
    archs = getattr(getattr(model, 'config', None), 'architectures', []) or []
    return 'NemotronH' in str(archs)


def patch_model_compat(model: nn.Module) -> None:
    """Add model.model alias for NemotronH which uses backbone."""
    if _is_nemotron_h(model):
        model.model = model.backbone
        model.__class__._supports_flash_attn_2 = True
        model.__class__._supports_sdpa = True


def get_inner_model(model: nn.Module) -> nn.Module:
    if _is_nemotron_h(model):
        return model.backbone
    return model.model


def get_embed_tokens(model: nn.Module) -> nn.Module:
    inner = get_inner_model(model)
    if _is_nemotron_h(model):
        return inner.embeddings
    return inner.embed_tokens


def get_final_norm(model: nn.Module) -> nn.Module:
    inner = get_inner_model(model)
    if _is_nemotron_h(model):
        return inner.norm_f
    return inner.norm


def get_layers(model: nn.Module) -> nn.ModuleList:
    return get_inner_model(model).layers


def get_ffn(block: nn.Module) -> nn.Module | None:
    return getattr(block, 'mlp', None)


def get_load_balance_stats(
    model: nn.Module, reset_stats: bool = True, try_to_avoid_padding_experts: bool = True
) -> dict[str, Tensor | None]:
    per_layer_max_vio = []
    for transformer_block in get_layers(model):
        ffn = get_ffn(transformer_block)
        # This is necessary for models that have mixed dense layers
        if ffn is None or not hasattr(ffn, "tokens_per_expert"):
            continue
        tokens_per_expert: torch.Tensor = ffn.tokens_per_expert
        if try_to_avoid_padding_experts:
            tokens_per_expert = tokens_per_expert.sort(dim=0, descending=True).values[
                ffn.router.top_k :
            ]
        balanced_load = tokens_per_expert.mean()
        max_vio = (tokens_per_expert.max() - balanced_load) / balanced_load
        per_layer_max_vio.append(max_vio.item())
        if reset_stats:
            ffn.tokens_per_expert.zero_()
    if len(per_layer_max_vio) == 0:
        return {"max_vio": None}
    return {"max_vio": torch.tensor(per_layer_max_vio, device=torch.device("cuda"))}


def _patch_model_class_for_flash_attn(model_config: PretrainedConfig, model_path: str) -> None:
    """Patch NemotronH to declare flash attention support."""
    arch = getattr(model_config, 'architectures', [None])[0]
    if not (arch and 'NemotronH' in arch):
        return
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        for cls_name in ('modeling_nemotron_h.NemotronHForCausalLM', 'modeling_nemotron_h.NemotronHModel'):
            cls = get_class_from_dynamic_module(cls_name, model_path)
            cls._supports_flash_attn_2 = cls._supports_sdpa = True
    except Exception:
        pass


def get_model(
    config: ModelConfig, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.bfloat16
) -> nn.Module:
    logger = get_logger()
    logger.info(
        f"Loading model config (name={config.name}, attn={config.attn}, trust_remote_code={config.trust_remote_code})"
    )
    model_config = cast(
        PretrainedConfig,
        AutoConfig.from_pretrained(
            config.name, attn_implementation=config.attn, trust_remote_code=config.trust_remote_code
        ),
    )

    # Patch model class for flash attention support before loading
    if config.attn == 'flash_attention_2':
        _patch_model_class_for_flash_attn(model_config, config.name)
    model_config.use_cache = False
    model_config.use_grouped_mm = config.moe_use_grouped_mm
    # Disable rescale_prenorm_residual to prevent HF from reinitializing out_proj weights after loading
    if getattr(model_config, 'rescale_prenorm_residual', False):
        model_config.rescale_prenorm_residual = False
    logger.debug(f"Loaded model config ({model_config.to_dict()})")

    if config.debug.num_layers is not None:
        num_hidden_layers = min(config.debug.num_layers, model_config.num_hidden_layers)
        logger.warning(
            f"Setting the number of layers to {config.debug.num_layers} in the model config. This means {model_config.num_hidden_layers - num_hidden_layers} layers will not be loaded."
        )
        model_config.num_hidden_layers = num_hidden_layers

    # Determine the implementation to use
    if config.impl == "auto":
        impl_to_use = "custom" if supports_custom_impl(model_config) else "hf"
        logger.info(
            f"Auto-selected implementation: {impl_to_use} (custom implementation {'supported' if supports_custom_impl(model_config) else 'not supported'})"
        )
    else:
        impl_to_use = config.impl

    with device:
        match impl_to_use:
            case "hf":
                model_cls = AutoModelForCausalLM
            case "liger_kernel":
                logger.warning("liger_kernel impl requested but disabled due to transformers 5.0 incompatibility, falling back to hf")
                model_cls = AutoModelForCausalLM
            case "custom":
                model_cls = AutoModelForCausalLMPrimeRL

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
        logger.debug(f"Loaded model {config.name} in {time.perf_counter() - load_model_start_time:.2f} seconds")

    assert model.lm_head.weight.dtype == dtype, (
        f"LM head dtype wasnt loaded correctly {model.lm_head.weight.dtype} != {dtype}"
    )
    return model


def setup_tokenizer(config: TokenizerConfig) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config.name, trust_remote_code=config.trust_remote_code)
    if config.chat_template is not None:
        tokenizer.chat_template = config.chat_template
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def setup_fsdp(model: nn.Module, config: ModelConfig, parallel_dims: ParallelDims):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=DTYPE_MAP[config.reduce_dtype])
    offload_policy: OffloadPolicy = CPUOffloadPolicy(pin_memory=True) if config.fsdp_cpu_offload else OffloadPolicy()

    fsdp_config = {
        "mp_policy": mp_policy,
        "offload_policy": offload_policy,
        "reshard_after_forward": config.reshard_after_forward,
    }

    if config.dp_replicate > 1:
        hsdp_mesh = parallel_dims.world_mesh["dp_replicate", "dp_shard_cp"]
    else:
        hsdp_mesh = parallel_dims.world_mesh["dp_shard_cp"]

    dp_mod_ep_mesh: DeviceMesh | None = None
    if parallel_dims.ep_enabled:
        dp_mod_ep_mesh_dim_names = []
        if parallel_dims.dp_replicate_enabled:
            dp_mod_ep_mesh_dim_names.append("dp_replicate")
        dp_mod_ep_mesh_dim_names.append("dp_shard_mod_ep")

        dp_mod_ep_mesh = parallel_dims.world_mesh[tuple(dp_mod_ep_mesh_dim_names)]

    layers = get_layers(model)
    for transformer_block in layers:
        ffn = get_ffn(transformer_block)
        if parallel_dims.ep_enabled and ffn is not None and isinstance(ffn, MoE):
            fully_shard(ffn.experts, mesh=dp_mod_ep_mesh, **fsdp_config)

            ffn.experts.set_gradient_divide_factor(parallel_dims.fsdp_gradient_divide_factor)

        fully_shard(
            transformer_block,
            mesh=hsdp_mesh,
            **fsdp_config,
        )

    shard_norm_and_lm_head = hasattr(model, "config") and not model.config.tie_word_embeddings
    embed_tokens = get_embed_tokens(model)
    final_norm = get_final_norm(model)

    if shard_norm_and_lm_head:
        # This optimization breaks weight tying
        fully_shard(
            embed_tokens,
            mesh=hsdp_mesh,
            **fsdp_config,
        )
        fully_shard(
            [model.lm_head, final_norm],
            mesh=hsdp_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=False,
        )
    else:
        get_logger().warning("Model uses tied word embeddings, so skipping the last-layer no-reshard optimization.")

    fully_shard(
        model,
        mesh=hsdp_mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        reshard_after_forward=config.reshard_after_forward,
    )

    if not parallel_dims.ep_enabled:
        return

    # if EP is enabled, d2h syncs in the dispatch/combine can interfere with FSDP prefetch, that's why we set it below manually
    # the rest of the function handles only that

    transformer_blocks = list(layers)
    next_transformer_blocks = transformer_blocks[1:] + [None]

    if embed_tokens is not None and len(layers) > 0:
        if shard_norm_and_lm_head:
            embed_tokens.set_modules_to_forward_prefetch([transformer_blocks[0]])

    for transformer_block, next_transformer_block in zip(transformer_blocks, next_transformer_blocks):
        if next_transformer_block is not None:
            next_ffn = get_ffn(next_transformer_block)
            if next_ffn is not None and isinstance(next_ffn, MoE):
                transformer_block.set_modules_to_forward_prefetch(
                    [next_transformer_block, next_ffn.experts]
                )
            else:
                transformer_block.set_modules_to_forward_prefetch([next_transformer_block])
        elif final_norm is not None and model.lm_head is not None:
            if shard_norm_and_lm_head:
                transformer_block.set_modules_to_forward_prefetch([final_norm, model.lm_head])

    # backward
    reversed_transformer_blocks = list(reversed(layers))
    prev_transformer_blocks = reversed_transformer_blocks[1:] + [None]

    if final_norm is not None and model.lm_head is not None and len(layers) > 0:
        if shard_norm_and_lm_head:
            model.lm_head.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])
        else:
            model.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])

    for transformer_block, prev_transformer_block in zip(reversed_transformer_blocks, prev_transformer_blocks):
        if prev_transformer_block is not None:
            prev_ffn = get_ffn(prev_transformer_block)
            if prev_ffn is not None and isinstance(prev_ffn, MoE):
                transformer_block.set_modules_to_backward_prefetch(
                    [prev_transformer_block, prev_ffn.experts]
                )
            else:
                transformer_block.set_modules_to_backward_prefetch([prev_transformer_block])
        elif embed_tokens is not None:
            if shard_norm_and_lm_head:
                transformer_block.set_modules_to_backward_prefetch([embed_tokens])


def load_dcp_from_hf(model: nn.Module, config: ModelConfig, parallel_dims: ParallelDims):
    device = "cpu" if config.fsdp_cpu_offload else "cuda"
    model.to_empty(device=device)
    torch.distributed.barrier()

    def _init_buffers_post_meta():
        if isinstance(model, PreTrainedModelPrimeRL):
            model.init_buffers_post_meta()
        else:
            fix_model_post_empty(model)

    logger = get_logger()
    if config.debug.random_init:
        logger.warning("Randomly initializing model. Skipping loading weights from HF.")
        _init_buffers_post_meta()
        _move_buffers_to_cuda(model, config)
        return

    if not Path(config.name).exists():
        snapshot_path = Path(snapshot_download(repo_id=config.name, repo_type="model"))
    else:
        logger.info(
            f"Loading model weights from path {config.name}, skipping snapshot download. If this is not expected, please remove the directory {config.name} and run again"
        )
        snapshot_path = Path(config.name)

    # Load the snapshot state
    snapshot_state_dict = load_state_dict(snapshot_path)
    model_state_dict = model.state_dict()

    # Dynamically convert between different weight formats if needed
    if isinstance(model, PreTrainedModelPrimeRL):
        if model.is_hf_state_dict(snapshot_state_dict) and model.is_prime_state_dict(model_state_dict):
            logger.warning(
                "Found HF weight format in snapshot state dict and PrimeRL weight format in model state dict. Trying to auto-convert..."
            )
            snapshot_path = snapshot_path / "prime"
            if snapshot_path.exists():
                logger.debug(f"Conversion found at {snapshot_path}.")
            else:
                if get_world().is_master:
                    logger.debug(
                        f"Converting snapshot state dict to PrimeRL format and saving to {snapshot_path} on master rank. This is a one-time operation."
                    )
                    model.convert_to_prime(snapshot_state_dict)
                    save_state_dict(snapshot_state_dict, snapshot_path)

        elif model.is_prime_state_dict(snapshot_state_dict) and model.is_hf_state_dict(model_state_dict):
            logger.warning(
                "Found PrimeRL weight format in snapshot state dict and HF weight format in model state dict. Trying to auto-convert..."
            )
            snapshot_path = snapshot_path / "hf"
            if snapshot_path.exists():
                logger.debug(f"Conversion found at {snapshot_path}.")
            else:
                if get_world().is_master:
                    logger.debug(
                        f"Converting snapshot state dict to HF format and saving to {snapshot_path} on master rank. This is a one-time operation."
                    )
                    model.convert_to_hf(snapshot_state_dict)
                    save_state_dict(snapshot_state_dict, snapshot_path)

    # All ranks wait for master rank to finish conversion
    torch.distributed.barrier()

    logger.info(f"Loading weights using HF DCP from {snapshot_path}")
    load_dcp_start_time = time.perf_counter()
    state_dict = model.state_dict()
    state_dict = strip_lora_from_state_dict(state_dict)
    if model.config.tie_word_embeddings:
        del state_dict["lm_head.weight"]
    dcp_load(
        state_dict,
        storage_reader=HuggingFaceStorageReader(path=snapshot_path.as_posix()),
    )
    _init_buffers_post_meta()

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
    logger.debug(f"Loaded weights using HF DCP in {time.perf_counter() - load_dcp_start_time:.2f} seconds")


def can_reinit_empty_buffers(model: nn.Module):
    """Whether the model will be loaded correctly by load_dcp_from_hf.

    The main issue is with anything that is not in the checkpoint.
    This is usually any non-persistent buffers.
    """
    buffer_names = [name for name, _ in model.named_buffers()]

    # TT MoE buffers
    buffer_names = [
        name
        for name in buffer_names
        if not (name.startswith("model.layers.") and name.endswith("mlp.tokens_per_expert"))
    ]
    buffer_names = [
        name for name in buffer_names if not (name.startswith("model.layers.") and name.endswith("mlp.expert_bias"))
    ]
    # HF standard transformer model
    if len(buffer_names) == 1 and buffer_names[0] == "model.rotary_emb.inv_freq":
        return True

    get_logger().warning(f"Model cannot be loaded using meta device because of buffers: {buffer_names}")
    return False


def fix_model_post_empty(model: nn.Module):
    buffer_names = [name for name, _ in model.named_buffers()]
    # HF standard transformer model
    if "model.rotary_emb.inv_freq" in buffer_names:
        rotary_emb = get_inner_model(model).rotary_emb
        inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(rotary_emb.config, rotary_emb.inv_freq.device)
        rotary_emb.inv_freq.copy_(inv_freq)


def reshard_module(model: nn.Module):
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def apply_ac(model: nn.Module, ac_config: ActivationCheckpointConfig):
    layers = get_layers(model)
    for layer_id, (layer_name, transformer_block) in enumerate(layers.named_children()):
        if layer_id % ac_config.freq == 0:
            transformer_block = checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        layers.register_module(layer_name, transformer_block)
    get_logger().info(f"Applied activation checkpointing (freq={ac_config.freq})")


def apply_compile(model: nn.Module, compile_config: CompileConfig):
    torch._dynamo.config.capture_scalar_outputs = True
    layers = get_layers(model)
    for layer_id in range(len(layers)):
        # Doing it in-place avoids mangled fqn which can break checkpoint loading
        layers[layer_id].compile(fullgraph=compile_config.fullgraph)
    get_logger().info(f"Compiled {len(layers)} layers (fullgraph={compile_config.fullgraph})")


def apply_ep(model: nn.Module, parallel_dims: ParallelDims):
    for transformer_block in get_layers(model):
        ffn = get_ffn(transformer_block)
        if ffn is not None and isinstance(ffn, MoE):
            parallelize_module(
                ffn.experts,
                device_mesh=parallel_dims.world_mesh["ep"],
                parallelize_plan=ExpertParallel(),
            )


def _move_buffers_to_cuda(model: nn.Module, config: ModelConfig) -> None:
    """FSDP CPU offloading only manages parameters, not buffers. Move buffers to CUDA."""
    if not config.fsdp_cpu_offload:
        return
    for _, buffer in model.named_buffers():
        if buffer.device.type == "cpu":
            buffer.data = buffer.data.to("cuda")


def setup_model(
    config: ModelConfig, parallel_dims: ParallelDims, loading_from_checkpoint_later: bool = False
) -> nn.Module:
    if config.attn == "flash_attention_3" and not is_flash_attn_3_available():
        raise ValueError(
            "Flash attention 3 is only supported if the flash_attn_3 package is installed. Install with `uv pip install 'flash-attn-3 @ git+https://github.com/Dao-AILab/flash-attention.git@main#subdirectory=hopper' --no-build-isolation`"
        )
    logger = get_logger()

    # 1. We load to meta device by default
    model = get_model(config, device=torch.device("meta"), dtype=DTYPE_MAP[config.optimization_dtype])

    # Patch models with non-standard naming (e.g., NemotronH)
    patch_model_compat(model)

    possible_to_load_to_meta = can_reinit_empty_buffers(model)

    if config.debug.random_init and not possible_to_load_to_meta:
        raise ValueError(
            "It's not possible to load to meta device and random initialize is enabled. Please disable random initialize or use a different model."
        )

    # 1a. We load to CPU if we cannot reinit empty buffers
    if not possible_to_load_to_meta:
        logger.warning("Cannot load model to meta device only, loading to CPU instead.")
        model = get_model(config, device=torch.device("cpu"), dtype=DTYPE_MAP[config.optimization_dtype])
        patch_model_compat(model)

    lm_head_chunk_size: int | None = None
    if isinstance(config.fused_lm_head_chunk_size, int):
        lm_head_chunk_size = config.fused_lm_head_chunk_size

    inject_prime_lm_head(model, chunk_size=lm_head_chunk_size)

    # Apply LoRA before FSDP setup
    if config.lora is not None:
        apply_lora_to_model(model, config.lora)

    if parallel_dims.ep_enabled:
        apply_ep(model, parallel_dims)

    # the right order is AC -> Compile -> FSDP
    if config.ac is not None:
        apply_ac(model, config.ac)
    if config.compile is not None:
        apply_compile(model, config.compile)

    setup_fsdp(model, config, parallel_dims)

    if not possible_to_load_to_meta:
        _move_buffers_to_cuda(model, config)

    # 2. if we can load to meta, we either:
    if possible_to_load_to_meta:
        # - load from checkpoint later if needed
        if loading_from_checkpoint_later:
            logger.warning(
                "Skipping loading weights. Initializing an empty model on device, loading from checkpoint later."
            )
            device = "cpu" if config.fsdp_cpu_offload else "cuda"
            model.to_empty(device=device)
            torch.distributed.barrier()
            if isinstance(model, PreTrainedModelPrimeRL):
                model.init_buffers_post_meta()
            else:
                fix_model_post_empty(model)

            _move_buffers_to_cuda(model, config)
        # - or load from HF with dcp
        else:
            load_dcp_from_hf(model, config, parallel_dims)

    logger.debug(f"Model signature: {get_module_signature(model, compress=True)}")
    return model


def _get_segment_boundaries(position_ids: Tensor) -> list[tuple[int, int]]:
    """Find segment boundaries where position_ids resets to 0."""
    position_ids = position_ids.flatten()
    boundaries = []
    start = 0

    for i in range(1, len(position_ids)):
        if position_ids[i] == 0 and position_ids[i-1] != 0:
            boundaries.append((start, i))
            start = i

    boundaries.append((start, len(position_ids)))
    return boundaries


def _forward_mamba_segmented(
    model: nn.Module,
    input_ids: Tensor,
    position_ids: Tensor,
    labels: Tensor | None = None,
    temperature: Tensor | None = None,
) -> PrimeLmOutput:
    """
    Forward pass for Mamba-hybrid models with packed sequences.

    Splits input at position_id boundaries and processes as batch so each segment
    has independent Mamba state. This matches inference behavior where each sample
    was generated with fresh state.
    """
    boundaries = _get_segment_boundaries(position_ids)

    # If only one segment, use regular forward
    if len(boundaries) == 1:
        out = model(input_ids=input_ids, position_ids=position_ids, labels=labels, temperature=temperature)
        if isinstance(out, dict):
            return cast_float_and_contiguous(out)
        return cast_float_and_contiguous(PrimeLmOutput(logits=out.logits))

    # Split into segments
    segments_input = []
    segments_position = []
    segments_labels = []
    segments_temp = []
    max_len = 0

    for start, end in boundaries:
        seg_len = end - start
        max_len = max(max_len, seg_len)
        segments_input.append(input_ids[0, start:end])
        segments_position.append(position_ids[0, start:end])
        if labels is not None:
            segments_labels.append(labels[0, start:end])
        if temperature is not None and isinstance(temperature, Tensor) and temperature.numel() > 1:
            segments_temp.append(temperature[0, start:end] if temperature.dim() > 1 else temperature[start:end])

    # Pad and batch
    device = input_ids.device
    dtype = input_ids.dtype
    n_segments = len(segments_input)

    batched_input = torch.zeros(n_segments, max_len, dtype=dtype, device=device)
    batched_position = torch.zeros(n_segments, max_len, dtype=dtype, device=device)
    batched_labels = torch.zeros(n_segments, max_len, dtype=dtype, device=device) if labels is not None else None
    batched_temp = None
    if segments_temp:
        batched_temp = torch.ones(n_segments, max_len, dtype=segments_temp[0].dtype, device=device)

    segment_lengths = []
    for i, (seg_in, seg_pos) in enumerate(zip(segments_input, segments_position)):
        seg_len = len(seg_in)
        segment_lengths.append(seg_len)
        batched_input[i, :seg_len] = seg_in
        batched_position[i, :seg_len] = seg_pos
        if batched_labels is not None:
            batched_labels[i, :seg_len] = segments_labels[i]
        if batched_temp is not None:
            batched_temp[i, :seg_len] = segments_temp[i]

    # Forward with batched segments (each has independent Mamba state)
    out = model(
        input_ids=batched_input,
        position_ids=batched_position,
        labels=batched_labels,
        temperature=batched_temp,
    )

    if isinstance(out, dict):
        logits = out.get('logits')
        logprobs = out.get('logprobs')
        entropy = out.get('entropy')
    else:
        logits = getattr(out, 'logits', None)
        logprobs = getattr(out, 'logprobs', None)
        entropy = getattr(out, 'entropy', None)

    # Build result dict, reassembling whichever outputs exist
    result = {}

    if logits is not None:
        reassembled = []
        for i, seg_len in enumerate(segment_lengths):
            reassembled.append(logits[i, :seg_len])
        result['logits'] = torch.cat(reassembled, dim=0).unsqueeze(0)

    if logprobs is not None:
        reassembled = []
        for i, seg_len in enumerate(segment_lengths):
            reassembled.append(logprobs[i, :seg_len])
        result['logprobs'] = torch.cat(reassembled, dim=0).unsqueeze(0)

    if entropy is not None:
        reassembled = []
        for i, seg_len in enumerate(segment_lengths):
            reassembled.append(entropy[i, :seg_len])
        result['entropy'] = torch.cat(reassembled, dim=0).unsqueeze(0)

    return cast_float_and_contiguous(result)


@jaxtyped(typechecker=typechecker)
def forward(
    model: nn.Module,
    input_ids: Int[Tensor, "batch seq"],
    position_ids: Int[Tensor, "batch seq"],
    labels: Int[Tensor, "batch seq"] | None = None,
    temperature: Tensor | None = None,
) -> PrimeLmOutput:
    # For Mamba-hybrid models with packed sequences (batch=1), use segmented forward
    if input_ids.shape[0] == 1 and _is_nemotron_h(model):
        return _forward_mamba_segmented(model, input_ids, position_ids, labels, temperature)

    out = model(input_ids=input_ids, position_ids=position_ids, labels=labels, temperature=temperature)

    # PrimeLmOutput is a TypedDict (dict at runtime), HF outputs are dataclass-like objects
    if isinstance(out, dict):
        return cast_float_and_contiguous(out)

    return cast_float_and_contiguous(PrimeLmOutput(logits=out.logits))
