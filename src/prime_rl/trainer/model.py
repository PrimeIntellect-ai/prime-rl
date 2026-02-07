import logging
import time
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from beartype import beartype as typechecker
from huggingface_hub import snapshot_download
from jaxtyping import Float, Int, jaxtyped
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
from prime_rl.utils.vlm import is_vlm_model

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


def freeze_vision_encoder(model: nn.Module) -> None:
    """Freeze the vision encoder parameters for VLM training.

    For Qwen3-VL, the vision encoder is at model.model.visual.
    This freezes all parameters in the vision encoder so only the
    language model (with LoRA) is trained.
    """
    logger = get_logger()

    # Qwen3-VL structure: model.model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        vision_encoder = model.model.visual
    # Qwen2-VL structure: model.visual
    elif hasattr(model, "visual"):
        vision_encoder = model.visual
    else:
        raise ValueError("Could not find vision encoder to freeze. Expected model.model.visual or model.visual")

    num_frozen = 0
    for param in vision_encoder.parameters():
        param.requires_grad = False
        num_frozen += 1
    logger.info(f"Froze {num_frozen} parameters in vision encoder")


def is_tt_moe_model(model: nn.Module) -> bool:
    return hasattr(model.config, "num_experts") or hasattr(model.config, "n_routed_experts")


def get_language_model(model: nn.Module) -> nn.Module:
    """Get the language model component containing transformer layers.

    For VLM models (Qwen3-VL): model.model.language_model
    For text-only models: model.model
    """
    if hasattr(model.model, "language_model"):
        return model.model.language_model
    return model.model


def get_load_balance_stats(
    model: nn.Module, reset_stats: bool = True, try_to_avoid_padding_experts: bool = True
) -> dict[str, Tensor | None]:
    per_layer_max_vio = []
    language_model = get_language_model(model)
    for transformer_block in language_model.layers:
        # This is necessary for models that have mixed dense layers
        if not hasattr(transformer_block.mlp, "tokens_per_expert"):
            continue
        tokens_per_expert: torch.Tensor = transformer_block.mlp.tokens_per_expert
        if try_to_avoid_padding_experts:
            tokens_per_expert = tokens_per_expert.sort(dim=0, descending=True).values[
                transformer_block.mlp.router.top_k :
            ]
        balanced_load = tokens_per_expert.mean()
        max_vio = (tokens_per_expert.max() - balanced_load) / balanced_load
        per_layer_max_vio.append(max_vio.item())
        if reset_stats:
            transformer_block.mlp.tokens_per_expert.zero_()
    if len(per_layer_max_vio) == 0:
        return {"max_vio": None}
    return {"max_vio": torch.tensor(per_layer_max_vio, device=torch.device("cuda"))}


def get_model(
    config: ModelConfig, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.bfloat16
) -> nn.Module:
    logger = get_logger()
    logger.info(
        f"Loading model config (name={config.name}, attn={config.attn}, trust_remote_code={config.trust_remote_code})"
    )

    # Check if this is a vision-language model
    is_vlm = is_vlm_model(config.name)
    if is_vlm:
        logger.info(f"Detected vision-language model: {config.name}")

    model_config = cast(
        PretrainedConfig,
        AutoConfig.from_pretrained(
            config.name, attn_implementation=config.attn, trust_remote_code=config.trust_remote_code
        ),
    )
    model_config.use_cache = False
    model_config.use_grouped_mm = config.moe_use_grouped_mm

    # NOTE: For VLM models, we do NOT propagate dtype to sub_configs.
    # The model should load in its default dtype (bf16) to match vLLM inference.
    # The FSDP MixedPrecisionPolicy handles compute dtype separately.

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

    if is_vlm and impl_to_use != "hf":
        raise ValueError(
            f"VLM models only support impl='hf', but got impl='{config.impl}' (resolved to '{impl_to_use}'). "
            f"Set impl='hf' or impl='auto' in your model config."
        )

    with device:
        # For VLM models, use AutoModelForVision2Seq or import specific model class
        if is_vlm:
            from transformers import AutoModelForVision2Seq

            model_cls = AutoModelForVision2Seq
        else:
            match impl_to_use:
                case "hf":
                    model_cls = AutoModelForCausalLM
                case "custom":
                    model_cls = AutoModelForCausalLMPrimeRL

        load_model_start_time = time.perf_counter()
        # VLM models use standard HF API which requires torch_dtype, custom models use dtype
        dtype_kwarg = {"torch_dtype": dtype} if is_vlm else {"dtype": dtype}
        if device == torch.device("meta"):
            logger.info(f"Loading model {config.name} using {model_cls.__name__} to meta device")
            model = model_cls.from_config(model_config, trust_remote_code=config.trust_remote_code, **dtype_kwarg)
        else:
            logger.info(f"Loading model {config.name} using {model_cls.__name__} to CPU")
            model = model_cls.from_pretrained(
                pretrained_model_name_or_path=config.name,
                config=model_config,
                trust_remote_code=config.trust_remote_code,
                **dtype_kwarg,
            )
        logger.debug(f"Loaded model {config.name} in {time.perf_counter() - load_model_start_time:.2f} seconds")

    # For VLM models, freeze the vision encoder
    if is_vlm:
        freeze_vision_encoder(model)

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

    # For VLM models, shard the frozen vision encoder as a single unit
    # This allows FSDP to manage the memory while keeping it frozen
    is_vlm = is_vlm_model(config.name)
    if is_vlm:
        if hasattr(model, "model") and hasattr(model.model, "visual"):
            vision_encoder = model.model.visual
        elif hasattr(model, "visual"):
            vision_encoder = model.visual
        else:
            raise ValueError(f"VLM model {config.name} does not have a recognized vision encoder attribute")

        fully_shard(
            vision_encoder,
            mesh=hsdp_mesh,
            **fsdp_config,
        )
        get_logger().info("Applied FSDP to frozen vision encoder")

    # Get the language model layers (handle VLM structure)
    # For Qwen3-VL: model.model.language_model contains the transformer layers
    # For text-only models: model.model contains the layers directly
    if is_vlm:
        language_model = model.model.language_model
        transformer_layers = language_model.layers
    else:
        language_model = model.model
        transformer_layers = language_model.layers

    for transformer_block in transformer_layers:
        if parallel_dims.ep_enabled and isinstance(transformer_block.mlp, MoE):
            fully_shard(transformer_block.mlp.experts, mesh=dp_mod_ep_mesh, **fsdp_config)

            transformer_block.mlp.experts.set_gradient_divide_factor(parallel_dims.fsdp_gradient_divide_factor)

        fully_shard(
            transformer_block,
            mesh=hsdp_mesh,
            **fsdp_config,
        )

    shard_norm_and_lm_head = hasattr(model, "config") and not model.config.tie_word_embeddings

    if shard_norm_and_lm_head:
        # This optimization breaks weight tying
        fully_shard(
            language_model.embed_tokens,
            mesh=hsdp_mesh,
            **fsdp_config,
        )
        fully_shard(
            [model.lm_head, language_model.norm],
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

    transformer_blocks = list(language_model.layers)
    next_transformer_blocks = transformer_blocks[1:] + [None]

    if language_model.embed_tokens is not None and len(language_model.layers) > 0:
        if shard_norm_and_lm_head:
            language_model.embed_tokens.set_modules_to_forward_prefetch([transformer_blocks[0]])

    for transformer_block, next_transformer_block in zip(transformer_blocks, next_transformer_blocks):
        if next_transformer_block is not None:
            if isinstance(next_transformer_block.mlp, MoE):
                transformer_block.set_modules_to_forward_prefetch(
                    [next_transformer_block, next_transformer_block.mlp.experts]
                )
            else:
                transformer_block.set_modules_to_forward_prefetch([next_transformer_block])
        elif language_model.norm is not None and model.lm_head is not None:
            if shard_norm_and_lm_head:
                transformer_block.set_modules_to_forward_prefetch([language_model.norm, model.lm_head])

    # backward
    reversed_transformer_blocks = list(reversed(language_model.layers))
    prev_transformer_blocks = reversed_transformer_blocks[1:] + [None]

    if language_model.norm is not None and model.lm_head is not None and len(language_model.layers) > 0:
        if shard_norm_and_lm_head:
            model.lm_head.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])
        else:
            model.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])

    for transformer_block, prev_transformer_block in zip(reversed_transformer_blocks, prev_transformer_blocks):
        if prev_transformer_block is not None:
            if isinstance(prev_transformer_block.mlp, MoE):
                transformer_block.set_modules_to_backward_prefetch(
                    [prev_transformer_block, prev_transformer_block.mlp.experts]
                )
            else:
                transformer_block.set_modules_to_backward_prefetch([prev_transformer_block])
        elif language_model.embed_tokens is not None:
            if shard_norm_and_lm_head:
                transformer_block.set_modules_to_backward_prefetch([language_model.embed_tokens])


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

    # Gemma3 model (has embed_scale and local rotary emb)
    gemma3_buffers = {"model.embed_tokens.embed_scale", "model.rotary_emb.inv_freq", "model.rotary_emb_local.inv_freq"}
    if set(buffer_names) == gemma3_buffers:
        return True

    get_logger().warning(f"Model cannot be loaded using meta device because of buffers: {buffer_names}")
    return False


def fix_model_post_empty(model: nn.Module):
    buffer_names = [name for name, _ in model.named_buffers()]
    # HF standard transformer model
    if "model.rotary_emb.inv_freq" in buffer_names:
        rotary_emb = model.model.rotary_emb
        inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(rotary_emb.config, rotary_emb.inv_freq.device)
        rotary_emb.inv_freq.copy_(inv_freq)
    # Gemma3 local rotary emb
    if "model.rotary_emb_local.inv_freq" in buffer_names:
        rotary_emb_local = model.model.rotary_emb_local
        inv_freq_local, rotary_emb_local.attention_scaling = rotary_emb_local.rope_init_fn(
            rotary_emb_local.config, rotary_emb_local.inv_freq.device
        )
        rotary_emb_local.inv_freq.copy_(inv_freq_local)
    # Gemma3 embed_scale (scalar computed from hidden_size)
    if "model.embed_tokens.embed_scale" in buffer_names:
        embed_scale = model.config.hidden_size**0.5
        model.model.embed_tokens.embed_scale.fill_(embed_scale)


def reshard_module(model: nn.Module):
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def apply_ac(model: nn.Module, ac_config: ActivationCheckpointConfig):
    language_model = get_language_model(model)
    for layer_id, (layer_name, transformer_block) in enumerate(language_model.layers.named_children()):
        if layer_id % ac_config.freq == 0:
            transformer_block = checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        language_model.layers.register_module(layer_name, transformer_block)
    get_logger().info(f"Applied activation checkpointing (freq={ac_config.freq})")


def apply_compile(model: nn.Module, compile_config: CompileConfig):
    torch._dynamo.config.capture_scalar_outputs = True
    language_model = get_language_model(model)
    for layer_id in range(len(language_model.layers)):
        # Doing it in-place avoids mangled fqn which can break checkpoint loading
        language_model.layers[layer_id].compile(fullgraph=compile_config.fullgraph)
    get_logger().info(f"Compiled {len(language_model.layers)} layers (fullgraph={compile_config.fullgraph})")


def apply_ep(model: nn.Module, parallel_dims: ParallelDims):
    language_model = get_language_model(model)
    for transformer_block in language_model.layers:
        if isinstance(transformer_block.mlp, MoE):
            parallelize_module(
                transformer_block.mlp.experts,
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


def _register_fa4_attention_interface() -> None:
    """Register a dummy `fa4` attention with transformers so AutoConfig accepts it.

    The `flash_attention_*` naming pattern triggers transformers to attempt
    installing a kernel from the hub, so we use the short name `fa4` internally.
    This dummy is never called because fa4 is only supported with our custom
    model implementation.
    """
    from transformers import AttentionInterface

    def _noop(*args, **kwargs) -> None:
        pass

    AttentionInterface.register("fa4", _noop)


def setup_model(
    config: ModelConfig, parallel_dims: ParallelDims, loading_from_checkpoint_later: bool = False
) -> nn.Module:
    if config.attn == "flash_attention_3" and not is_flash_attn_3_available():
        raise ValueError(
            "Flash attention 3 is only supported if the flash_attn_3 package is installed. Install with `uv pip install 'flash-attn-3 @ git+https://github.com/Dao-AILab/flash-attention.git@main#subdirectory=hopper' --no-build-isolation`"
        )

    if config.attn == "fa4":
        _validate_flash_attn_4_installed()
        _register_fa4_attention_interface()

    logger = get_logger()

    # 1. We load to meta device by default
    model = get_model(config, device=torch.device("meta"), dtype=DTYPE_MAP[config.optimization_dtype])

    possible_to_load_to_meta = can_reinit_empty_buffers(model)

    if config.debug.random_init and not possible_to_load_to_meta:
        raise ValueError(
            "It's not possible to load to meta device and random initialize is enabled. Please disable random initialize or use a different model."
        )

    # 1a. We load to CPU if we cannot reinit empty buffers
    if not possible_to_load_to_meta:
        logger.warning("Cannot load model to meta device only, loading to CPU instead.")
        model = get_model(config, device=torch.device("cpu"), dtype=DTYPE_MAP[config.optimization_dtype])

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


@jaxtyped(typechecker=typechecker)
def forward(
    model: nn.Module,
    input_ids: Int[Tensor, "batch seq"],
    position_ids: Int[Tensor, "batch seq"],
    labels: Int[Tensor, "batch seq"] | None = None,
    temperature: Tensor | None = None,
    # Multimodal fields (Qwen3-VL)
    pixel_values: Float[Tensor, "num_patches patch_dim"] | None = None,
    image_grid_thw: Int[Tensor, "num_images 3"] | None = None,
) -> PrimeLmOutput:
    # Build kwargs for model forward
    kwargs = {
        "input_ids": input_ids,
        "labels": labels,
        "temperature": temperature,
    }

    # For multimodal (VLM), don't pass position_ids - let the model compute MRoPE internally
    # using image_grid_thw. Qwen3-VL only computes proper MRoPE when position_ids is None.
    if pixel_values is not None:
        assert image_grid_thw is not None, "pixel_values requires image_grid_thw for MRoPE computation"
        kwargs["pixel_values"] = pixel_values
        kwargs["image_grid_thw"] = image_grid_thw
    else:
        kwargs["position_ids"] = position_ids

    out = model(**kwargs)

    # PrimeLmOutput is a TypedDict (dict at runtime), HF outputs are dataclass-like objects
    if isinstance(out, dict):
        return cast_float_and_contiguous(out)

    return cast_float_and_contiguous(PrimeLmOutput(logits=out.logits))
