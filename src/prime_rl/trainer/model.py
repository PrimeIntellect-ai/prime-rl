import logging
import time
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from beartype import beartype as typechecker
from huggingface_hub import snapshot_download
from jaxtyping import Float, Int, jaxtyped
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from torch import Tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, OffloadPolicy, fully_shard
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.import_utils import is_flash_attn_3_available

from prime_rl.trainer.config import ActivationCheckpointConfig, CompileConfig, ModelConfig, TokenizerConfig
from prime_rl.trainer.lora import apply_lora_to_model, strip_lora_from_state_dict
from prime_rl.trainer.models import AutoModelForCausalLMPrimeRL, PreTrainedModelPrimeRL, supports_custom_impl
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


def get_load_balance_stats(
    model: nn.Module, reset_stats: bool = True, try_to_avoid_padding_experts: bool = True
) -> dict[str, Tensor | None]:
    per_layer_max_vio = []
    for transformer_block in model.model.layers:
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
    model_config = cast(
        PretrainedConfig,
        AutoConfig.from_pretrained(
            config.name, attn_implementation=config.attn, trust_remote_code=config.trust_remote_code
        ),
    )
    model_config.use_cache = False
    model_config.use_grouped_mm = config.moe_use_grouped_mm
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
                model_cls = AutoLigerKernelForCausalLM
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
    # TODO: Support dp_replicate
    if config.dp_replicate > 1:
        hsdp_mesh = parallel_dims.world_mesh["dp_replicate", "dp_shard_cp"]
    else:
        hsdp_mesh = parallel_dims.world_mesh["dp_shard_cp"]

    offload_policy: OffloadPolicy = CPUOffloadPolicy(pin_memory=True) if config.fsdp_cpu_offload else OffloadPolicy()

    for transformer_block in model.model.layers:
        fully_shard(
            transformer_block,
            mesh=hsdp_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=config.reshard_after_forward,
        )

    if hasattr(model, "config") and model.config.tie_word_embeddings:
        fully_shard(
            [model.model.embed_tokens, model.lm_head],
            mesh=hsdp_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=config.reshard_after_forward,
        )
    else:
        fully_shard(
            model.lm_head,
            mesh=hsdp_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=config.reshard_after_forward,
        )
    fully_shard(
        model.model,
        mesh=hsdp_mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        reshard_after_forward=config.reshard_after_forward,
    )


def load_dcp_from_hf(model: nn.Module, config: ModelConfig, parallel_dims: ParallelDims):
    model.to_empty(device="cuda")
    torch.distributed.barrier()

    logger = get_logger()
    if config.debug.random_init:
        logger.warning("Randomly initializing model. Skipping loading weights from HF.")
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
    if isinstance(model, PreTrainedModelPrimeRL):
        model.init_buffers_post_meta()
    else:
        fix_model_post_empty(model)
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
        rotary_emb = model.model.rotary_emb
        inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(rotary_emb.config, rotary_emb.inv_freq.device)
        rotary_emb.inv_freq.copy_(inv_freq)


def reshard_module(model: nn.Module):
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def apply_ac(model: nn.Module, ac_config: ActivationCheckpointConfig):
    for layer_id, (layer_name, transformer_block) in enumerate(model.model.layers.named_children()):
        if layer_id % ac_config.freq == 0:
            transformer_block = checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        model.model.layers.register_module(layer_name, transformer_block)
    get_logger().info(f"Applied activation checkpointing (freq={ac_config.freq})")


def apply_compile(model: nn.Module, compile_config: CompileConfig):
    torch._dynamo.config.capture_scalar_outputs = True
    for layer_id in range(len(model.model.layers)):
        # Doing it in-place avoids mangled fqn which can break checkpoint loading
        model.model.layers[layer_id].compile(fullgraph=compile_config.fullgraph)
    get_logger().info(f"Compiled {len(model.model.layers)} layers (fullgraph={compile_config.fullgraph})")


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
    possible_to_load_to_meta = can_reinit_empty_buffers(model)

    if config.debug.random_init and not possible_to_load_to_meta:
        raise ValueError(
            "It's not possible to load to meta device and random initialize is enabled. Please disable random initialize or use a different model."
        )

    # 1a. We load to CPU if we cannot reinit empty buffers
    if not possible_to_load_to_meta:
        logger.warning("Cannot load model to meta device only, loading to CPU instead.")
        model = get_model(config, device=torch.device("cpu"), dtype=DTYPE_MAP[config.optimization_dtype])

    # Apply LoRA before FSDP setup
    if config.lora is not None:
        apply_lora_to_model(model, config.lora)

    # the right order is AC -> Compile -> FSDP
    if config.ac is not None:
        apply_ac(model, config.ac)
    if config.compile is not None:
        apply_compile(model, config.compile)

    setup_fsdp(model, config, parallel_dims)

    # 2. if we can load to meta, we either:
    if possible_to_load_to_meta:
        # - load from checkpoint later if needed
        if loading_from_checkpoint_later:
            logger.warning(
                "Skipping loading weights. Initializing an empty model on device, loading from checkpoint later."
            )
            model.to_empty(device="cuda")
            torch.distributed.barrier()
            if isinstance(model, PreTrainedModelPrimeRL):
                model.init_buffers_post_meta()
            else:
                fix_model_post_empty(model)
        # - or load from HF with dcp
        else:
            load_dcp_from_hf(model, config, parallel_dims)

    logger.debug(f"Model signature: {get_module_signature(model, compress=True)}")
    return model


@jaxtyped(typechecker=typechecker)
def forward(
    model: nn.Module, input_ids: Int[Tensor, "batch seq"], position_ids: Int[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq vocab"]:
    return model(input_ids=input_ids, position_ids=position_ids).logits
