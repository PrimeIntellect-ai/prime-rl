import logging
import os
import time
from pathlib import Path
from typing import cast

# Disable transformers hub kernel interception by default. The `kernels` package, when installed,
# causes transformers to auto-replace modules (e.g. mamba-ssm) with hub kernel versions that may
# have incompatible CUDA requirements. We only enable it explicitly for models that need it (GPT-OSS).
os.environ.setdefault("USE_HUB_KERNELS", "NO")

import torch
import torch._dynamo
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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PretrainedConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.import_utils import is_flash_attn_3_available

from prime_rl.configs.trainer import ActivationCheckpointConfig, CompileConfig, ModelConfig, TokenizerConfig
from prime_rl.trainer.distributed import DeepEPExpertParallel
from prime_rl.trainer.lora import apply_lora_to_model, freeze_all_except_lora_and_specified, strip_lora_from_state_dict
from prime_rl.trainer.models import (
    AutoModelForCausalLMPrimeRL,
    PreTrainedModelPrimeRL,
    PrimeLmOutput,
    cast_float_and_contiguous,
    get_custom_vlm_cls,
    supports_custom_impl,
)
from prime_rl.trainer.models.layers.checkpointing import (
    get_supported_targets,
    set_selective_activation_checkpointing,
    supports_selective_activation_checkpointing,
)
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.layers.moe import LatentMoE, MoE
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.weights import (
    load_state_dict,
    load_state_dict_keys,
    save_state_dict,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.vlm import get_language_model, get_vision_encoder, is_vlm_architecture


def _patch_qwen3_5_moe_conversion_mapping():
    """Fix Qwen3.5 MoE conversion mapping incorrectly applying qwen2_moe expert weight splitting.

    Qwen3.5 MoE stores expert weights as fused 3D tensors natively in the checkpoint
    (e.g. experts.gate_up_proj [num_experts, 2*intermediate, hidden]). The upstream mapping
    incorrectly maps qwen3_5_moe → qwen2_moe, which assumes per-expert 2D checkpoint weights,
    causing revert_weight_conversion to produce wrong shapes during weight broadcasting.

    Remove once the pinned transformers commit fixes this.
    """
    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping,
        register_checkpoint_conversion_mapping,
    )

    # qwen3_5_moe_text: keep only the qwen3_5_text renaming, remove qwen2_moe expert conversion
    qwen3_5_text_mapping = get_checkpoint_conversion_mapping("qwen3_5_text")
    if qwen3_5_text_mapping is not None:
        register_checkpoint_conversion_mapping("qwen3_5_moe_text", qwen3_5_text_mapping, overwrite=True)

    # qwen3_5_moe: remove the qwen2_moe fallback entirely
    register_checkpoint_conversion_mapping("qwen3_5_moe", [], overwrite=True)


def _patch_qwen3_5_text_position_ids():
    """Fix Qwen3.5 passing 3D MRoPE position_ids to decoder layers instead of 2D text_position_ids.

    Upstream fix: https://github.com/huggingface/transformers/pull/44399
    Remove once the pinned transformers commit includes this fix.
    """
    import inspect

    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextModel
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextModel

    for text_model_cls, decoder_layer_cls in [
        (Qwen3_5TextModel, Qwen3_5DecoderLayer),
        (Qwen3_5MoeTextModel, Qwen3_5MoeDecoderLayer),
    ]:
        source = inspect.getsource(text_model_cls.forward)
        if "decoder_layer" in source and "position_ids=text_position_ids" in source.split("decoder_layer")[-1]:
            continue  # already fixed upstream

        _original_forward = decoder_layer_cls.forward

        def _make_patched_forward(original):
            def _patched_forward(self, hidden_states, position_ids=None, **kwargs):
                if position_ids is not None and position_ids.ndim == 3:
                    position_ids = position_ids[0]
                return original(self, hidden_states, position_ids=position_ids, **kwargs)

            return _patched_forward

        decoder_layer_cls.forward = _make_patched_forward(_original_forward)


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
        if mlp is None:
            continue

        # Custom implementation: MoE/LatentMoE class with router attribute
        if isinstance(mlp, (MoE, LatentMoE)):
            for param in mlp.router.parameters():
                param.requires_grad = False
                num_frozen += 1
        # HuggingFace implementation: gate attribute (nn.Linear)
        elif hasattr(mlp, "gate") and isinstance(mlp.gate, nn.Linear):
            for param in mlp.gate.parameters():
                param.requires_grad = False
                num_frozen += 1

    if num_frozen == 0:
        raise ValueError("No MoE router parameters found to freeze. Is this a MoE model?")

    logger.info(f"Froze {num_frozen} MoE router parameters")


def is_tt_moe_model(model: nn.Module) -> bool:
    return hasattr(model.config, "num_experts") or hasattr(model.config, "n_routed_experts")


def configure_moe_ep_backend(model: nn.Module, config: ModelConfig) -> None:
    backend = config.ep_comm_backend
    if backend == "deepep":
        from prime_rl.trainer.distributed.deepep import configure_num_sms

        configure_num_sms(config.deepep_num_sms)
    language_model = get_language_model(model)
    for transformer_block in language_model.layers:
        if not isinstance(transformer_block.mlp, (MoE, LatentMoE)):
            continue
        transformer_block.mlp.set_ep_comm_backend(backend)
        transformer_block.mlp.set_deepep_token_chunk_size(config.deepep_token_chunk_size)


def get_load_balance_stats(
    model: nn.Module, reset_stats: bool = True, try_to_avoid_padding_experts: bool = True
) -> dict[str, Tensor | None]:
    per_layer_max_vio = []
    language_model = get_language_model(model)
    for transformer_block in language_model.layers:
        # This is necessary for models that have mixed dense layers
        block_mlp = getattr(transformer_block, "mlp", None)
        if block_mlp is None or not hasattr(block_mlp, "tokens_per_expert"):
            continue
        tokens_per_expert: torch.Tensor = block_mlp.tokens_per_expert
        if try_to_avoid_padding_experts:
            tokens_per_expert = tokens_per_expert.sort(dim=0, descending=True).values[block_mlp.router.top_k :]
        balanced_load = tokens_per_expert.mean()
        max_vio = (tokens_per_expert.max() - balanced_load) / balanced_load
        per_layer_max_vio.append(max_vio.item())
        if reset_stats:
            block_mlp.tokens_per_expert.zero_()
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

    is_vlm_training = config.vlm is not None

    if "Qwen3.5" in config.name or "qwen3_5" in config.name.lower():
        _patch_qwen3_5_text_position_ids()
        _patch_qwen3_5_moe_conversion_mapping()

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
        if config.optimization_dtype != "bfloat16" or config.reduce_dtype != "bfloat16":
            raise ValueError(
                "VLM models must use optimization_dtype='bfloat16' and reduce_dtype='bfloat16' to match vLLM inference."
            )

    # GPT-OSS only supports FlashAttention via kernels-community/vllm-flash-attn3, which requires Hopper (SM 90).
    # On other architectures (e.g. Blackwell), users must fall back to eager attention.
    HOPPER_MAJOR = 9
    if getattr(model_config, "model_type", "") == "gpt_oss":
        if config.attn != "eager":
            major, minor = torch.cuda.get_device_capability()
            if major != HOPPER_MAJOR:
                raise ValueError(
                    f"GPT-OSS requires 'attn = \"eager\"' on non-Hopper GPUs (detected SM {major}{minor}). "
                    f"The only flash attention kernel supported by GPT-OSS (kernels-community/vllm-flash-attn3) is Hopper-only. "
                    f'Set [trainer.model] attn = "eager" in your config.'
                )
        # Enable hub kernels for GPT-OSS (disabled by default to avoid interfering with other models).
        import transformers.integrations.hub_kernels as _hub_kernels

        _hub_kernels._kernels_enabled = True

    # Fallback Qwen3.5 patch detection from loaded config model_type
    if getattr(model_config, "model_type", "").startswith("qwen3_5_moe"):
        _patch_qwen3_5_text_position_ids()
        _patch_qwen3_5_moe_conversion_mapping()
    for subconfig_key in getattr(model_config, "sub_configs", {}):
        subconfig = getattr(model_config, subconfig_key, None)
        if subconfig is not None and hasattr(subconfig, "use_cache"):
            subconfig.use_cache = False
    model_config.use_grouped_mm = config.moe_use_grouped_mm

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

    # Determine the implementation to use
    custom_vlm_cls = get_custom_vlm_cls(model_config) if is_vlm_arch else None
    if config.impl == "auto":
        if is_vlm_arch:
            impl_to_use = "custom" if custom_vlm_cls is not None else "hf"
        else:
            impl_to_use = "custom" if supports_custom_impl(model_config) else "hf"
        logger.info(f"Auto-selected implementation: {impl_to_use}")
    else:
        impl_to_use = config.impl

    with device:
        if impl_to_use == "custom" and custom_vlm_cls is not None:
            model_cls = custom_vlm_cls
        elif is_vlm_arch:
            from transformers import AutoModelForImageTextToText

            model_cls = AutoModelForImageTextToText
        else:
            match impl_to_use:
                case "hf":
                    model_cls = AutoModelForCausalLM
                case "custom":
                    model_cls = AutoModelForCausalLMPrimeRL

        load_model_start_time = time.perf_counter()
        # HF VLM models require torch_dtype; custom PrimeRL models and text Auto models use dtype
        use_torch_dtype = is_vlm_arch and model_cls is not custom_vlm_cls
        dtype_kwarg = {"torch_dtype": dtype} if use_torch_dtype else {"dtype": dtype}
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

    # For VLM models, optionally freeze the vision encoder
    if is_vlm_training and config.vlm.freeze_vision_encoder:
        freeze_vision_encoder(model, override_attr=config.vlm.vision_encoder_attr)

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

    hsdp_mesh = parallel_dims.get_mesh("hsdp")

    dp_mod_ep_mesh: DeviceMesh | None = None
    if parallel_dims.ep_enabled:
        dp_mod_ep_mesh_dim_names = []
        if parallel_dims.dp_replicate_enabled:
            dp_mod_ep_mesh_dim_names.append("dp_replicate")
        dp_mod_ep_mesh_dim_names.append("dp_shard_mod_ep")

        dp_mod_ep_mesh = parallel_dims.world_mesh[tuple(dp_mod_ep_mesh_dim_names)]

    is_vlm_training = config.vlm is not None
    if is_vlm_training:
        vision_encoder = get_vision_encoder(model, override=config.vlm.vision_encoder_attr)
        if vision_encoder is None:
            raise ValueError(f"VLM model {config.name} has no recognized vision encoder")

        fully_shard(vision_encoder, mesh=hsdp_mesh, **fsdp_config)
        get_logger().info(f"Applied FSDP to vision encoder (frozen={config.vlm.freeze_vision_encoder})")

    language_model = get_language_model(model, override=config.vlm.language_model_attr if is_vlm_training else None)
    transformer_layers = language_model.layers

    for transformer_block in transformer_layers:
        block_mlp = getattr(transformer_block, "mlp", None)
        if parallel_dims.ep_enabled and block_mlp is not None and isinstance(block_mlp, (MoE, LatentMoE)):
            fully_shard(block_mlp.experts, mesh=dp_mod_ep_mesh, **fsdp_config)

            block_mlp.experts.set_gradient_divide_factor(parallel_dims.fsdp_gradient_divide_factor)

        fully_shard(
            transformer_block,
            mesh=hsdp_mesh,
            **fsdp_config,
        )

    shard_norm_and_lm_head = hasattr(model, "config") and not model.config.tie_word_embeddings

    if shard_norm_and_lm_head:
        # This optimization breaks weight tying
        embed_module = getattr(language_model, "embed_tokens", None) or getattr(language_model, "embeddings", None)
        fully_shard(
            embed_module,
            mesh=hsdp_mesh,
            **fsdp_config,
        )
        norm_module = getattr(language_model, "norm", None) or language_model.norm_f
        fully_shard(
            [model.lm_head, norm_module],
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

    embed_module = getattr(language_model, "embed_tokens", None) or getattr(language_model, "embeddings", None)
    if embed_module is not None and len(language_model.layers) > 0:
        if shard_norm_and_lm_head:
            embed_module.set_modules_to_forward_prefetch([transformer_blocks[0]])

    for transformer_block, next_transformer_block in zip(transformer_blocks, next_transformer_blocks):
        if next_transformer_block is not None:
            next_mlp = getattr(next_transformer_block, "mlp", None)
            if next_mlp is not None and isinstance(next_mlp, (MoE, LatentMoE)):
                transformer_block.set_modules_to_forward_prefetch([next_transformer_block, next_mlp.experts])
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
            prev_mlp = getattr(prev_transformer_block, "mlp", None)
            if prev_mlp is not None and isinstance(prev_mlp, (MoE, LatentMoE)):
                transformer_block.set_modules_to_backward_prefetch([prev_transformer_block, prev_mlp.experts])
            else:
                transformer_block.set_modules_to_backward_prefetch([prev_transformer_block])
        elif embed_module is not None:
            if shard_norm_and_lm_head:
                transformer_block.set_modules_to_backward_prefetch([embed_module])


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

    # Dynamically convert between different weight formats if needed.
    # All ranks read just the key names (cheap) to determine the path independently.
    # Only master loads the full state dict when conversion is actually needed.
    if isinstance(model, PreTrainedModelPrimeRL):
        snapshot_keys = dict.fromkeys(load_state_dict_keys(snapshot_path))
        model_keys = dict.fromkeys(model.state_dict().keys())

        if model.is_hf_state_dict(snapshot_keys) and model.is_prime_state_dict(model_keys):
            logger.warning(
                "Found HF weight format in snapshot state dict and PrimeRL weight format in model state dict. Trying to auto-convert..."
            )
            snapshot_path = snapshot_path / "prime"
            if not snapshot_path.exists() and get_world().is_master:
                logger.debug(
                    f"Converting snapshot state dict to PrimeRL format and saving to {snapshot_path} on master rank. This is a one-time operation."
                )
                snapshot_state_dict = load_state_dict(snapshot_path.parent)
                model.convert_to_prime(snapshot_state_dict)
                save_state_dict(snapshot_state_dict, snapshot_path)
                del snapshot_state_dict

        elif model.is_prime_state_dict(snapshot_keys) and model.is_hf_state_dict(model_keys):
            logger.warning(
                "Found PrimeRL weight format in snapshot state dict and HF weight format in model state dict. Trying to auto-convert..."
            )
            snapshot_path = snapshot_path / "hf"
            if not snapshot_path.exists() and get_world().is_master:
                logger.debug(
                    f"Converting snapshot state dict to HF format and saving to {snapshot_path} on master rank. This is a one-time operation."
                )
                snapshot_state_dict = load_state_dict(snapshot_path.parent)
                model.convert_to_hf(snapshot_state_dict)
                save_state_dict(snapshot_state_dict, snapshot_path)
                del snapshot_state_dict

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
    # Restore weight tying broken by to_empty() for HF models
    if not isinstance(model, PreTrainedModelPrimeRL) and model.config.tie_word_embeddings:
        model.tie_weights()
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
    # Custom PrimeRL models handle buffer reinit via init_buffers_post_meta
    if isinstance(model, PreTrainedModelPrimeRL):
        return True

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
    logger = get_logger()
    language_model = get_language_model(model)
    target_list = sorted(frozenset(ac_config.targets))
    selective_layers = 0
    full_layers = 0
    fallback_layer_types: set[str] = set()
    model_supported_targets: set[str] = set()

    for layer_id, (layer_name, transformer_block) in enumerate(language_model.layers.named_children()):
        if layer_id % ac_config.freq != 0:
            continue

        if ac_config.mode == "selective" and supports_selective_activation_checkpointing(transformer_block):
            model_supported_targets.update(get_supported_targets(transformer_block))
            set_selective_activation_checkpointing(transformer_block, target_list)
            selective_layers += 1
        else:
            if ac_config.mode == "selective":
                fallback_layer_types.add(type(transformer_block).__name__)
            transformer_block = checkpoint_wrapper(transformer_block, preserve_rng_state=False)
            full_layers += 1

        language_model.layers.register_module(layer_name, transformer_block)

    if ac_config.mode == "selective":
        unsupported_targets = frozenset(target_list) - model_supported_targets
        if unsupported_targets:
            raise ValueError(
                f"Selective activation checkpoint targets {sorted(unsupported_targets)} are not supported "
                f"by the selected model layers. Supported targets across the model: {sorted(model_supported_targets)}"
            )
        if fallback_layer_types:
            logger.warning(
                "Selective activation checkpointing is not supported for layer types "
                f"{sorted(fallback_layer_types)}; falling back to full checkpointing for those layers."
            )
        logger.info(
            "Applied selective activation checkpointing "
            f"(freq={ac_config.freq}, targets={target_list}, selective_layers={selective_layers}, "
            f"full_fallback_layers={full_layers})"
        )
        return

    logger.info(f"Applied activation checkpointing (freq={ac_config.freq})")


def apply_compile(model: nn.Module, compile_config: CompileConfig):
    torch._dynamo.config.capture_scalar_outputs = True
    language_model = get_language_model(model)
    for layer_id in range(len(language_model.layers)):
        # Doing it in-place avoids mangled fqn which can break checkpoint loading
        language_model.layers[layer_id].compile(fullgraph=compile_config.fullgraph)
    get_logger().info(f"Compiled {len(language_model.layers)} layers (fullgraph={compile_config.fullgraph})")


def apply_ep(model: nn.Module, config: ModelConfig, parallel_dims: ParallelDims):
    language_model = get_language_model(model)
    for transformer_block in language_model.layers:
        block_mlp = getattr(transformer_block, "mlp", None)
        if block_mlp is not None and isinstance(block_mlp, (MoE, LatentMoE)):
            if config.ep_comm_backend == "torch":
                parallelize_plan = ExpertParallel()
            else:
                parallelize_plan = DeepEPExpertParallel()
            parallelize_module(
                block_mlp.experts,
                device_mesh=parallel_dims.get_mesh("ep"),
                parallelize_plan=parallelize_plan,
            )


def _move_buffers_to_cuda(model: nn.Module, config: ModelConfig) -> None:
    """FSDP CPU offloading only manages parameters, not buffers. Move buffers to CUDA."""
    if not config.fsdp_cpu_offload:
        return
    for _, buffer in model.named_buffers():
        if buffer.device.type == "cpu":
            buffer.data = buffer.data.to("cuda")


def _reset_runtime_moe_buffers(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (MoE, LatentMoE)) and module.tokens_per_expert.device.type != "meta":
            module.tokens_per_expert.zero_()


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
    config: ModelConfig,
    parallel_dims: ParallelDims,
    loading_from_checkpoint_later: bool = False,
    fused_cross_entropy: bool | str = False,
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
    configure_moe_ep_backend(model, config)

    possible_to_load_to_meta = can_reinit_empty_buffers(model)

    if config.debug.random_init and not possible_to_load_to_meta:
        raise ValueError(
            "It's not possible to load to meta device and random initialize is enabled. Please disable random initialize or use a different model."
        )

    # 1a. We load to CPU if we cannot reinit empty buffers
    if not possible_to_load_to_meta:
        logger.warning("Cannot load model to meta device only, loading to CPU instead.")
        model = get_model(config, device=torch.device("cpu"), dtype=DTYPE_MAP[config.optimization_dtype])
        configure_moe_ep_backend(model, config)

    lm_head_chunk_size: int | None = None
    if isinstance(config.fused_lm_head_token_chunk_size, int):
        lm_head_chunk_size = config.fused_lm_head_token_chunk_size

    inject_prime_lm_head(model, chunk_size=lm_head_chunk_size, fused_cross_entropy=fused_cross_entropy)

    # Apply LoRA before FSDP setup
    if config.lora is not None:
        apply_lora_to_model(model, config.lora)

    if config.freeze_moe_router:
        freeze_moe_router(model)

    if parallel_dims.ep_enabled:
        apply_ep(model, config, parallel_dims)
        # EP replaces params with DTensors that default to requires_grad=True,
        # re-freeze base params that LoRA froze earlier.
        if config.lora is not None:
            freeze_all_except_lora_and_specified(model, config.lora)

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
                # Restore weight tying broken by to_empty() for HF models
                if model.config.tie_word_embeddings:
                    model.tie_weights()

            _move_buffers_to_cuda(model, config)
        # - or load from HF with dcp
        else:
            load_dcp_from_hf(model, config, parallel_dims)

    _reset_runtime_moe_buffers(model)
    return model


@jaxtyped(typechecker=typechecker)
def forward(
    model: nn.Module,
    input_ids: Int[Tensor, "batch seq"],
    position_ids: Int[Tensor, "batch seq"],
    labels: Int[Tensor, "batch seq"] | None = None,
    temperature: Tensor | None = None,
    routed_experts: Int[Tensor, "batch seq layers topk"] | None = None,
    # Multimodal fields (Qwen3-VL)
    pixel_values: Float[Tensor, "num_patches patch_dim"] | None = None,
    image_grid_thw: Int[Tensor, "num_images 3"] | None = None,
    mm_token_type_ids: Int[Tensor, "batch seq"] | None = None,
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
        kwargs["mm_token_type_ids"] = mm_token_type_ids
    else:
        kwargs["position_ids"] = position_ids

    if routed_experts is not None:
        kwargs["routed_experts"] = routed_experts

    out = model(**kwargs)

    # PrimeLmOutput is a TypedDict (dict at runtime), HF outputs are dataclass-like objects
    if isinstance(out, dict):
        return cast_float_and_contiguous(out)

    return cast_float_and_contiguous(PrimeLmOutput(logits=out.logits))
