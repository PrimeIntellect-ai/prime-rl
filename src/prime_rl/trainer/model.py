import logging

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from torch import Tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.config import ActivationCheckpointConfig, CompileConfig, ModelConfig
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.utils.logger import get_logger

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


def get_load_balance_stats(model: nn.Module, reset_stats: bool = True) -> dict[str, Tensor | None]:
    per_layer_max_vio = []
    for transformer_block in model.model.layers:
        # This is necessary for models that have mixed dense layers
        if not hasattr(transformer_block.mlp, "tokens_per_expert"):
            continue
        tokens_per_expert = transformer_block.mlp.tokens_per_expert
        balanced_load = tokens_per_expert.mean()
        max_vio = (tokens_per_expert.max() - balanced_load) / balanced_load
        per_layer_max_vio.append(max_vio.item())
        if reset_stats:
            tokens_per_expert.zero_()
    if len(per_layer_max_vio) == 0:
        return {"max_vio": None}
    return {"max_vio": torch.tensor(per_layer_max_vio)}


def get_model(
    config: ModelConfig, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.bfloat16
) -> nn.Module:
    config_model = AutoConfig.from_pretrained(
        config.name, attn_implementation=config.attn, trust_remote_code=config.trust_remote_code
    )
    config_model.use_cache = False

    with device:
        model_cls = AutoLigerKernelForCausalLM if config.liger_kernel else AutoModelForCausalLM
        if device == torch.device("meta"):
            model = model_cls.from_config(config_model, trust_remote_code=config.trust_remote_code, dtype=dtype)
        else:
            model = model_cls.from_pretrained(
                pretrained_model_name_or_path=config.name,
                config=config_model,
                trust_remote_code=config.trust_remote_code,
                dtype=dtype,
            )

    assert model.lm_head.weight.dtype == dtype, (
        f"LM head dtype wasnt loaded correctly {model.lm_head.weight.dtype} != {dtype}"
    )
    return model


def setup_tokenizer(config: ModelConfig) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config.name, trust_remote_code=config.trust_remote_code)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def setup_fsdp(model: nn.Module, config: ModelConfig, parallel_dims: ParallelDims):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    # TODO: Support dp_replicate
    hsdp_mesh = parallel_dims.world_mesh["dp_shard_cp"]

    for transformer_block in model.model.layers:
        fully_shard(
            transformer_block,
            mesh=hsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=config.reshard_after_forward,
        )

    if hasattr(model, "config") and not model.config.tie_word_embeddings:
        # This optimization breaks weight tying
        fully_shard(
            model.model.embed_tokens,
            mesh=hsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=config.reshard_after_forward,
        )
        fully_shard(
            [model.lm_head, model.model.norm],
            mesh=hsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=False,
        )
    else:
        get_logger().warning("Model is tied word embeddings, so not doing the last layer not resharding optimization")

    fully_shard(model, mesh=hsdp_mesh, mp_policy=mp_policy, reshard_after_forward=config.reshard_after_forward)


def load_dcp_from_hf(model: nn.Module, config: ModelConfig):
    from huggingface_hub import snapshot_download
    from torch.distributed.checkpoint import DefaultLoadPlanner, HuggingFaceStorageReader

    path_snapshot = snapshot_download(repo_id=config.name, repo_type="model")
    model.to_empty(device="cuda")
    dcp.load(
        model.state_dict(),
        storage_reader=HuggingFaceStorageReader(path=path_snapshot),
        # Note: This allow is needed by weight tying but could cause silent issues
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )
    fix_model_post_empty(model)


def can_load_dcp_from_hf(model: nn.Module):
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

    # TODO: Init TT MoE buffers
    # I think .to_empty() on gpu by default fills 0 so we are ok but this might not be guaranteed behavior


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
    for layer_id in range(len(model.model.layers)):
        # Doing it in-place avoids mangled fqn which can break checkpoint loading
        model.model.layers[layer_id].compile(fullgraph=compile_config.fullgraph)
    get_logger().info(f"Compiled {len(model.model.layers)} layers (fullgraph={compile_config.fullgraph})")


def setup_model(config: ModelConfig, parallel_dims: ParallelDims) -> nn.Module:
    model = get_model(
        config,
        device=torch.device("meta" if config.load_using_meta else "cpu"),
        dtype=DTYPE_MAP[config.optimization_dtype],
    )
    if config.load_using_meta and not can_load_dcp_from_hf(model):
        model = get_model(config, device=torch.device("cpu"), dtype=DTYPE_MAP[config.optimization_dtype])

    # the right order is AC -> Compile -> FSDP
    if config.ac is not None:
        apply_ac(model, config.ac)
    if config.compile is not None:
        apply_compile(model, config.compile)

    setup_fsdp(model, config, parallel_dims)

    if config.load_using_meta and can_load_dcp_from_hf(model):
        load_dcp_from_hf(model, config)

    if config.log_signature:
        from prime_rl.utils.tensor_hashing import get_module_signature

        get_logger().info(f"model signature: {get_module_signature(model, compress=True)}")
    return model


@jaxtyped(typechecker=typechecker)
def forward(
    model: nn.Module, input_ids: Int[Tensor, "batch seq"], position_ids: Int[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq vocab"]:
    return model(input_ids=input_ids, position_ids=position_ids).logits
