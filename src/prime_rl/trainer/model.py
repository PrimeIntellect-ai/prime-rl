import torch
import torch.distributed as dist
import torch.nn as nn
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.config import ActivationCheckpointConfig, ModelConfig


def get_model(config: ModelConfig) -> nn.Module:
    config_model = AutoConfig.from_pretrained(
        config.name, attn_implementation=config.attn, trust_remote_code=config.trust_remote_code
    )
    config_model.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.name,
        config=config_model,
        trust_remote_code=config.trust_remote_code,
    )

    return model


def setup_tokenizer(config: ModelConfig) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config.name, trust_remote_code=config.trust_remote_code)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def setup_fsdp(model: nn.Module, config: ModelConfig):
    dist.init_process_group(device_id=torch.device("cuda", torch.cuda.current_device()))
    assert dist.get_world_size() % config.ep == 0, "World size must be divisible by EP"
    fsdp_dim = dist.get_world_size() // config.ep
    world_mesh = dist.init_device_mesh("cuda", (fsdp_dim, config.ep), mesh_dim_names=("fsdp", "ep"))
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    # TODO: Lets not support EP for now.
    # (1) Im not sure how the hsdp mesh is supposed to work with EP.
    # (2) The EP implementation seems to have changes every week in TT and is unstable at the moment
    # (3) There doesnt seem to be significant MFU gains from EP in my benchmarks
    assert config.ep == 1, "EP is not supported for now"
    hsdp_mesh = world_mesh["fsdp"]
    for layer_id, transformer_block in enumerate(model.model.layers):
        if config.reshard_after_forward:
            layer_reshard_after_forward = layer_id < len(model.model.layers) - 1
        else:
            layer_reshard_after_forward = False
        fully_shard(
            transformer_block,
            mesh=hsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=layer_reshard_after_forward,
        )

    fully_shard(model, mesh=hsdp_mesh, mp_policy=mp_policy, reshard_after_forward=config.reshard_after_forward)


def reshard_module(model: nn.Module):
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def apply_ac(model: nn.Module, ac_config: ActivationCheckpointConfig):
    for layer_id, (layer_name, transformer_block) in enumerate(model.model.layers.named_children()):
        if layer_id % ac_config.freq == 0:
            transformer_block = checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        model.model.layers.register_module(layer_name, transformer_block)


def setup_model(config: ModelConfig) -> nn.Module:
    model = get_model(config)
    setup_fsdp(model, config)
    if config.ac is not None:
        apply_ac(model, config.ac)
    if config.compile:
        model = torch.compile(model)
    # TODO: This should be type-hinted as FSDP version of the model
    return model


@jaxtyped(typechecker=typechecker)
def forward(
    model: nn.Module, input_ids: Int[Tensor, "batch seq"], position_ids: Int[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq vocab"]:
    return model(input_ids=input_ids, position_ids=position_ids).logits.float()
