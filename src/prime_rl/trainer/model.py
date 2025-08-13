import torch
import torch.distributed as dist
import torch.nn as nn
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor.parallel import parallelize_module
from torchtitan.distributed.expert_parallel import ExpertParallel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from prime_rl.trainer.config import ModelConfig


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


def get_tokenizer(config: ModelConfig) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config.name, trust_remote_code=config.trust_remote_code)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def setup_fsdp(model: nn.Module, config: ModelConfig):
    dist.init_process_group()
    assert dist.get_world_size() % config.ep == 0, "World size must be divisible by EP"
    fsdp_dim = dist.get_world_size() // config.ep
    world_mesh = dist.init_device_mesh("cuda", (fsdp_dim, config.ep), mesh_dim_names=("fsdp", "ep"))
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    # TODO: Do we need to include ep in hsdp mesh?
    hsdp_mesh = world_mesh["fsdp"]
    ep_mesh = world_mesh["ep"] if config.ep > 1 else None
    fsdp_mesh = world_mesh["fsdp"]

    for layer_id, transformer_block in enumerate(model.model.layers):
        if config.reshard_after_forward:
            layer_reshard_after_forward = layer_id < len(model.model.layers) - 1
        else:
            layer_reshard_after_forward = False
        if hasattr(transformer_block.mlp, "experts"):
            if ep_mesh is not None:
                parallelize_module(
                    module=transformer_block.mlp.experts,
                    device_mesh=ep_mesh,
                    parallelize_plan=ExpertParallel(),
                )
            fully_shard(
                transformer_block,
                mesh=fsdp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=layer_reshard_after_forward,
            )
        else:
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


def setup_ac(model: nn.Module, config: ModelConfig) -> None:
    if not config.ac:
        return
    for layer_id, transformer_block in model.model.layers.named_children():
        transformer_block = checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        model.model.layers.register_module(layer_id, transformer_block)


def setup_model(config: ModelConfig) -> nn.Module:
    model = get_model(config)
    setup_fsdp(model, config)
    setup_ac(model, config)
    if config.compile:
        model = torch.compile(model)
    # TODO: This should be type-hinted as FSDP version of the model
    return model


@jaxtyped(typechecker=typechecker)
def forward(
    model: nn.Module, input_ids: Int[Tensor, "batch seq"], position_ids: Int[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq vocab"]:
    return model(input_ids=input_ids, position_ids=position_ids).logits.float()
