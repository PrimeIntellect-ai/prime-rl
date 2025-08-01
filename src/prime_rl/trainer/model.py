from typing import TypeAlias

import torch
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
    LlamaForCausalLM,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
)

from prime_rl.trainer.config import ModelConfig

Model: TypeAlias = LlamaForCausalLM | Qwen2ForCausalLM | Qwen3ForCausalLM


def get_model(config: ModelConfig) -> Model:
    config_model = AutoConfig.from_pretrained(config.name, attn_implementation=config.attn)
    config_model.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=config.name, config=config_model)
    return model


def get_tokenizer(config: ModelConfig) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config.name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def setup_fsdp(model: Model, config: ModelConfig):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    for layer_id, transformer_block in enumerate(model.model.layers):
        if config.reshard_after_forward:
            layer_reshard_after_forward = layer_id < len(model.model.layers) - 1
        else:
            layer_reshard_after_forward = False
        fully_shard(
            transformer_block,
            mp_policy=mp_policy,
            reshard_after_forward=layer_reshard_after_forward,
        )
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=config.reshard_after_forward)


def reshard_module(model: nn.Module):
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def setup_ac(model: Model, config: ModelConfig) -> None:
    if not config.ac:
        return
    for layer_id, transformer_block in model.model.layers.named_children():
        transformer_block = checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        model.model.layers.register_module(layer_id, transformer_block)


def setup_model(config: ModelConfig) -> Model:
    model = get_model(config)
    setup_fsdp(model, config)
    setup_ac(model, config)
    if config.compile:
        model = torch.compile(model)
    # TODO: This should be type-hinted as FSDP version of the model
    return model


@jaxtyped(typechecker=typechecker)
def forward(
    model: Model, input_ids: Int[Tensor, "batch seq"], position_ids: Int[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq vocab"]:
    return model(input_ids=input_ids, position_ids=position_ids).logits.float()
