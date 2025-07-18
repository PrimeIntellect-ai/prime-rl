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
    return model(input_ids=input_ids, position_ids=position_ids).logits


@jaxtyped(typechecker=typechecker)
def shift_logits(logits: Float[Tensor, "batch seq vocab"]) -> Float[Tensor, "batch seq vocab"]:
    """Removes final token logits and adds a zero logit for the first token."""
    # We drop the last logit because it corresponds to the next token that will be sampled but is not here yet
    logits = logits[:, :-1, :]  # (B, L-1, V)
    # We add a zero logit for the first token, indicating that no probability is assigned to it
    logits = torch.cat([torch.zeros(logits.shape[0], 1, logits.shape[2]).to(logits.device), logits], dim=1)  # (B, L, V)
    return logits


@jaxtyped(typechecker=typechecker)
def selective_log_softmax(
    logits: Float[Tensor, "batch seq vocab"], index: Int[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq"]:
    """
    credits to https://github.com/huggingface/trl/blob/07cfe1677e552b7d5c92b7740e5b2f0b057661d8/trl/trainer/utils.py#L1659

    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


@jaxtyped(typechecker=typechecker)
def compute_logprobs(
    model: Model,
    input_ids: Int[Tensor, "batch seq"],
    position_ids: Int[Tensor, "batch seq"],
    temperature: float,
) -> Float[Tensor, "batch seq vocab"]:
    logits = forward(model, input_ids, position_ids).contiguous()
    shifted_logits = shift_logits(logits)
    shifted_logits = shifted_logits / temperature
    logprobs = selective_log_softmax(shifted_logits, input_ids)
    del logits, shifted_logits
    return logprobs
