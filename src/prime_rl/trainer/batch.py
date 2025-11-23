from typing import TypedDict

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from prime_rl.utils.vf import Rollout


class BatchSample(TypedDict):
    input_ids: Int[Tensor, "seq"]
    position_ids: Int[Tensor, "seq"]
    loss_mask: Bool[Tensor, "seq"]
    advantages: Float[Tensor, "seq"]
    inference_logprobs: Float[Tensor, "seq"]
    lora_idx: int


class MicroBatch(TypedDict):
    # Token level
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    advantages: Float[Tensor, "batch seq"]
    inference_logprobs: Float[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]

    # Batch level
    temperature: float
    lora_cu_offsets: Int[Tensor, "n_loras"]


def prepare_sample(
    rollout: Rollout,
    seq_len: int,
    lora_idx: int,
) -> BatchSample:
    """
    Prepare a problem for sequence packing training.
    Tokenize and prepare tensors.
    """

    # Prepare prompt tokens
    prompt_token_ids = torch.tensor(rollout["prompt_ids"]).long()
    prompt_token_mask = torch.tensor(rollout["prompt_mask"]).long()

    # Prepare completion tokens
    completion_token_ids = torch.tensor(rollout["completion_ids"]).long()
    completion_token_mask = torch.tensor(rollout["completion_mask"]).long()

    # Prepare input_ids, loss_mask, position_ids, inference_logprobs, and advantages
    input_ids = torch.cat([prompt_token_ids, completion_token_ids]).long()
    loss_mask = torch.cat([prompt_token_mask, completion_token_mask]).bool()
    inference_logprobs = torch.cat(
        [torch.zeros(len(prompt_token_ids)), torch.tensor(rollout["completion_logprobs"])]
    ).float()
    position_ids = torch.arange(len(input_ids)).long()
    advantages = torch.tensor(rollout["advantage"]).repeat(len(input_ids)).float()

    if len(input_ids) > seq_len:
        # We should never truncate as it would create a really bad learning signal. Instead, always set the maximum sequence length
        # on the inference worker accordingly, e.g. by setting the `max_tokens` parameter.
        raise ValueError(
            f"Number of tokens {len(input_ids)} is greater than sequence length {seq_len}. This should not happen."
        )

    assert len(input_ids) == len(advantages) == len(loss_mask) == len(position_ids) == len(inference_logprobs), (
        f"input_ids: {len(input_ids)}, advantages: {len(advantages)}, loss_mask: {len(loss_mask)}, position_ids: {len(position_ids)}, inference_logprobs: {len(inference_logprobs)}"
    )
    return {
        "input_ids": input_ids,
        "advantages": advantages,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "inference_logprobs": inference_logprobs,
        "lora_idx": lora_idx,
    }
