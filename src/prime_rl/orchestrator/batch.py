import copy
from typing import TypedDict

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.orchestrator.buffer import Rollout
from prime_rl.trainer.rl.data import MicroBatch


class BatchSample(TypedDict):
    input_ids: Int[Tensor, "seq"]
    position_ids: Int[Tensor, "seq"]
    loss_mask: Bool[Tensor, "seq"]
    advantages: Float[Tensor, "seq"]
    logprobs: Float[Tensor, "seq"]


def prepare_sample(
    rollout: Rollout,
    seq_len: int,
    tokenizer: PreTrainedTokenizer,
) -> BatchSample:
    """
    Prepare a problem for sequence packing training.
    Tokenize and prepare tensors.
    """

    # Prepare prompt tokens
    prompt_token_ids = torch.tensor(rollout.prompt_tokens).long()
    prompt_token_mask = torch.tensor(rollout.prompt_mask).long()

    # Prepare completion tokens
    completion_token_ids = torch.tensor(rollout.completion_tokens).long()
    completion_token_mask = torch.tensor(rollout.completion_mask).long()

    # Prepare input_ids, loss_mask, position_ids, logprobs, and advantages
    input_ids = torch.cat([prompt_token_ids, completion_token_ids]).long()
    loss_mask = torch.cat([prompt_token_mask, completion_token_mask]).bool()
    logprobs = torch.cat([torch.zeros(len(prompt_token_ids)), torch.tensor(rollout.completion_logprobs)]).float()
    position_ids = torch.arange(len(input_ids)).long()
    advantages = torch.tensor(rollout.advantage).repeat(len(input_ids)).float()

    if len(input_ids) > seq_len:
        # We should never truncate as it would create a really bad learning signal. Instead, always set the maximum sequence length
        # on the inference worker accordingly, e.g. by setting the `max_tokens` parameter.
        raise ValueError(
            f"Number of tokens {len(input_ids)} is greater than sequence length {seq_len}. This should not happen."
        )

    assert len(input_ids) == len(advantages) == len(loss_mask) == len(position_ids) == len(logprobs), (
        f"input_ids: {len(input_ids)}, advantages: {len(advantages)}, loss_mask: {len(loss_mask)}, position_ids: {len(position_ids)}, logprobs: {len(logprobs)}"
    )
    return {
        "input_ids": input_ids,
        "advantages": advantages,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "logprobs": logprobs,
    }


def prepare_micro_batch(samples: list[MicroBatch], temperature: float):
    micro_batch = {}

    for key in ["input_ids", "advantages", "loss_mask", "logprobs", "position_ids"]:
        micro_batch[key] = torch.stack([sample[key] for sample in samples], dim=0)

    micro_batch["temperature"] = temperature

    return micro_batch


def packed_samples_into_micro_bs(samples: list[BatchSample], max_seq_len: int) -> list[list[BatchSample]]:
    """
    Pack samples into micro_batch efficiently.
    We follow the First Fit Decreasing algorithm to pack the samples into bins and minimize potential padding while never truncating.
    """
    sorted_samples = sorted(samples, key=lambda x: len(x["input_ids"]), reverse=True)

    ## we create bins
    micro_batches = []

    for sample in sorted_samples:
        # Try to find a bin that can fit this sequence
        bin_found = False
        for bin_idx, bin_content in enumerate(micro_batches):
            # Calculate current bin length
            bin_len = sum(len(s["input_ids"]) for s in bin_content)
            # Check if sequence fits in this bin
            if bin_len + len(sample["input_ids"]) <= max_seq_len:
                micro_batches[bin_idx].append(sample)
                bin_found = True
                break

        # If no suitable bin found, create a new bin
        if not bin_found:
            micro_batches.append([sample])

    return micro_batches


def prepare_micro_batch_packing(samples: list[BatchSample], max_seq_len: int, temperature: float) -> MicroBatch:
    """
    Prepare a micro batch for packing mode. take multi sample and return a batch of shape [1, micro_bs * max_seq_len].
    Would additionally pad the batch to the max sequence length.
    """
    micro_batch = {}
    assert sum([len(sample["input_ids"]) for sample in samples]) <= max_seq_len, (
        "Total tokens of samples is greater than max sequence length"
    )

    for key in ["input_ids", "advantages", "loss_mask", "position_ids", "logprobs"]:
        micro_batch[key] = torch.cat([sample[key] for sample in samples], dim=0).unsqueeze(0)

    micro_batch["temperature"] = temperature

    return micro_batch


def prepare_batch(
    rollouts: list[Rollout],
    temperature: float,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    micro_batch_size: int,
    seq_len: int,
    num_train_workers: int,
) -> list[list[MicroBatch]]:
    """
    Prepare a batch of problems for each GPU. Each batch is a list of micro batches.
    Each micro batch is shape [1, micro_bs * max_seq_len], the namber of sample is not fixed per micro batch.
    """
    rollouts = copy.deepcopy(rollouts)
    max_seq_len = seq_len * micro_batch_size

    all_samples = [
        prepare_sample(
            rollout,
            max_seq_len,
            tokenizer,
        )
        for rollout in rollouts
    ]

    micro_batches_list = packed_samples_into_micro_bs(all_samples, max_seq_len)
    micro_batches = [
        prepare_micro_batch_packing(micro_batch, max_seq_len, temperature) for micro_batch in micro_batches_list
    ]

    num_padding_batch = num_train_workers - len(micro_batches) % num_train_workers

    # because of fsdp we need to make sure that each data ran has the same number of micro batches otherwise training will hang.
    # We create fake micro batches to fill the gap with real data but zero advantages, they would not contribute to the loss.
    if num_train_workers > 1 and num_padding_batch > 0:
        padded_batch = copy.deepcopy(micro_batches[0])
        padded_batch["advantages"] = torch.zeros_like(padded_batch["advantages"])
        padded_batch["loss_mask"] = torch.zeros_like(padded_batch["loss_mask"], dtype=torch.bool)
        micro_batches.extend([padded_batch for _ in range(num_padding_batch)])

    assert len(micro_batches) % num_train_workers == 0, (
        "Number of micro batches is not divisible by number of data ranks"
    )

    per_gpu_micro_batches = len(micro_batches) // num_train_workers
    batches_per_gpu = []
    for _ in range(num_train_workers):
        batches = []
        for _ in range(per_gpu_micro_batches):
            batches.append(micro_batches.pop(0))
        batches_per_gpu.append(batches)

    return batches_per_gpu
