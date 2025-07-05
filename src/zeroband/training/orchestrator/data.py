import copy
from typing import Literal, TypedDict

import torch
from jaxtyping import Float, Int
from transformers import AutoTokenizer

from zeroband.training.data import MicroBatch


class Sample(TypedDict):
    input_ids: Int[torch.Tensor, "seq"]
    position_ids: Int[torch.Tensor, "seq"]
    loss_mask: Int[torch.Tensor, "seq"]
    advantages: Float[torch.Tensor, "seq"]
    logprobs: Float[torch.Tensor, "seq_minus_1"]

    total_tokens: int


def prepare_sample(
    input_tokens: list[int],
    output_tokens: list[int],
    output_logprobs: list[float],
    advantage: float,
    seq_len: int,
    tokenizer: AutoTokenizer,
    pad: bool,
) -> Sample:
    """
    Prepare a problem and pad it for training.
    Tokenize and
    """

    input_token_ids = torch.tensor(input_tokens).long()
    output_token_ids = torch.tensor(output_tokens).long()
    input_ids = torch.cat([input_token_ids, output_token_ids]).long()
    total_tokens = input_ids.shape[0]
    logprobs = torch.cat([torch.zeros(len(input_tokens) - 1), torch.tensor(output_logprobs)]).float()
    loss_mask = torch.cat([torch.zeros(len(input_tokens)), torch.ones(len(output_tokens))]).long()
    position_ids = torch.arange(total_tokens).long()
    advantages = torch.tensor(advantage).repeat(total_tokens).float()

    if total_tokens > seq_len:
        # We should never truncate as it would create a really bad learning signal. Instead, always set the maximum sequence length
        # on the inference worker accordingly, e.g. by setting the `max_tokens` parameter.
        raise ValueError(
            f"Number of tokens {total_tokens} is greater than sequence length {seq_len}. This should not happen."
        )

    # Pad the sequence to the sequence length
    if pad:
        num_padding_tokens = seq_len - total_tokens
        input_ids = torch.cat([input_ids, torch.full((num_padding_tokens,), tokenizer.pad_token_id)])
        loss_mask = torch.cat([loss_mask, torch.zeros(num_padding_tokens)]).long()
        position_ids = torch.cat([position_ids, torch.zeros(num_padding_tokens)]).long()
        logprobs = torch.cat([logprobs, torch.zeros(num_padding_tokens)]).float()
        advantages = torch.cat([advantages, torch.zeros(num_padding_tokens)]).float()

    assert len(input_ids) == len(advantages) == len(loss_mask) == len(position_ids) == len(logprobs) + 1
    return {
        "input_ids": input_ids,
        "advantages": advantages,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "logprobs": logprobs,
        "total_tokens": total_tokens,
    }


def prepare_micro_batch(samples: list[MicroBatch], temperature: float):
    micro_batch = {}

    for key in ["input_ids", "advantages", "loss_mask", "logprobs", "position_ids"]:
        micro_batch[key] = torch.stack([sample[key] for sample in samples], dim=0)

    micro_batch["temperature"] = temperature
    micro_batch["total_tokens"] = sum([sample["total_tokens"] for sample in samples])

    return micro_batch


def prepare_batch_padding(
    input_tokens: list[list[int]],
    output_tokens: list[list[int]],
    output_logprobs: list[list[float]],
    advantages: list[float],
    temperature: float,
    tokenizer: AutoTokenizer,
    batch_size: int,
    micro_batch_size: int,
    seq_len: int,
    num_train_workers: int,
) -> list[list[MicroBatch]]:
    input_tokens = copy.deepcopy(input_tokens)
    output_tokens = copy.deepcopy(output_tokens)
    output_logprobs = copy.deepcopy(output_logprobs)
    advantages = copy.deepcopy(advantages)

    """
    Prepare a batch of problems for each GPU. Each batch is a list of micro batches.
    Each micro batch is shape [micro_bs, max_seq_len] and contains micro_bs samples that are padded to the max lenght
    """
    assert len(input_tokens) == len(output_tokens) == len(output_logprobs) == len(advantages), (
        "input_tokens, output_tokens, output_logprobs, and advantages must have the same length"
    )
    batch_size = len(input_tokens)

    assert batch_size % (micro_batch_size * num_train_workers) == 0, "Batch size must be divisible by micro batch size"
    per_gpu_micro_batches = batch_size // (num_train_workers * micro_batch_size)

    batches_per_gpu = []
    for _ in range(num_train_workers):
        batches = []
        for _ in range(per_gpu_micro_batches):
            micro_batches = []
            for _ in range(micro_batch_size):
                sample = prepare_sample(
                    input_tokens.pop(),
                    output_tokens.pop(),
                    output_logprobs.pop(),
                    advantages.pop(),
                    seq_len,
                    tokenizer,
                    pad=True,
                )
                print(sample["position_ids"].dtype)
                micro_batches.append(sample)
            print(micro_batches[0]["position_ids"].dtype)
            batches.append(prepare_micro_batch(micro_batches, temperature))
        print(batches[0][0]["position_ids"].dtype)

        batches_per_gpu.append(batches)

    return batches_per_gpu


def packed_samples_into_micro_bs(samples: list[Sample], max_seq_len: int) -> list[list[Sample]]:
    """
    Pack samples into micro_batch efficiently.
    We follow the First Fit Decreasing algorithm to pack the samples into bins and minimize potential padding while never truncating.
    """
    sorted_samples = sorted(samples, key=lambda x: x["total_tokens"], reverse=True)

    ## we create bins
    micro_batches = []

    for sample in sorted_samples:
        # Try to find a bin that can fit this sequence
        bin_found = False
        for bin_idx, bin_content in enumerate(micro_batches):
            # Calculate current bin length
            bin_len = sum(s["total_tokens"] for s in bin_content)
            # Check if sequence fits in this bin
            if bin_len + sample["total_tokens"] <= max_seq_len:
                micro_batches[bin_idx].append(sample)
                bin_found = True
                break

        # If no suitable bin found, create a new bin
        if not bin_found:
            micro_batches.append([sample])

    return micro_batches


def prepare_micro_batch_packing(samples: list[Sample], max_seq_len: int, temperature: float) -> MicroBatch:
    """
    Prepare a micro batch for packing mode. take multi sample and return a batch of shape [1, micro_bs * max_seq_len].
    Would additionally pad the batch to the max sequence length.
    """
    micro_batch = {}
    assert sum([sample["total_tokens"] for sample in samples]) <= max_seq_len, (
        "Total tokens of samples is greater than max sequence length"
    )

    for key in ["input_ids", "advantages", "loss_mask", "position_ids", "logprobs"]:
        micro_batch[key] = torch.cat([sample[key] for sample in samples], dim=0).unsqueeze(0)

    micro_batch["temperature"] = temperature
    micro_batch["total_tokens"] = sum([sample["total_tokens"] for sample in samples])

    return micro_batch


def prepare_batch_packing(
    input_tokens: list[list[int]],
    output_tokens: list[list[int]],
    output_logprobs: list[list[float]],
    advantages: list[float],
    temperature: float,
    tokenizer: AutoTokenizer,
    batch_size: int,
    micro_batch_size: int,
    seq_len: int,
    num_train_workers: int,
) -> list[list[MicroBatch]]:
    input_tokens = copy.deepcopy(input_tokens)
    output_tokens = copy.deepcopy(output_tokens)
    output_logprobs = copy.deepcopy(output_logprobs)
    advantages = copy.deepcopy(advantages)

    """
    Prepare a batch of problems for each GPU. Each batch is a list of micro batches.
    Each micro batch is shape [1, micro_bs * max_seq_len], the namber of sample is not fixed per micro batch.
    """
    assert len(input_tokens) == len(output_tokens) == len(output_logprobs) == len(advantages), (
        "input_tokens, output_tokens, output_logprobs, and advantages must have the same length"
    )

    max_seq_len = seq_len * micro_batch_size

    all_samples = [
        prepare_sample(input_token, output_token, output_logprob, advantage, max_seq_len, tokenizer, pad=False)
        for input_token, output_token, output_logprob, advantage in zip(
            input_tokens, output_tokens, output_logprobs, advantages
        )
    ]

    micro_batches_list = packed_samples_into_micro_bs(all_samples, max_seq_len)
    micro_batches = [
        prepare_micro_batch_packing(micro_batch, max_seq_len, temperature) for micro_batch in micro_batches_list
    ]

    num_padding_batch = num_train_workers - len(micro_batches) % num_train_workers

    # because of fsdp we need to make sure that each data ran has the same number of micro batches otherwise training will hang.
    # We create fake micro batches to fill the gap with real data but zero advantages, they would not contribute to the loss.
    if num_padding_batch > 0:
        padded_batch = copy.deepcopy(micro_batches[0])
        padded_batch["advantages"] = torch.zeros_like(padded_batch["advantages"])
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


def prepare_batch(
    input_tokens: list[list[int]],
    output_tokens: list[list[int]],
    output_logprobs: list[list[float]],
    advantages: list[float],
    temperature: float,
    tokenizer: AutoTokenizer,
    batch_size: int,
    micro_batch_size: int,
    seq_len: int,
    num_train_workers: int,
    collate_mode: Literal["packing", "padding"],
) -> list[list[MicroBatch]]:
    """
    Prepare a batch of problems for each GPU. Each batch is a list of micro batches.
    """
    match collate_mode:
        case "padding":
            return prepare_batch_padding(
                input_tokens,
                output_tokens,
                output_logprobs,
                advantages,
                temperature,
                tokenizer,
                batch_size,
                micro_batch_size,
                seq_len,
                num_train_workers,
            )
        case "packing":
            return prepare_batch_packing(
                input_tokens,
                output_tokens,
                output_logprobs,
                advantages,
                temperature,
                tokenizer,
                batch_size,
                micro_batch_size,
                seq_len,
                num_train_workers,
            )
        case _:
            raise ValueError(f"Invalid collate mode: {collate_mode}")
