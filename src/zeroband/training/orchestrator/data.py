import copy
from typing import Literal, TypedDict

import torch
from jaxtyping import Float, Int
from transformers import AutoTokenizer

from zeroband.training.data import MicroBatch


class Sample(TypedDict):
    input_ids: Int[torch.Tensor, "seq"]
    advantages: Float[torch.Tensor, "seq"]
    loss_mask: Int[torch.Tensor, "seq"]
    position_ids: Float[torch.Tensor, "seq"]
    logprobs: Float[torch.Tensor, "seq"]

    total_tokens: int


def prepare_sample(
    prompt: str, completion: str, advantage: float, max_seq_len: int, tokenizer: AutoTokenizer, pad: bool
) -> Sample:
    """
    Prepare a problem and pad it for training.
    Tokenize and
    """

    input_tokens = torch.tensor(tokenizer.encode(prompt))
    output_tokens = torch.tensor(tokenizer.encode(completion))

    inputs_ids = torch.cat([input_tokens, output_tokens], dim=0)

    loss_mask = torch.cat([torch.zeros(len(input_tokens)), torch.ones(len(output_tokens))], dim=0).int()
    advantages = torch.tensor(advantage).repeat(inputs_ids.shape[0]).float()
    position_ids = torch.arange(inputs_ids.shape[0]).float()
    logprobs = torch.ones_like(inputs_ids).float()  # todo add real logprobs
    total_tokens = inputs_ids.shape[0]  # total token should always be before padding

    if inputs_ids.shape[0] > max_seq_len:
        raise ValueError(f"Sequence length {inputs_ids.shape[0]} is greater than max sequence length {max_seq_len}")
        # not we should never truncate as it would create a really bad learning signal in a grpo style rl run.
        # we should make sure that we always train with at least the max sequence length as we are using for inference.

    elif pad:
        padding_len = max_seq_len - inputs_ids.shape[0]
        inputs_ids = torch.cat([inputs_ids, torch.full((padding_len,), tokenizer.pad_token_id)])
        loss_mask = torch.cat([loss_mask, torch.zeros(padding_len)]).int()
        advantages = torch.cat([advantages, torch.zeros(padding_len)]).float()
        position_ids = torch.cat([position_ids, torch.arange(padding_len)]).float()
        logprobs = torch.cat([logprobs, torch.ones(padding_len)]).float()

    return {
        "input_ids": inputs_ids,
        "advantages": advantages,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "logprobs": logprobs,
        "total_tokens": total_tokens,
    }


def prepare_micro_batch(samples: list[MicroBatch], temperature: float):
    micro_batch = {}

    for key in ["input_ids", "advantages", "loss_mask", "position_ids", "logprobs"]:
        micro_batch[key] = torch.stack([sample[key] for sample in samples], dim=0)

    micro_batch["temperature"] = temperature
    micro_batch["total_tokens"] = sum([sample["total_tokens"] for sample in samples])

    return micro_batch


def prepare_batch_padding(
    prompts: list[str],
    completions: list[str],
    advantages: list[float],
    temperature: float,
    tokenizer: AutoTokenizer,
    batch_size: int,
    micro_batch_size: int,
    seq_len: int,
    num_train_workers: int,
) -> list[list[MicroBatch]]:
    """
    Prepare a batch of problems for each GPU. Each batch is a list of micro batches.
    Each micro batch is shape [micro_bs, max_seq_len] and contains micro_bs samples that are padded to the max lenght
    """
    assert len(prompts) == len(completions) == len(advantages), (
        "Prompts, completions, and advantages must have the same length"
    )
    batch_size = len(prompts)

    assert batch_size % (micro_batch_size * num_train_workers) == 0, "Batch size must be divisible by micro batch size"
    per_gpu_micro_batches = batch_size // (num_train_workers * micro_batch_size)

    batches_per_gpu = []
    for _ in range(num_train_workers):
        batches = []
        for _ in range(per_gpu_micro_batches):
            micro_batches = []
            for _ in range(micro_batch_size):
                sample = prepare_sample(
                    prompts.pop(), completions.pop(), advantages.pop(), seq_len, tokenizer, pad=True
                )
                micro_batches.append(sample)
            batches.append(prepare_micro_batch(micro_batches, temperature))

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
    prompts: list[str],
    completions: list[str],
    advantages: list[float],
    temperature: float,
    tokenizer: AutoTokenizer,
    batch_size: int,
    micro_batch_size: int,
    seq_len: int,
    num_train_workers: int,
) -> list[list[MicroBatch]]:
    """
    Prepare a batch of problems for each GPU. Each batch is a list of micro batches.
    Each micro batch is shape [1, micro_bs * max_seq_len], the namber of sample is not fixed per micro batch.
    """
    assert len(prompts) == len(completions) == len(advantages), (
        "Prompts, completions, and advantages must have the same length"
    )

    max_seq_len = seq_len * micro_batch_size

    all_samples = [
        prepare_sample(prompt, completion, advantage, max_seq_len, tokenizer, pad=False)
        for prompt, completion, advantage in zip(prompts, completions, advantages)
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
    prompts: list[str],
    completions: list[str],
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
                prompts,
                completions,
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
                prompts,
                completions,
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
