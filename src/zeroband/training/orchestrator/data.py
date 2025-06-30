import torch
from transformers import AutoTokenizer

from zeroband.training.data import MicroBatch


def prepare_sample(
    prompt: str, completion: str, advantage: float, max_seq_len: int, tokenizer: AutoTokenizer, pad: bool
):
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

    assert config.collate_mode == "padding", "Padding mode is not supported for this collate mode"

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
    raise NotImplementedError("Packing mode is not implemented yet")
    # assert len(prompts) == len(completions) == len(advantages), (
    #     "Prompts, completions, and advantages must have the same length"
    # )

    # all_samples = [
    #     prepare_sample(prompt, completion, advantage, config.max_seq_len, tokenizer, pad=False)
    #     for prompt, completion, advantage in zip(prompts, completions, advantages)
    # ]

    # sorted_samples = sorted(all_samples, key=lambda x: x["total_tokens"], reverse=True)

    # ## we create bins
    # bins = []

    # for seq_len, sample in sorted_samples:
    #     # Try to find a bin that can fit this sequence
    #     bin_found = False
    #     for bin_idx, bin_content in enumerate(bins):
    #         # Calculate current bin length
    #         bin_len = sum(s["total_tokens"] for s in bin_content)
    #         # Check if sequence fits in this bin
    #         if bin_len + sample["total_tokens"] <= config.max_seq_len:
    #             bins[bin_idx].append(sample)
    #             bin_found = True
    #             break

    #     # If no suitable bin found, create a new bin
    #     if not bin_found:
    #         batches.append([sample])

    # return batches


def prepare_batch(
    prompts: list[str],
    completions: list[str],
    advantages: list[float],
    temperature: float,
    tokenizer: AutoTokenizer,
    config: TrainConfig,
) -> list[list[BatchOutput]]:
    """
    Prepare a batch of problems for each GPU. Each batch is a list of micro batches.
    """
    match config.collate_mode:
        case "padding":
            return prepare_batch_padding(prompts, completions, advantages, temperature, tokenizer, config)
        case "packing":
            raise NotImplementedError("Packing mode is not implemented yet")
        case _:
            raise ValueError(f"Invalid collate mode: {config.collate_mode}")
