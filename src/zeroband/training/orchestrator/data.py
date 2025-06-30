import torch
from transformers import AutoTokenizer

from zeroband.training.data import MicroBatch


def prepare_sample(prompt: str, completion: str, advantage: float, max_seq_len: int, tokenizer: AutoTokenizer):
    """
    Prepare a problem and pad it for training.
    Tokenize and
    """

    input_tokens = torch.tensor(tokenizer.encode(prompt))
    output_tokens = torch.tensor(tokenizer.encode(completion))

    inputs_ids = torch.cat([input_tokens, output_tokens], dim=0)
    total_tokens = inputs_ids.shape[0]

    loss_mask = torch.cat([torch.zeros(len(input_tokens)), torch.ones(len(output_tokens))], dim=0).int()

    if inputs_ids.shape[0] > max_seq_len:
        inputs_ids = inputs_ids[:max_seq_len]
        loss_mask = loss_mask[:max_seq_len].int()
        advantages = torch.tensor(advantage).repeat(max_seq_len).float()
    else:
        padding_len = max_seq_len - inputs_ids.shape[0]
        inputs_ids = torch.cat([inputs_ids, torch.full((padding_len,), tokenizer.pad_token_id)])
        loss_mask = torch.cat([loss_mask, torch.zeros(padding_len)]).int()
        advantages = torch.tensor(advantage).repeat(inputs_ids.shape[0]).float()
        advantages = torch.cat([advantages, torch.zeros(padding_len)])

    advantages = torch.tensor(advantage).repeat(max_seq_len).float()

    logprobs = torch.ones_like(inputs_ids).float()
    position_ids = torch.arange(max_seq_len)

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


def prepare_batch(
    prompts: list[str],
    completions: list[str],
    advantages: list[float],
    temperature: float,
    tokenizer: AutoTokenizer,
    micro_bs: int,
    max_seq_len: int,
    n_data_ranks: int,
) -> list[list[MicroBatch]]:
    """
    Prepare a batch of problems for each GPU. Each batch is a list of micro batches.
    """

    assert len(prompts) == len(completions) == len(advantages), (
        "Prompts, completions, and advantages must have the same length"
    )
    batch_size = len(prompts)

    assert batch_size % (micro_bs * n_data_ranks) == 0, "Batch size must be divisible by micro batch size"
    per_gpu_micro_batches = batch_size // (n_data_ranks * micro_bs)

    batches_per_gpu = []
    for _ in range(n_data_ranks):
        batches = []
        for _ in range(per_gpu_micro_batches):
            micro_batches = []
            for _ in range(micro_bs):
                sample = prepare_sample(prompts.pop(), completions.pop(), advantages.pop(), max_seq_len, tokenizer)
                micro_batches.append(sample)
            batches.append(prepare_micro_batch(micro_batches, temperature))

        batches_per_gpu.append(batches)

    return batches_per_gpu
