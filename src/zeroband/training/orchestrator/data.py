import torch
from transformers import AutoTokenizer

from zeroband.training.data import MicroBatch


def prepare_sample(prompt: str, completion: str, advantage: float, seq_len: int, tokenizer: AutoTokenizer):
    input_token_ids = torch.tensor(tokenizer.encode(prompt))
    output_token_ids = torch.tensor(tokenizer.encode(completion))
    input_ids = torch.cat([input_token_ids, output_token_ids], dim=0)

    if len(input_ids) > seq_len:
        # Truncate the sequence to seq_len
        input_ids = input_ids[:seq_len]
        advantages = torch.cat(
            [torch.zeros(len(input_token_ids)), torch.tensor(advantage).repeat(len(output_token_ids)).float()]
        )
    else:
        # Pad the sequence to seq_len
        padding_len = seq_len - len(input_ids)
        pad_tokens = torch.full((padding_len,), tokenizer.pad_token_id)
        input_ids = torch.cat([input_ids, pad_tokens])
        advantages = torch.cat(
            [
                torch.zeros(len(input_token_ids)),
                torch.tensor(advantage).repeat(len(output_token_ids)).float(),
                torch.zeros(padding_len),
            ]
        )

    # TODO(Mika): Compute logprobs, if available
    logprobs = torch.ones_like(input_ids).float()
    position_ids = torch.arange(seq_len)

    return {
        "input_ids": input_ids,
        "advantages": advantages,
        "position_ids": position_ids,
        "logprobs": logprobs,
    }


def prepare_micro_batch(samples: list[MicroBatch], temperature: float):
    micro_batch = {}
    for key in ["input_ids", "advantages", "position_ids", "logprobs"]:
        micro_batch[key] = torch.stack([sample[key] for sample in samples], dim=0)

    micro_batch["total_tokens"] = sum([len(sample["input_ids"]) for sample in samples])
    micro_batch["temperature"] = temperature

    return micro_batch


def prepare_batch(
    prompts: list[str],
    completions: list[str],
    advantages: list[float],
    temperature: float,
    tokenizer: AutoTokenizer,
    batch_size: int,
    micro_batch_size: int,
    num_train_workers: int,
    seq_len: int,
) -> list[list[MicroBatch]]:
    assert len(prompts) == len(completions) == len(advantages), (
        "Prompts, completions, and advantages must have the same length"
    )
    batch_size = len(prompts)
    assert batch_size % (micro_batch_size * num_train_workers) == 0, "Batch size must be divisible by micro batch size"
    per_gpu_micro_batches = batch_size // (num_train_workers * micro_batch_size)

    all_batches = []
    for _ in range(num_train_workers):
        batches = []
        for _ in range(per_gpu_micro_batches):
            micro_batches = []
            for _ in range(micro_batch_size):
                sample = prepare_sample(prompts.pop(), completions.pop(), advantages.pop(), seq_len, tokenizer)
                micro_batches.append(sample)
            batches.append(prepare_micro_batch(micro_batches, temperature))

        all_batches.append(batches)

    return all_batches
