import torch
from typing import TypedDict

from jaxtyping import Bool, Float, Int
from transformers import AutoTokenizer
from torch import Tensor

from prime_rl.trainer.batch import Sample


class RLSample(Sample):
    position_ids: list[int]
    loss_mask: list[int]
    logprobs: list[float]
    advantages: list[float]

class RLBatch(TypedDict):
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]
    advantages: Float[Tensor, "batch seq"]
    logprobs: Float[Tensor, "batch seq"]

def collate_rl(samples: list[RLSample], seq_len: int, tokenizer: AutoTokenizer) -> RLBatch:
    """Truncates and pads samples to seq_len."""
    for sample in samples:
        if len(sample["input_ids"]) > seq_len:  # Truncate
            sample["input_ids"] = sample["input_ids"][:seq_len]
            sample["loss_mask"] = sample["loss_mask"][:seq_len]
            sample["position_ids"] = sample["position_ids"][:seq_len]
            sample["logprobs"] = sample["logprobs"][:seq_len]
            sample["advantages"] = sample["advantages"][:seq_len]
        if len(sample["input_ids"]) < seq_len:  # Pad
            num_pad_tokens = seq_len - len(sample["input_ids"])
            sample["input_ids"] += [tokenizer.pad_token_id] * num_pad_tokens
            sample["loss_mask"] += [0] * num_pad_tokens
            sample["position_ids"] += [0] * num_pad_tokens
            sample["logprobs"] += [0] * num_pad_tokens
            sample["advantages"] += [0] * num_pad_tokens

    # Stack tensors into tensors of size (batch_size, seq_len)
    batch_input_ids = torch.stack([torch.tensor(sample["input_ids"]) for sample in samples]).long()
    batch_position_ids = torch.stack([torch.tensor(sample["position_ids"]) for sample in samples]).long()
    batch_loss_mask = torch.stack([torch.tensor(sample["loss_mask"]) for sample in samples]).bool()
    batch_advantages = torch.stack([torch.tensor(sample["advantages"]) for sample in samples]).float()
    batch_logprobs = torch.stack([torch.tensor(sample["logprobs"]) for sample in samples]).float()

    return {
        "input_ids": batch_input_ids.contiguous(),
        "position_ids": batch_position_ids.contiguous(),
        "loss_mask": batch_loss_mask.contiguous(),
        "advantages": batch_advantages.contiguous(),
        "logprobs": batch_logprobs.contiguous(),
    }