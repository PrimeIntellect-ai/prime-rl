import torch
from transformers import AutoTokenizer
from typing import TypedDict

from jaxtyping import Bool, Int
from torch import Tensor

from prime_rl.trainer.batch import Sample

class SFTSample(Sample):
    input_ids: list[int]
    position_ids: list[int]
    loss_mask: list[int]

class SFTBatch(TypedDict):
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]
    target_ids: Int[Tensor, "batch seq"]

def collate(samples: list[SFTSample], seq_len: int, tokenizer: AutoTokenizer) -> SFTBatch:
    """Truncates and pads samples to seq_len."""
    seq_len += 1  # One more token because we lose one
    for sample in samples:
        if len(sample["input_ids"]) > seq_len:  # Truncate
            sample["input_ids"] = sample["input_ids"][:seq_len]
            sample["loss_mask"] = sample["loss_mask"][:seq_len]
            sample["position_ids"] = sample["position_ids"][:seq_len]
        if len(sample["input_ids"]) < seq_len:  # Pad
            num_pad_tokens = seq_len - len(sample["input_ids"])
            sample["input_ids"] += [tokenizer.pad_token_id] * num_pad_tokens
            sample["loss_mask"] += [0] * num_pad_tokens
            sample["position_ids"] += [0] * num_pad_tokens

    # Stack tensors into tensors of size (batch_size, seq_len)
    batch_input_ids = torch.stack([torch.tensor(sample["input_ids"]) for sample in samples]).long()
    batch_position_ids = torch.stack([torch.tensor(sample["position_ids"]) for sample in samples]).long()
    batch_loss_mask = torch.stack([torch.tensor(sample["loss_mask"]) for sample in samples]).bool()

    return {
        "input_ids": batch_input_ids[:, :-1].contiguous(),
        "target_ids": batch_input_ids[:, 1:].contiguous(),
        "position_ids": batch_position_ids[:, :-1].contiguous(),
        "loss_mask": batch_loss_mask[:, :-1].contiguous(),
    }
