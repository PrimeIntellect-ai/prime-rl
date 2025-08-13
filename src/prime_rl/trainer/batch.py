from dataclasses import dataclass
from functools import partial
from typing import Iterable, TypedDict

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from prime_rl.trainer.config import BatchConfig
from prime_rl.trainer.world import get_world

class SFTSample(TypedDict):
    input_ids: list[int]
    position_ids: list[int]
    loss_mask: list[int]

class RLSample(TypedDict):
    position_ids: list[int]
    loss_mask: list[int]
    logprobs: list[float]
    advantages: list[float]

class SFTBatch(TypedDict):
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]
    target_ids: Int[Tensor, "batch seq"]

class RLBatch(TypedDict):
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]
    advantages: Float[Tensor, "batch seq"]
    logprobs: Float[Tensor, "batch seq"]



class BatchDataset(Dataset):
    """A dataset wrapping a list of samples in a batch."""

    def __init__(self, samples: list[SFTSample | RLSample], config: BatchConfig):
        self.config = config
        self.samples = samples

    def get_num_samples(self) -> int:
        """The number of samples in the batch"""
        return len(self.samples)

    def get_num_tokens(self) -> int:
        """The number of tokens in the batch."""
        return sum(len(sample["input_ids"]) for sample in self.samples)

    def __len__(self) -> int:
        return self.get_num_samples()

    def __getitem__(self, index: int) -> SFTSample | RLSample:
        return self.samples[index]


class PackedBatchDataset(BatchDataset):
    """A dataset wrapping a list of samples in a batch."""

    def __init__(self, samples: list[SFTSample | RLSample], config: BatchConfig):
        super().__init__(samples, config)
        self.packed_samples = self._pack_samples(self.samples)

    def _pack_samples(self, samples: list[SFTSample | RLSample]) -> list[SFTSample | RLSample]:
        """Offline sample packing using `First Fit Decreasing` algorithm."""
        # Sort samples in reverse order of length
        sorted_samples = sorted(samples, key=lambda x: len(x["input_ids"]), reverse=True)

        # Create packed samples
        packed_samples: list[SFTSample | RLSample] = []
        for sample in sorted_samples:
            # Try to find a packed sample that can fit this sequence
            packed_sample_found = False
            for packed_sample in packed_samples:
                # Check if current sample fits in packed sample
                if len(packed_sample["input_ids"]) + len(sample["input_ids"]) <= self.config.seq_len:
                    for key in packed_sample.keys():
                        assert isinstance(packed_sample[key], list), f"Key {key} is not a list"
                        packed_sample[key].extend(sample[key])
                    packed_sample_found = True
                    break

            # If no suitable packed sample found, create a new packed sample
            if not packed_sample_found:
                packed_samples.append(sample)

        return packed_samples

    def __len__(self) -> int:
        world_size = get_world().world_size
        return len(self.packed_samples) // world_size * world_size # Ensure divisible by number of ranks

    def __getitem__(self, index: int) -> SFTSample | RLSample:
        return self.packed_samples[index]


def collate_sft(samples: list[SFTSample | RLSample], seq_len: int, tokenizer: AutoTokenizer) -> SFTBatch | RLBatch:
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

# TODO(Mika): Too much duplication here, can make nicer later
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

def prepare_batch(samples: list[SFTSample | RLSample], tokenizer: AutoTokenizer, config: BatchConfig, collate_fn) -> tuple[BatchDataset, Iterable[SFTBatch | RLBatch]]:
    """Returns the global batch dataset and an iterator over local micro batches."""
    # Initialize rank-aware sampler
    batch_dataset = PackedBatchDataset(samples, config) if config.collate_mode == "packing" else BatchDataset(samples, config)
    assert len(batch_dataset) % get_world().world_size == 0, "Batch size must be divisible by number of ranks"
    sampler = DistributedSampler(batch_dataset, drop_last=True)  # TODO(Mika): Check if we wanna drop or pad
    return batch_dataset, iter(DataLoader(batch_dataset, batch_size=config.micro_batch_size, collate_fn=collate_fn, sampler=sampler))
