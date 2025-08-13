from functools import partial
from typing import Iterable, TypedDict

import torch
from jaxtyping import Bool, Int
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from prime_rl.trainer.sft.config import BatchConfig
from prime_rl.trainer.sft.data import Sample


class Batch(TypedDict):
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]
    target_ids: Int[Tensor, "batch seq"]


class BatchDataset(Dataset):
    """A dataset wrapping a list of samples in a batch."""

    def __init__(self, samples: list[Sample], config: BatchConfig):
        self.config = config
        self.n = len(samples)
        self.samples = samples

    def get_num_samples(self) -> int:
        """The number of samples in the batch"""
        return self.n

    def __len__(self) -> int:
        return self.get_num_samples()

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]


class PackedBatchDataset(BatchDataset):
    """A dataset wrapping a list of samples in a batch."""

    def __init__(self, samples: list[Sample], config: BatchConfig):
        super().__init__(samples)
        self.samples = self._pack_samples(self.samples)

    def _pack_samples(self, samples: list[Sample]) -> list[Sample]:
        """Offline sample packing using `First Fit Decreasing` algorithm."""
        # Sort samples in reverse order of length
        sorted_samples = sorted(samples, key=lambda x: len(x["input_ids"]), reverse=True)

        # Create packed samples
        packed_samples: list[Sample] = []
        for sample in sorted_samples:
            # Try to find a packed sample that can fit this sequence
            packed_sample_found = False
            for packed_sample in packed_samples:
                # Check if current sample fits in packed sample
                if len(packed_sample["input_ids"]) + len(sample["input_ids"]) <= self.config.seq_len:
                    packed_sample["input_ids"].extend(sample["input_ids"])
                    packed_sample["loss_mask"].extend(sample["loss_mask"])
                    packed_sample["position_ids"].extend(sample["position_ids"])
                    packed_sample_found = True
                    break

            # If no suitable packed sample found, create a new packed sample
            if not packed_sample_found:
                packed_samples.append(sample)

        return packed_samples

    def get_num_samples(self) -> int:
        """The number of samples in the unpacked batch."""
        return self.n

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]


def collate(samples: list[Sample], seq_len: int, tokenizer: AutoTokenizer) -> Batch:
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


def setup_dataloader(samples: list[Sample], tokenizer: AutoTokenizer, config: BatchConfig) -> Iterable[Batch]:
    # Initialize padding collate function
    collate_fn = partial(collate, seq_len=config.seq_len, tokenizer=tokenizer)

    # Initialize rank-aware sampler
    batch_dataset = (
        PackedBatchDataset(samples, config) if config.collate_mode == "packing" else BatchDataset(samples, config)
    )
    sampler = DistributedSampler(batch_dataset, shuffle=config.shuffle, drop_last=True)
    return iter(DataLoader(batch_dataset, batch_size=config.micro_batch_size, collate_fn=collate_fn, sampler=sampler))
