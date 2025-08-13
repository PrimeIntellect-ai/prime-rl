from typing import Iterable, TypedDict

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from prime_rl.trainer.config import BatchConfig
from prime_rl.trainer.world import get_world

class Sample(TypedDict):
    num_tokens: int
    num_samples: int = 1

class BatchDataset(Dataset):
    """
    A dataset wrapping a list of samples in a batch with methods to get the
    total number of samples and tokens in the batch.
    """

    def __init__(self, samples: list[Sample], config: BatchConfig):
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

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]


class PackedBatchDataset(BatchDataset):
    """A dataset wrapping a list of packed samples in a batch."""

    def __init__(self, samples: list[Sample], config: BatchConfig):
        super().__init__(samples, config)
        self.packed_samples = self._pack_samples(self.samples)

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
                if packed_sample["num_tokens"] + sample["num_tokens"] <= self.config.seq_len:
                    packed_sample["num_tokens"] += sample["num_tokens"]
                    packed_sample["num_samples"] += sample["num_samples"]
                    remaining_keys = set(packed_sample.keys()) - {"num_tokens", "num_samples"}
                    for key in remaining_keys:
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

    def __getitem__(self, index: int) -> Sample:
        return self.packed_samples[index]



def prepare_batch(samples: list[Sample], config: BatchConfig, collate_fn) -> tuple[BatchDataset, Iterable]:
    """Returns the global batch dataset and an iterator over local micro batches."""
    # Initialize rank-aware sampler
    batch_dataset = PackedBatchDataset(samples, config) if config.collate_mode == "packing" else BatchDataset(samples, config)
    assert len(batch_dataset) % get_world().world_size == 0, "Batch size must be divisible by number of ranks"
    sampler = DistributedSampler(batch_dataset, drop_last=True)  # TODO(Mika): Check if we wanna drop or pad
    return batch_dataset, iter(DataLoader(batch_dataset, batch_size=config.micro_batch_size, collate_fn=collate_fn, sampler=sampler))
