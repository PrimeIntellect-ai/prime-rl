from pathlib import Path
from typing import TypedDict

import torch
from jaxtyping import Float, Int

from prime_rl.trainer.config import FakeDataLoaderConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.utils import wait_for_path


class MicroBatch(TypedDict):
    # Token level
    input_ids: Int[torch.Tensor, "micro_bs seq"]
    position_ids: Int[torch.Tensor, "micro_bs seq"]
    advantages: Float[torch.Tensor, "micro_bs seq"]
    logprobs: Float[torch.Tensor, "micro_bs seq"]
    loss_mask: Int[torch.Tensor, "micro_bs seq"]

    # Batch level
    temperature: float
    total_tokens: int


class FakeDataLoader:
    def __init__(self, config: FakeDataLoaderConfig):
        self.batch_size = config.batch_size
        self.micro_batch_size = config.micro_batch_size
        self.num_micro_batches = self.batch_size // self.micro_batch_size // get_world().world_size
        self.seq_len = config.seq_len

    def wait_for_batch(self) -> None:
        return

    def get_batch(self) -> list[MicroBatch]:
        return [self._get_micro_batch() for _ in range(self.num_micro_batches)]

    def _get_micro_batch(self) -> MicroBatch:
        return {
            "input_ids": torch.randint(0, 100, (self.micro_batch_size, self.seq_len)),
            "position_ids": torch.stack([torch.arange(self.seq_len)] * self.micro_batch_size, dim=0),
            "advantages": torch.randn(self.micro_batch_size, self.seq_len),
            "logprobs": torch.randn(self.micro_batch_size, self.seq_len),
            "temperature": 1.0,
            "loss_mask": torch.ones(self.micro_batch_size, self.seq_len, dtype=torch.int32),
        }


class DataLoader:
    """Loads serialized data from a data path written by the orchestrator."""

    def __init__(self, data_path: Path, start_step: int):
        self.data_path = data_path
        self.current_step = start_step
        self.world = get_world()

    def get_batch_path(self) -> Path:
        return self.data_path / f"step_{self.current_step}" / f"rank_{self.world.rank}.pt"

    def wait_for_batch(self) -> None:
        wait_for_path(self.get_batch_path())

    def get_batch(self) -> list[MicroBatch]:
        batches = torch.load(self.get_batch_path())
        self.current_step += 1
        return batches
