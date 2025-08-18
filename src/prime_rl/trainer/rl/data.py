from pathlib import Path
from typing import TypedDict

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from prime_rl.trainer.rl.config import FakeDataLoaderConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.utils import get_rollout_dir, wait_for_path


class MicroBatch(TypedDict):
    # Token level
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    advantages: Float[Tensor, "batch seq"]
    logprobs: Float[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]

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
            "loss_mask": torch.ones(self.micro_batch_size, self.seq_len, dtype=torch.bool),
        }


class DataLoader:
    """Loads serialized data from a data path written by the orchestrator."""

    def __init__(self, output_dir: Path, start_step: int):
        self.rollout_dir = get_rollout_dir(output_dir)
        self.current_step = start_step
        self.world = get_world()

    def get_rollout_path(self) -> Path:
        return self.rollout_dir / f"step_{self.current_step}" / f"rank_{self.world.rank}.pt"

    def wait_for_batch(self) -> None:
        wait_for_path(self.get_rollout_path())

    def get_batch(self) -> list[MicroBatch]:
        batches = torch.load(self.get_rollout_path())
        self.current_step += 1
        return batches
