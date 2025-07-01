import time
from pathlib import Path
from typing import TypedDict

import torch
from jaxtyping import Float, Int

from zeroband.training.config import FakeDataLoaderConfig
from zeroband.training.world import get_world


class MicroBatch(TypedDict):
    # Token level
    input_ids: Int[torch.Tensor, "micro_bs seq"]
    position_ids: Int[torch.Tensor, "micro_bs seq"]
    advantages: Float[torch.Tensor, "micro_bs seq"]
    logprobs: Float[torch.Tensor, "micro_bs seq_minus_1"]

    # Batch level
    temperature: float
    total_tokens: int


class FakeDataLoader:
    def __init__(self, config: FakeDataLoaderConfig):
        self.batch_size = config.batch_size
        self.micro_batch_size = config.micro_batch_size
        self.num_micro_batches = self.batch_size // self.micro_batch_size
        self.seq_len = config.seq_len

    def get_batch(self) -> list[MicroBatch]:
        return [self._get_micro_batch() for _ in range(self.num_micro_batches)]

    def _get_micro_batch(self) -> MicroBatch:
        return {
            "input_ids": torch.randint(0, 100, (self.micro_batch_size, self.seq_len)),
            "position_ids": torch.stack([torch.arange(self.seq_len)] * self.micro_batch_size, dim=0),
            "advantages": torch.randn(self.micro_batch_size, self.seq_len),
            "logprobs": torch.randn(self.micro_batch_size, self.seq_len - 1),
            "temperature": 1.0,
            "total_tokens": self.micro_batch_size * self.seq_len,
        }


class DataLoader:
    """Loads serialized data from a data path written by the orchestrator."""

    def __init__(self, data_path: Path, start_step: int):
        self.data_path = data_path
        self.current_step = start_step
        self.world = get_world()

    def get_batch(self) -> list[MicroBatch]:
        while True:
            # here adding step + 1 because orchestrator count step is offset by 1 bc of @mika
            step_path = self.data_path / f"step_{self.current_step + 1}" / f"rank_{self.world.rank}.pt"
            if step_path.exists():
                batches = torch.load(step_path)
                self.current_step += 1
                return batches
            # Prevent busy waiting
            time.sleep(0.01)
