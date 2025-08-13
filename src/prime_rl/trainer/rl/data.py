import random
from pathlib import Path

import torch

from prime_rl.trainer.rl.config import FakeDataLoaderConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.utils import get_rollout_dir, wait_for_path
from prime_rl.trainer.batch import RLSample


class FakeBatchLoader:
    def __init__(self, config: FakeDataLoaderConfig):
        self.config = config

    def wait_for_samples(self) -> None:
        return

    def get_samples(self) -> list[RLSample]:
        return [{
            "input_ids": [random.randint(0, 100) for _ in range(self.config.seq_len)],
            "position_ids": list(range(self.seq_len)),
            "advantages": [random.random() for _ in range(self.seq_len)],
            "logprobs": [random.random() for _ in range(self.seq_len)],
            "loss_mask": [1] * self.config.seq_len
        } for _ in range(self.batch_size)]


class BatchLoader:
    """Loads serialized data from a data path written by the orchestrator."""

    def __init__(self, outputs_dir: Path, start_step: int):
        self.rollout_dir = get_rollout_dir(outputs_dir)
        self.current_step = start_step

    def get_rollout_path(self) -> Path:
        return self.rollout_dir / f"step_{self.current_step}" / f"rollouts.pt"

    def wait_for_samples(self) -> None:
        wait_for_path(self.get_rollout_path())

    def get_samples(self) -> list[RLSample]:
        samples = torch.load(self.get_rollout_path())
        self.current_step += 1
        return samples