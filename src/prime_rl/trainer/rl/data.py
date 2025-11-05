from pathlib import Path
from typing import TypedDict

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from prime_rl.trainer.rl.config import FakeDataLoaderConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_rollout_dir, sync_wait_for_path


class MicroBatch(TypedDict):
    # Token level
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    advantages: Float[Tensor, "batch seq"]
    inference_logprobs: Float[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]

    # Batch level
    temperature: float


class FakeDataLoader:
    def __init__(self, config: FakeDataLoaderConfig):
        self.batch_size = config.batch_size
        self.num_micro_batches = self.batch_size // get_world().world_size
        self.seq_len = config.seq_len
        self.generate_documents = config.generate_documents

    def wait_for_batch(self) -> None:
        return

    def get_batch(self) -> list[MicroBatch]:
        fn = self._get_micro_batch if not self.generate_documents else self._get_document_micro_batch
        return [fn() for _ in range(self.num_micro_batches)]

    def _get_document_micro_batch(self) -> MicroBatch:
        max_len = self.seq_len // 8
        total_len = 0

        input_ids = []
        position_ids = []
        advantages = []
        inference_logprobs = []
        loss_mask = []
        temperature = 1.0

        while total_len < self.seq_len - max_len:
            len = torch.randint(1, max_len // 4, (1,)).item() * 4
            total_len += len
            input_ids.append(torch.randint(0, 100, (len,)))
            position_ids.append(torch.arange(len))
            advantages.append(torch.randn(len))
            inference_logprobs.append(torch.randn(len))
            loss_mask.append(torch.ones(len, dtype=torch.bool))

        remaining_len = self.seq_len - total_len
        input_ids.append(torch.randint(0, 100, (remaining_len,)))
        position_ids.append(torch.arange(remaining_len))
        advantages.append(torch.randn(remaining_len))
        inference_logprobs.append(torch.randn(remaining_len))
        loss_mask.append(torch.ones(remaining_len, dtype=torch.bool))

        return {
            "input_ids": torch.cat(input_ids).unsqueeze(0),
            "position_ids": torch.cat(position_ids).unsqueeze(0),
            "advantages": torch.cat(advantages).unsqueeze(0),
            "inference_logprobs": torch.cat(inference_logprobs).unsqueeze(0),
            "loss_mask": torch.cat(loss_mask).unsqueeze(0),
            "temperature": temperature,
        }

    def _get_micro_batch(self) -> MicroBatch:
        return {
            "input_ids": torch.randint(
                0,
                100,
                (
                    1,
                    self.seq_len,
                ),
            ),
            "position_ids": torch.cat([torch.arange(self.seq_len)]).unsqueeze(0),
            "advantages": torch.randn(self.seq_len).unsqueeze(0),
            "inference_logprobs": torch.randn(self.seq_len).unsqueeze(0),
            "temperature": 1.0,
            "loss_mask": torch.ones(self.seq_len, dtype=torch.bool).unsqueeze(0),
        }


class DataLoader:
    """Loads serialized data from a data path written by the orchestrator."""

    def __init__(self, output_dir: Path, start_step: int, num_non_data_parallel_ranks: int):
        self.rollout_dir = get_rollout_dir(output_dir)
        self.current_step = start_step
        self.world = get_world()

        self.dp_rank = self.world.rank // num_non_data_parallel_ranks

    def get_rollout_path(self) -> Path:
        get_logger().debug(f"Getting rollout path for step {self.current_step} and rank {self.dp_rank}")
        return self.rollout_dir / f"step_{self.current_step}" / f"rank_{self.dp_rank}.pt"

    def wait_for_batch(self) -> None:
        sync_wait_for_path(self.get_rollout_path())

    def get_batch(self) -> list[MicroBatch]:
        batches = torch.load(self.get_rollout_path())
        self.current_step += 1
        return batches
