from pathlib import Path
from typing import TypedDict

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from prime_rl.trainer.rl.config import FakeDataLoaderConfig
from prime_rl.trainer.world import get_world
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
    def __init__(self, config: FakeDataLoaderConfig, num_non_data_parallel_ranks: int = 1):
        self.world = get_world()
        num_dp_ranks = self.world.world_size // num_non_data_parallel_ranks

        self.dp_rank = self.world.rank // num_non_data_parallel_ranks
        self.batch_size = config.batch_size
        self.num_micro_batches = self.batch_size // num_dp_ranks
        self.seq_len = config.seq_len

        self.generate_documents = config.generate_documents

    def wait_for_batch(self) -> None:
        return

    def get_batch(self) -> list[MicroBatch]:
        if not self.generate_documents:
            fn = self._get_micro_batch
        else:
            fn = self._get_document_micro_batch
        return [fn() for _ in range(self.num_micro_batches)]

    def _get_document_micro_batch(self) -> MicroBatch:
        total_seq_len = 0
        input_ids = []
        position_ids = []

        while total_seq_len < self.seq_len:
            seq_len = torch.randint(1, self.seq_len // 8, (1,)).item()
            seq_len = seq_len if seq_len % 2 == 0 else seq_len + 1

            total_seq_len += seq_len
            tmp_input_ids = torch.randint(0, 120000, (seq_len,)).long()
            tmp_position_ids = torch.arange(seq_len).long()

            input_ids.append(tmp_input_ids)
            position_ids.append(tmp_position_ids)

        input_ids = torch.cat(input_ids, dim=0)
        position_ids = torch.cat(position_ids, dim=0)
        loss_mask = torch.ones(input_ids.shape[0], dtype=torch.bool)
        advantages = torch.randn(input_ids.shape[0])
        inference_logprobs = torch.randn(input_ids.shape[0])

        return {
            "input_ids": input_ids.unsqueeze(0),
            "position_ids": position_ids.unsqueeze(0),
            "advantages": advantages.unsqueeze(0),
            "inference_logprobs": inference_logprobs.unsqueeze(0),
            "temperature": 1.0,
            "loss_mask": loss_mask.unsqueeze(0),
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

    def __init__(self, output_dir: Path, start_step: int, num_non_data_parallel_ranks: int = 1):
        self.rollout_dir = get_rollout_dir(output_dir)
        self.current_step = start_step
        self.world = get_world()

        self.dp_rank = self.world.rank // num_non_data_parallel_ranks

    def get_rollout_path(self) -> Path:
        return self.rollout_dir / f"step_{self.current_step}" / f"rank_{self.dp_rank}.pt"

    def wait_for_batch(self) -> None:
        sync_wait_for_path(self.get_rollout_path())

    def get_batch(self) -> list[MicroBatch]:
        batches = torch.load(self.get_rollout_path())
        self.current_step += 1
        return batches
