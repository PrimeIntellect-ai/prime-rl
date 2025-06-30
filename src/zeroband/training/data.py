from pathlib import Path
from typing import TypedDict

import torch
from jaxtyping import Float, Int

from zeroband.training.logger import get_logger
from zeroband.training.world_info import get_world_info


class MicroBatch(TypedDict):
    # token level
    token_ids: Int[torch.Tensor, "micro_bs seq"]
    position_ids: Int[torch.Tensor, "micro_bs seq"]
    advantages: Float[torch.Tensor, "micro_bs seq"]
    loss_mask: Int[torch.Tensor, "micro_bs seq"]
    logprobs: Float[torch.Tensor, "micro_bs seq_minus_1"]

    # batch level
    temperature: float
    total_tokens: int


class FakeDataLoader:
    def __init__(self, max_seq_len: int, pad_token_id: int, micro_bs: int, batch_size: int):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.micro_bs = micro_bs
        self.batch_size = batch_size

    def get_batch(self) -> list[MicroBatch]:
        micro_batches = []
        for _ in range(self.batch_size // self.micro_bs):
            micro_batches.append(self._get_micro_batch())
        return micro_batches

    def _get_micro_batch(self) -> MicroBatch:
        return {
            "input_ids": torch.randint(0, 100, (self.micro_bs, self.max_seq_len)),
            "advantages": torch.randn(self.micro_bs, self.max_seq_len),
            "loss_mask": torch.randint(0, 2, (self.micro_bs, self.max_seq_len)),
            "position_ids": torch.stack([torch.arange(self.max_seq_len)] * self.micro_bs, dim=0),
            "logprobs": torch.randn(self.micro_bs, self.max_seq_len - 1),
            "temperature": 1.0,
            "total_tokens": self.micro_bs * self.max_seq_len,
        }


class DataLoader:
    """
    Simply load the data from the data path.
    """

    def __init__(self, data_path: Path, start_step: int):
        self.data_path = data_path
        self.world_info = get_world_info()

    def get_batch(self) -> list[MicroBatch]:
        get_logger().info(f"Loading data from path {self.data_path}")
        while True:
            # here adding step + 1 because orchestator count step is offset by 1 bc of @mika
            step_path = self.data_path / f"step_{self.current_step + 1}" / f"data_rank_{self.world_info.rank}.pt"
            if step_path.exists():
                batches = torch.load(step_path)
                self.current_step += 1
                return batches
