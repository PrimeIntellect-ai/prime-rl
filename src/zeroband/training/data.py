from typing import TypedDict

from jaxtyping import Float, Int
import torch


class BatchOutput(TypedDict):
    # token level
    input_ids: Int[torch.Tensor, "micro_bs seq"]
    advantages: Float[torch.Tensor, "micro_bs seq"]
    loss_mask: Int[torch.Tensor, "micro_bs seq"]
    position_ids: Int[torch.Tensor, "micro_bs seq"]
    logprobs: Float[torch.Tensor, "micro_bs seq_minus_1"]

    # batch level
    temperature: float
    total_tokens: int


class DataLoader:
    def __init__(self, max_seq_len: int, pad_token_id: int, micro_bs: int, batch_size: int):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.micro_bs = micro_bs
        self.batch_size = batch_size

    def get_batch(self) -> list[BatchOutput]:
        micro_batches = []
        for _ in range(self.batch_size // self.micro_bs):
            micro_batches.append(self._get_micro_batch())
        return micro_batches

    def _get_micro_batch(self) -> BatchOutput:
        return {
            "input_ids": torch.randint(0, 100, (self.micro_bs, self.max_seq_len)),
            "advantages": torch.randn(self.micro_bs, self.max_seq_len),
            "loss_mask": torch.randint(0, 2, (self.micro_bs, self.max_seq_len)),
            "position_ids": torch.stack([torch.arange(self.max_seq_len)] * self.micro_bs, dim=0),
            "logprobs": torch.randn(self.micro_bs, self.max_seq_len - 1),
            "temperature": 1.0,
            "total_tokens": self.micro_bs * self.max_seq_len,
        }
