from pathlib import Path
from typing import TypedDict

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.rl.config import DEFAULT_PACKER_TIMEOUT_SECONDS, DataLoaderConfig, FakeDataLoaderConfig
from prime_rl.trainer.rl.packer import Packer
from prime_rl.trainer.world import get_world
from prime_rl.transport import MicroBatch, MicroBatchReceiver, TransportConfigType, setup_micro_batch_receiver


class TensorMicroBatch(TypedDict):
    """A micro batch of data for training."""

    # Token level
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    advantages: Float[Tensor, "batch seq"]
    inference_logprobs: Float[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]

    # Batch level
    temperature: float


def micro_batch_to_tensor(micro_batch: MicroBatch) -> TensorMicroBatch:
    """Convert a MicroBatch (msgspec struct with lists) to a TensorMicroBatch (dict with tensors)."""
    return TensorMicroBatch(
        input_ids=torch.tensor(micro_batch.input_ids, dtype=torch.long).unsqueeze(0),
        position_ids=torch.tensor(micro_batch.position_ids, dtype=torch.long).unsqueeze(0),
        advantages=torch.tensor(micro_batch.advantages, dtype=torch.float).unsqueeze(0),
        inference_logprobs=torch.tensor(micro_batch.inference_logprobs, dtype=torch.float).unsqueeze(0),
        loss_mask=torch.tensor(micro_batch.loss_mask, dtype=torch.bool).unsqueeze(0),
        temperature=micro_batch.temperature if micro_batch.temperature is not None else 1.0,
    )


class FakeDataLoader:
    def __init__(self, config: FakeDataLoaderConfig, seq_len: int):
        self.batch_size = config.batch_size
        self.num_micro_batches = self.batch_size // get_world().world_size
        self.seq_len = seq_len

    def wait_for_batch(self) -> None:
        return

    def get_batch(self) -> list[TensorMicroBatch]:
        return [self._get_micro_batch() for _ in range(self.num_micro_batches)]

    def _get_micro_batch(self) -> TensorMicroBatch:
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

    def __init__(
        self,
        output_dir: Path,
        start_step: int,
        dp_world_size: int,
        seq_len: int,
        tokenizer: PreTrainedTokenizer,
        config: TransportConfigType,
        packer_timeout_seconds: int = DEFAULT_PACKER_TIMEOUT_SECONDS,
    ):
        self.world = get_world()

        if self.world.is_master:
            self.packer = Packer(
                dp_world_size=dp_world_size,
                seq_len=seq_len,
                tokenizer=tokenizer,
                config=config,
                start_step=start_step,
                timeout_seconds=packer_timeout_seconds,
            )

        self.receiver: MicroBatchReceiver = setup_micro_batch_receiver(output_dir, self.world.rank, start_step, config)

    def wait_for_batch(self) -> None:
        if self.world.is_master:
            self.packer.pack()
        self.receiver.wait()

    def get_batch(self) -> list[TensorMicroBatch]:
        micro_batches = self.receiver.receive()
        return [micro_batch_to_tensor(mb) for mb in micro_batches]
