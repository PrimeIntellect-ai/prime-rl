from pathlib import Path
from typing import TypedDict

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.rl.config import FakeDataLoaderConfig
from prime_rl.trainer.rl.packer import BasePacker, setup_packer
from prime_rl.trainer.runs import get_runs
from prime_rl.trainer.world import get_world
from prime_rl.transport import MicroBatch, MicroBatchReceiver, TransportConfigType, setup_micro_batch_receiver


class TensorMicroBatch(TypedDict):
    """A micro batch of data for training."""

    # Token level
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    advantages: Float[Tensor, "batch seq"]
    inference_logprobs: Float[Tensor, "batch seq"]
    teacher_logprobs: Float[Tensor, "batch seq"] | None
    loss_mask: Bool[Tensor, "batch seq"]

    # Batch level
    temperature: float
    lora_num_tokens: Int[Tensor, "n_loras"]


class FakeDataLoader:
    def __init__(self, config: FakeDataLoaderConfig, seq_len: int, dp_world_size: int):
        self.world = get_world()
        self.dp_world_size = dp_world_size
        self.non_dp_world_size = self.world.world_size // self.dp_world_size
        self.dp_rank = self.world.rank // self.non_dp_world_size

        self.batch_size = config.batch_size
        self.num_micro_batches = self.batch_size // self.dp_world_size
        self.seq_len = seq_len
        self.generate_samples = config.generate_samples
        self.batch_counter = 0
        self.runs = get_runs()

    def wait_for_batch(self) -> None:
        return

    def get_batch(self) -> list[TensorMicroBatch]:
        if not self.generate_samples:
            get_micro_batch_fn = self._get_micro_batch
        else:
            get_micro_batch_fn = self._get_sample_micro_batch

        # This is a pretty ugly hack to ensure that all CP ranks in a data parallel group receive the same micro batch.
        micro_batches = []
        for micro_batch_idx in range(self.num_micro_batches):
            seed = self.dp_rank * 1000000 + self.batch_counter * 1000 + micro_batch_idx
            generator = torch.Generator().manual_seed(seed)
            micro_batches.append(get_micro_batch_fn(generator))

        self.batch_counter += 1
        return micro_batches

    def _get_sample_micro_batch(self, generator: torch.Generator) -> TensorMicroBatch:
        total_seq_len = 0
        input_ids = []
        position_ids = []

        while total_seq_len < self.seq_len:
            # Generate reasonably long documents
            seq_len_to_generate = torch.randint(1, self.seq_len // 8, (1,), generator=generator).item()
            if seq_len_to_generate + total_seq_len > self.seq_len:
                seq_len_to_generate = self.seq_len - total_seq_len
            total_seq_len += seq_len_to_generate
            tmp_input_ids = torch.randint(0, 120000, (seq_len_to_generate,), generator=generator).long()
            tmp_position_ids = torch.arange(seq_len_to_generate).long()

            input_ids.append(tmp_input_ids)
            position_ids.append(tmp_position_ids)

        input_ids = torch.cat(input_ids, dim=0)
        position_ids = torch.cat(position_ids, dim=0)
        loss_mask = torch.ones(input_ids.shape[0], dtype=torch.bool)
        advantages = torch.randn(input_ids.shape[0], generator=generator)
        inference_logprobs = torch.randn(input_ids.shape[0], generator=generator)
        lora_num_tokens = torch.zeros(self.runs.max_runs, dtype=torch.int32)
        lora_num_tokens[0] = input_ids.shape[0]

        return {
            "input_ids": input_ids.unsqueeze(0),
            "position_ids": position_ids.unsqueeze(0),
            "advantages": advantages.unsqueeze(0),
            "inference_logprobs": inference_logprobs.unsqueeze(0),
            "teacher_logprobs": None,
            "temperature": 1.0,
            "loss_mask": loss_mask.unsqueeze(0),
            "lora_num_tokens": lora_num_tokens,
        }

    def _get_micro_batch(self, generator: torch.Generator) -> TensorMicroBatch:
        lora_num_tokens = torch.zeros(self.runs.max_runs, dtype=torch.int32)
        lora_num_tokens[0] = self.seq_len
        return {
            "input_ids": torch.randint(
                0,
                100,
                (
                    1,
                    self.seq_len,
                ),
                generator=generator,
            ),
            "position_ids": torch.cat([torch.arange(self.seq_len)]).unsqueeze(0),
            "advantages": torch.randn(self.seq_len, generator=generator).unsqueeze(0),
            "inference_logprobs": torch.randn(self.seq_len, generator=generator).unsqueeze(0),
            "teacher_logprobs": None,
            "temperature": 1.0,
            "loss_mask": torch.ones(self.seq_len, dtype=torch.bool).unsqueeze(0),
            "lora_num_tokens": lora_num_tokens,
        }


class DataLoader:
    """Loads serialized data from a data path written by the orchestrator."""

    def __init__(
        self,
        output_dir: Path,
        start_step: int,
        dp_world_size: int,
        seq_len: int,
        pad_to_multiple_of: int,
        tokenizer: PreTrainedTokenizer,
        config: TransportConfigType,
    ):
        self.world = get_world()

        if self.world.is_master:
            self.packer: BasePacker = setup_packer(
                dp_world_size=dp_world_size,
                seq_len=seq_len,
                tokenizer=tokenizer,
                transport_config=config,
                pad_to_multiple_of=pad_to_multiple_of,
                start_step=start_step,
            )

        non_dp_world_size = self.world.world_size // dp_world_size
        dp_rank = self.world.rank // non_dp_world_size
        self.runs = get_runs()

        self.receiver: MicroBatchReceiver = setup_micro_batch_receiver(output_dir, dp_rank, start_step, config)

    def wait_for_batch(self) -> None:
        if self.world.is_master:
            self.packer.pack()
        self.receiver.wait()
        self.runs.sync_runs()

    def get_batch(self) -> list[TensorMicroBatch]:
        micro_batches = self.receiver.receive()
        return [self._micro_batch_to_tensor(mb) for mb in micro_batches]

    def _micro_batch_to_tensor(self, micro_batch: MicroBatch) -> TensorMicroBatch:
        """Convert a MicroBatch (msgspec struct with lists) to a TensorMicroBatch (dict with tensors)."""
        if micro_batch.lora_num_tokens is None:
            micro_batch.lora_num_tokens = [0] * self.runs.max_runs
            micro_batch.lora_num_tokens[0] = len(micro_batch.input_ids)
        return TensorMicroBatch(
            input_ids=torch.tensor(micro_batch.input_ids, dtype=torch.long).unsqueeze(0),
            position_ids=torch.tensor(micro_batch.position_ids, dtype=torch.long).unsqueeze(0),
            advantages=torch.tensor(micro_batch.advantages, dtype=torch.float).unsqueeze(0),
            inference_logprobs=torch.tensor(micro_batch.inference_logprobs, dtype=torch.float).unsqueeze(0),
            teacher_logprobs=torch.tensor(micro_batch.teacher_logprobs, dtype=torch.float).unsqueeze(0)
            if micro_batch.teacher_logprobs is not None
            else None,
            loss_mask=torch.tensor(micro_batch.loss_mask, dtype=torch.bool).unsqueeze(0),
            temperature=micro_batch.temperature,
            lora_num_tokens=torch.tensor(micro_batch.lora_num_tokens, dtype=torch.int32),
        )
