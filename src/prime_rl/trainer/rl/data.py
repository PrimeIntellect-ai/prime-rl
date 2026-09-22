from collections.abc import Sequence
from pathlib import Path

import torch

from prime_rl.configs.trainer import FakeDataLoaderConfig
from prime_rl.trainer.world import get_world
from prime_rl.transports.batch import (
    BatchReceiver,
    MicroBatch,
    TransportConfig,
    setup_batch_receiver,
)
from prime_rl.transports.batch.mmap import MMapBatchReceiver
from prime_rl.transports.batch.tensors import TensorMicroBatch, micro_batch_to_tensor


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
        sequence_lengths = []

        while total_seq_len < self.seq_len:
            # Generate reasonably long documents
            seq_len_to_generate = torch.randint(1, self.seq_len // 8, (1,), generator=generator).item()
            if seq_len_to_generate + total_seq_len > self.seq_len:
                seq_len_to_generate = self.seq_len - total_seq_len
            total_seq_len += seq_len_to_generate
            sequence_lengths.append(seq_len_to_generate)
            tmp_input_ids = torch.randint(0, 120000, (seq_len_to_generate,), generator=generator).long()
            tmp_position_ids = torch.arange(seq_len_to_generate).long()

            input_ids.append(tmp_input_ids)
            position_ids.append(tmp_position_ids)

        input_ids = torch.cat(input_ids, dim=0)
        position_ids = torch.cat(position_ids, dim=0)
        loss_mask = torch.ones(input_ids.shape[0], dtype=torch.bool)
        advantages = torch.randn(input_ids.shape[0], generator=generator)
        inference_logprobs = torch.randn(input_ids.shape[0], generator=generator)

        return {
            "input_ids": input_ids.unsqueeze(0),
            "position_ids": position_ids.unsqueeze(0),
            "advantages": advantages.unsqueeze(0),
            "inference_logprobs": inference_logprobs.unsqueeze(0),
            "ref_logprobs": None,
            "temperatures": torch.ones(input_ids.shape[0]).unsqueeze(0),
            "env_names": ["fake"] * input_ids.shape[0],
            "sequence_lengths": sequence_lengths,
            "trace_ids": None,
            "branch_indices": None,
            "loss_mask": loss_mask.unsqueeze(0),
            "lora_num_tokens": torch.tensor([input_ids.shape[0]], dtype=torch.int32),
            "seq_lens": torch.tensor(sequence_lengths, dtype=torch.long),
            "routed_experts": None,
            "sampling_mask": None,
            "mm_refs": None,
            "mm_token_type_ids": None,
            "rl_weights": None,
            "ce_weights": None,
            "ref_kl_weights": None,
        }

    def _get_micro_batch(self, generator: torch.Generator) -> TensorMicroBatch:
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
            "ref_logprobs": None,
            "temperatures": torch.ones(self.seq_len).unsqueeze(0),
            "env_names": ["fake"] * self.seq_len,
            "sequence_lengths": [self.seq_len],
            "trace_ids": None,
            "branch_indices": None,
            "loss_mask": torch.ones(self.seq_len, dtype=torch.bool).unsqueeze(0),
            "lora_num_tokens": torch.tensor([self.seq_len], dtype=torch.int32),
            "seq_lens": torch.tensor([self.seq_len], dtype=torch.long),
            "routed_experts": None,
            "sampling_mask": None,
            "mm_refs": None,
            "mm_token_type_ids": None,
            "rl_weights": None,
            "ce_weights": None,
            "ref_kl_weights": None,
        }


class DataLoader:
    """Receives packed micro batches from the orchestrator, one stream per DP rank."""

    def __init__(
        self,
        output_dir: Path,
        start_step: int,
        dp_world_size: int,
        config: TransportConfig,
    ):
        self.world = get_world()

        non_dp_world_size = self.world.world_size // dp_world_size
        dp_rank = self.world.rank // non_dp_world_size

        self.receiver: BatchReceiver = setup_batch_receiver(output_dir, dp_rank, start_step, config)
        if isinstance(self.receiver, MMapBatchReceiver):
            if config.readers_per_rank != non_dp_world_size:
                raise ValueError(f"mmap readers_per_rank={config.readers_per_rank} != CP/PP size {non_dp_world_size}")
            self.receiver.reader_id = self.world.rank % non_dp_world_size

    def wait_for_batch(self) -> None:
        self.receiver.wait()

    def get_batch(self) -> Sequence[TensorMicroBatch]:
        micro_batches = self.receiver.receive()
        if isinstance(self.receiver, MMapBatchReceiver):
            return micro_batches
        return [self._micro_batch_to_tensor(mb) for mb in micro_batches]

    def _micro_batch_to_tensor(self, micro_batch: MicroBatch) -> TensorMicroBatch:
        return micro_batch_to_tensor(micro_batch)
