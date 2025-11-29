import copy
import shutil
import time
from pathlib import Path

import torch
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.batch import BatchSample, MicroBatch, prepare_sample
from prime_rl.trainer.runs import get_runs
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_rollout_dir, get_step_path
from prime_rl.utils.vf import Rollout


class Packer:
    def __init__(self, dp_world_size: int, seq_len: int, tokenizer: PreTrainedTokenizer):
        self.logger = get_logger()
        self.runs = get_runs()
        self.dp_world_size = dp_world_size
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.trainer_step = 0
        shutil.rmtree(get_rollout_dir(self.runs.output_dir), ignore_errors=True)

    def get_rollout_paths(self) -> list[tuple[int, Path]]:
        return [
            (
                idx,
                get_step_path(get_rollout_dir(self.runs.get_run_dir(idx)), self.runs.progress[idx].step)
                / "rollouts.pt",
            )
            for idx in self.runs.used_idxs
        ]

    def get_batch(self) -> dict[int, list[Rollout]]:
        rollouts: dict[int, list[Rollout]] = {}
        self.runs.check_for_changes()
        self.logger.debug(f"Looking in {self.get_rollout_paths()}")
        for idx, rollout_path in self.get_rollout_paths():
            if rollout_path.exists() and not self.runs.ready_to_update[idx]:
                try:
                    rollouts[idx] = torch.load(rollout_path)
                except Exception as e:
                    # This might happens if run is deleted midway in this loop
                    self.logger.error(f"Error loading rollouts for run {idx}: {e}")
                    self.runs.check_for_changes()
        return rollouts

    def has_enough_tokens(self, rollouts: dict[int, list[Rollout]]) -> bool:
        tokens = 0
        batches = 1e-5  # Avoid division by zero
        threshold = self.seq_len * self.dp_world_size
        for _rollouts in rollouts.values():
            for rollout in _rollouts:
                tokens += len(rollout["prompt_ids"]) + len(rollout["completion_ids"])
            batches += 1
            estimated_next_batch_tokens = tokens + tokens / batches
            if estimated_next_batch_tokens >= threshold:
                return True
        else:
            return False

    def pack(self):
        rollouts: dict[int, list[Rollout]] = self.get_batch()
        # TODO: Handle timeout case
        while not self.has_enough_tokens(rollouts):
            time.sleep(1)
            rollouts = self.get_batch()

        train_rollouts = []
        train_idxs = []
        for idx, rollouts in rollouts.items():
            self.runs.progress[idx].step += 1
            self.runs.progress[idx].total_tokens += sum(
                len(rollout["prompt_ids"]) + len(rollout["completion_ids"]) for rollout in rollouts
            )
            self.runs.progress[idx].total_samples += len(rollouts)
            train_rollouts.extend(rollouts)
            train_idxs.extend([idx] * len(rollouts))
            self.runs.ready_to_update[idx] = True

        # TODO: Handle different temperatures for each run
        some_config = self.runs.config[train_idxs[0]]
        all_data_ranks_batches = self.prepare_batch(
            rollouts=train_rollouts,
            idxs=train_idxs,
            temperature=some_config.sampling.temperature,
        )

        step_path = get_rollout_dir(self.runs.output_dir) / f"step_{self.trainer_step}"
        step_path.mkdir(parents=True, exist_ok=True)
        for i, batches in enumerate(all_data_ranks_batches):
            batch_path = step_path / f"rank_{i}.pt"
            tmp_path = batch_path.with_suffix(".tmp")
            self.logger.debug(f"Saving rollouts for step {self.trainer_step} for rank {i} to {batch_path}")
            torch.save(batches, tmp_path)
            tmp_path.rename(batch_path)

        self.trainer_step += 1

    def prepare_batch(
        self,
        rollouts: list[Rollout],
        idxs: list[int],
        temperature: float,
    ) -> list[list[MicroBatch]]:
        """
        Prepare a batch of problems for each GPU. Each batch is a list of micro batches.
        Each micro batch is shape [1, seq_len], the namber of sample is not fixed per micro batch.
        """
        rollouts = copy.deepcopy(rollouts)

        all_samples = [
            prepare_sample(
                rollout,
                self.seq_len,
                idx,
            )
            for idx, rollout in zip(idxs, rollouts)
        ]

        micro_batches = self.packed_samples_into_micro_bs(all_samples, temperature)
        num_padding_batch = -len(micro_batches) % self.dp_world_size

        # because of fsdp we need to make sure that each data ran has the same number of micro batches otherwise training will hang.
        # We create fake micro batches to fill the gap with real data but zero advantages, they would not contribute to the loss.
        if self.dp_world_size > 1 and num_padding_batch > 0:
            padded_batch = copy.deepcopy(micro_batches[0])
            padded_batch["advantages"] = torch.zeros_like(padded_batch["advantages"])
            padded_batch["loss_mask"] = torch.zeros_like(padded_batch["loss_mask"], dtype=torch.bool)
            micro_batches.extend([padded_batch for _ in range(num_padding_batch)])

        assert len(micro_batches) % self.dp_world_size == 0, (
            "Number of micro batches is not divisible by number of data ranks"
        )

        per_gpu_micro_batches = len(micro_batches) // self.dp_world_size
        batches_per_gpu = []
        for _ in range(self.dp_world_size):
            batches = []
            for _ in range(per_gpu_micro_batches):
                batches.append(micro_batches.pop(0))
            batches_per_gpu.append(batches)

        return batches_per_gpu

    def packed_samples_into_micro_bs(self, samples: list[BatchSample], temperature: float) -> list[MicroBatch]:
        """
        Pack samples into micro_batch efficiently.
        We follow the First Fit Decreasing algorithm to pack the samples into bins and minimize potential padding while never truncating.
        """
        # We have to put the same lora samples together for the offsets
        sorted_samples = sorted(samples, key=lambda x: (x["lora_idx"], -len(x["input_ids"])))

        ## we create bins
        bin_samples = []
        bin_len = 0
        idx_tokens = [0] * self.runs.max_runs
        micro_batches = []

        for sample in sorted_samples:
            # Check if sequence fits in this bin
            if bin_len + len(sample["input_ids"]) <= self.seq_len:
                bin_samples.append(sample)
                bin_len += len(sample["input_ids"])
                idx_tokens[sample["lora_idx"]] += len(sample["input_ids"])
            else:
                micro_batches.append(self.prepare_micro_batch_packing(bin_samples, temperature, idx_tokens))
                bin_samples = [sample]
                bin_len = len(sample["input_ids"])
                idx_tokens = [0] * self.runs.max_runs
                idx_tokens[sample["lora_idx"]] += len(sample["input_ids"])
        micro_batches.append(self.prepare_micro_batch_packing(bin_samples, temperature, idx_tokens))

        return micro_batches

    def prepare_micro_batch_packing(
        self, samples: list[BatchSample], temperature: float, idx_tokens: list[int]
    ) -> MicroBatch:
        """
        Prepare a micro batch for packing mode. take multi sample and return a batch of shape [1, micro_bs * max_seq_len].
        Would additionally pad the batch to the max sequence length.
        """
        micro_batch = {}
        assert sum([len(sample["input_ids"]) for sample in samples]) <= self.seq_len, (
            "Total tokens of samples is greater than max sequence length"
        )

        for key in ["input_ids", "advantages", "loss_mask", "position_ids", "inference_logprobs"]:
            micro_batch[key] = torch.cat([sample[key] for sample in samples], dim=0).unsqueeze(0)

        micro_batch["temperature"] = temperature
        micro_batch["lora_cu_offsets"] = torch.cumsum(torch.tensor(idx_tokens, dtype=torch.int32), dim=0)

        return micro_batch
