import shutil
import time
from pathlib import Path

import torch
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.batch import prepare_batch
from prime_rl.trainer.runs import get_runs
from prime_rl.transport import TrainingBatch, TrainingExample, TransportConfigType, setup_training_batch_receiver
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_rollout_dir, get_step_path


class Packer:
    def __init__(self, dp_world_size: int, seq_len: int, tokenizer: PreTrainedTokenizer, config: TransportConfigType):
        self.logger = get_logger()
        self.runs = get_runs()
        self.dp_world_size = dp_world_size
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.trainer_step = 0
        self.receiver = setup_training_batch_receiver(self.runs.output_dir, config)
        shutil.rmtree(get_rollout_dir(self.runs.output_dir), ignore_errors=True)

    def get_rollout_paths(self) -> list[tuple[int, Path]]:
        return [
            (
                idx,
                get_step_path(get_rollout_dir(self.runs.get_run_dir(idx)), self.runs.progress[idx].step)
                / "rollouts.bin",
            )
            for idx in self.runs.used_idxs
        ]

    def get_batch(self) -> dict[int, TrainingBatch]:
        rollouts: dict[int, TrainingBatch] = {}
        self.runs.check_for_changes()
        self.logger.debug(f"Looking in {self.get_rollout_paths()}")
        for idx, rollout_path in self.get_rollout_paths():
            if rollout_path.exists() and not self.runs.ready_to_update[idx]:
                try:
                    # TODO: Use the receiver properly
                    rollouts[idx] = self.receiver.decoder.decode(rollout_path.read_bytes())
                except Exception as e:
                    # This might happens if run is deleted midway in this loop
                    self.logger.error(f"Error loading rollouts for run {idx}: {e}")
                    self.runs.check_for_changes()
        return rollouts

    def has_enough_tokens(self, rollouts: dict[int, TrainingBatch]) -> bool:
        tokens = 0
        batches = 1e-5  # Avoid division by zero
        threshold = self.seq_len * self.dp_world_size
        for _rollouts in rollouts.values():
            for rollout in _rollouts.examples:
                tokens += len(rollout.prompt_ids) + len(rollout.completion_ids)
            batches += 1
            estimated_next_batch_tokens = tokens + tokens / batches
            if estimated_next_batch_tokens >= threshold:
                return True
        else:
            self.logger.warning(f"Not enough tokens to pack. Expected {threshold} tokens, got {tokens}")
            return False

    def pack(self):
        training_batches: dict[int, TrainingBatch] = self.get_batch()
        # TODO: Handle timeout case
        while not self.has_enough_tokens(training_batches):
            time.sleep(1)
            training_batches = self.get_batch()

        train_examples: list[TrainingExample] = []
        train_idxs = []
        for idx, training_batch in training_batches.items():
            self.runs.progress[idx].step += 1
            self.runs.progress[idx].total_tokens += sum(
                len(rollout.prompt_ids) + len(rollout.completion_ids) for rollout in training_batch.examples
            )
            self.runs.progress[idx].total_samples += len(training_batch.examples)
            train_examples.extend(training_batch.examples)
            train_idxs.extend([idx] * len(training_batch.examples))
            self.runs.ready_to_update[idx] = True

        # TODO: Handle different temperatures for each run
        some_temperature = next(iter(training_batches.values())).temperature
        all_data_ranks_batches = prepare_batch(
            rollouts=train_examples,
            temperature=some_temperature,
            seq_len=self.seq_len,
            num_train_workers=self.dp_world_size,
            # idxs=train_idxs, # Needed for lora later
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
