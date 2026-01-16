import shutil
import time
from abc import ABC, abstractmethod
from collections import deque

from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.batch import prepare_batch
from prime_rl.trainer.runs import get_runs
from prime_rl.transport import (
    MicroBatchSender,
    TrainingSample,
    TransportConfigType,
    setup_micro_batch_sender,
    setup_training_batch_receiver,
)
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_rollout_dir

TIMEOUT_SECONDS = 10


class BasePacker(ABC):
    def __init__(
        self,
        dp_world_size: int,
        seq_len: int,
        pad_to_multiple_of: int,
        tokenizer: PreTrainedTokenizer,
        config: TransportConfigType,
        start_step: int = 0,
    ):
        self.logger = get_logger()
        self.runs = get_runs()
        self.dp_world_size = dp_world_size
        self.seq_len = seq_len
        self.pad_to_multiple_of = pad_to_multiple_of
        self.tokenizer = tokenizer
        self.receiver = setup_training_batch_receiver(config)
        shutil.rmtree(get_rollout_dir(self.runs.output_dir), ignore_errors=True)
        self.sender: MicroBatchSender = setup_micro_batch_sender(
            self.runs.output_dir, dp_world_size, start_step, config
        )

    @abstractmethod
    def pack(self) -> None:
        """Pack samples for the next step."""
        pass


class SinglePacker(BasePacker):
    def __init__(
        self,
        dp_world_size: int,
        seq_len: int,
        pad_to_multiple_of: int,
        tokenizer: PreTrainedTokenizer,
        config: TransportConfigType,
        start_step: int = 0,
    ):
        super().__init__(dp_world_size, seq_len, pad_to_multiple_of, tokenizer, config, start_step)
        assert self.runs.max_runs == 1, "SinglePacker only supports one run"

    def pack(self):
        # Wait for batch to be available
        batches = []
        while len(batches) == 0:
            self.runs.check_for_changes()
            batches = self.receiver.receive()
            time.sleep(0.2)

        assert len(batches) == 1, "SinglePacker only supports one batch per step"
        batch = batches[0]

        self.runs.ready_to_update[0] = True
        self.runs.progress[0].step += 1
        micro_batch_grid = prepare_batch(
            rollouts=batch.examples,
            temperature=batch.temperature,
            seq_len=self.seq_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            num_train_workers=self.dp_world_size,
            idxs=[0] * len(batch.examples),
            num_loras=self.runs.max_runs,
        )

        self.sender.send(micro_batch_grid)


class MultiPacker(BasePacker):
    def __init__(
        self,
        dp_world_size: int,
        seq_len: int,
        pad_to_multiple_of: int,
        tokenizer: PreTrainedTokenizer,
        config: TransportConfigType,
        start_step: int = 0,
    ):
        super().__init__(dp_world_size, seq_len, pad_to_multiple_of, tokenizer, config, start_step)
        # Per-run buffer: stores (TrainingSample, temperature, step) tuples
        self.buffers: list[deque[tuple[TrainingSample, float, int]]] = [deque() for _ in range(self.runs.max_runs)]

        # Per-run sample count consumed in current step
        self.samples_consumed_this_step: list[int] = [0] * self.runs.max_runs

        # Round-robin position (persists across pack() calls)
        self._round_robin_position: int = 0

        # Register delete_run_data hook for receiver reset (master only, runs during check_for_changes)
        # This must happen when a run is deleted to prevent stale data from remaining
        self.runs.register_delete_run_data_hook(self._on_run_data_deleted)

    def _on_run_data_deleted(self, idx: int, run_id: str) -> None:
        """Reset run state when run data is deleted (master only)."""
        self.logger.debug(f"Packing is resetting run state for deleted run {idx}")
        self.receiver.reset_run(idx)

        # Reset run state
        self.buffers[idx].clear()
        self.samples_consumed_this_step[idx] = 0

    def _get_batch(self) -> None:
        """Receive batches from orchestrator and buffer samples per run."""
        self.runs.check_for_changes()
        batches = self.receiver.receive()

        for batch in batches:
            if batch.run_idx is None:
                self.logger.warning("Received batch with no run index")
                continue
            for sample in batch.examples:
                self.buffers[batch.run_idx].append((sample, batch.temperature, batch.step))

    def _has_enough_tokens(self) -> bool:
        """Check if we have enough samples in buffer to pack a step"""
        # When not using small batch granularity, require at least one full batch
        threshold = self.seq_len * self.dp_world_size
        tokens = 0

        for run_idx in self.runs.used_idxs:
            buffer = self.buffers[run_idx]
            current_step = self.runs.progress[run_idx].step
            for sample, _, step in buffer:
                if step != current_step:
                    continue
                tokens += len(sample.prompt_ids) + len(sample.completion_ids)
                if tokens >= threshold:
                    return True
        return False

    def _select_samples_round_robin(self, token_budget: int) -> list[tuple[int, TrainingSample, float]]:
        """Select samples using round-robin from runs with buffered work."""
        selected: list[tuple[int, TrainingSample, float]] = []
        tokens_collected = 0

        while tokens_collected < token_budget:
            # Round-robin until we find a run with work for the current step
            for _ in range(len(self.buffers)):
                if len(self.buffers[self._round_robin_position]) > 0:
                    _, _, step = self.buffers[self._round_robin_position][0]
                    if step == self.runs.progress[self._round_robin_position].step:
                        break
                self._round_robin_position = (self._round_robin_position + 1) % len(self.buffers)
            else:
                # TODO: We could probably make the logic safer. This is basically counting on _has_enough_tokens() to be correct.
                # We also need to cover the timeout case here.
                break
            run_idx = self._round_robin_position
            self._round_robin_position = (self._round_robin_position + 1) % len(self.buffers)
            current_step = self.runs.progress[run_idx].step

            while len(self.buffers[run_idx]) > 0:
                sample, temperature, step = self.buffers[run_idx][0]
                if step != current_step:
                    # Samples from different steps should be consumed later
                    break
                tokens_collected += len(sample.prompt_ids) + len(sample.completion_ids)
                if tokens_collected > token_budget:
                    if tokens_collected == (len(sample.prompt_ids) + len(sample.completion_ids)):
                        tokens_collected -= len(sample.prompt_ids) + len(sample.completion_ids)
                        # This means we have a sample that has more tokens than max seqlen
                        self.buffers[run_idx].popleft()
                        continue
                    return selected
                selected.append((run_idx, sample, temperature))
                self.buffers[run_idx].popleft()

        return selected

    def _update_run_progress(self, run_idx: int, num_samples: int, num_tokens: int) -> None:
        """Update progress, increment step only when batch_size reached."""
        self.samples_consumed_this_step[run_idx] += num_samples
        batch_size = self.runs.config[run_idx].batch_size

        # May complete multiple steps if we consumed more than batch_size worth
        while self.samples_consumed_this_step[run_idx] >= batch_size:
            self.runs.progress[run_idx].step += 1
            self.runs.ready_to_update[run_idx] = True
            self.samples_consumed_this_step[run_idx] -= batch_size

        self.runs.progress[run_idx].total_tokens += num_tokens
        self.runs.progress[run_idx].total_samples += num_samples

    def pack(self):
        """Pack samples from buffers using round-robin fair scheduling."""
        self._get_batch()
        start_time = time.time()

        while not self._has_enough_tokens():
            if time.time() - start_time > TIMEOUT_SECONDS and any(len(i) > 0 for i in self.buffers):
                self.logger.warning("Timeout waiting for enough tokens to pack")
                break
            time.sleep(1)
            self._get_batch()

        token_budget = self.seq_len * self.dp_world_size
        selected_samples = self._select_samples_round_robin(token_budget)
        assert selected_samples, "No samples selected"

        # Group by run for prepare_batch (MultiLoRAMoE requires same run_idx in microbatch)
        samples_by_run: dict[int, list[tuple[TrainingSample, float]]] = {}
        for run_idx, sample, temperature in selected_samples:
            if run_idx not in samples_by_run:
                samples_by_run[run_idx] = []
            samples_by_run[run_idx].append((sample, temperature))

        micro_batch_grid = [[] for _ in range(self.dp_world_size)]

        for run_idx, sample_temp_pairs in samples_by_run.items():
            samples = [s for s, _ in sample_temp_pairs]
            # We don't support dynamic temperatures in orchestrator yet
            # So this works for now
            temperature = sample_temp_pairs[0][1]

            num_samples = len(samples)
            num_tokens = sum(len(s.prompt_ids) + len(s.completion_ids) for s in samples)

            self._update_run_progress(run_idx, num_samples, num_tokens)

            _micro_batch_grid = prepare_batch(
                rollouts=samples,
                temperature=temperature,
                seq_len=self.seq_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                num_train_workers=self.dp_world_size,
                idxs=[run_idx] * num_samples,
                num_loras=self.runs.max_runs,
            )

            for i, micro_batch in enumerate(_micro_batch_grid):
                micro_batch_grid[i].extend(micro_batch)

        self.sender.send(micro_batch_grid)


def setup_packer(
    dp_world_size: int,
    seq_len: int,
    pad_to_multiple_of: int,
    tokenizer: PreTrainedTokenizer,
    transport_config: TransportConfigType,
    start_step: int = 0,
) -> BasePacker:
    runs = get_runs()
    if runs.max_runs == 1:
        return SinglePacker(dp_world_size, seq_len, pad_to_multiple_of, tokenizer, transport_config, start_step)
    else:
        return MultiPacker(dp_world_size, seq_len, pad_to_multiple_of, tokenizer, transport_config, start_step)
