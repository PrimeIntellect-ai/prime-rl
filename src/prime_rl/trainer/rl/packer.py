import shutil
import time
from collections import defaultdict, deque

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


class Packer:
    def __init__(
        self,
        dp_world_size: int,
        seq_len: int,
        pad_to_multiple_of: int,
        tokenizer: PreTrainedTokenizer,
        config: TransportConfigType,
        start_step: int = 0,
        small_batch_granularity: bool = False,
    ):
        self.logger = get_logger()
        self.runs = get_runs()
        self.dp_world_size = dp_world_size
        self.seq_len = seq_len
        self.pad_to_multiple_of = pad_to_multiple_of
        self.tokenizer = tokenizer
        self.small_batch_granularity = small_batch_granularity
        self.receiver = setup_training_batch_receiver(config)
        shutil.rmtree(get_rollout_dir(self.runs.output_dir), ignore_errors=True)
        self.sender: MicroBatchSender = setup_micro_batch_sender(
            self.runs.output_dir, dp_world_size, start_step, config
        )

        # Per-run buffer: stores (TrainingSample, temperature) tuples
        self.buffers: dict[int, deque[tuple[TrainingSample, float]]] = defaultdict(deque)

        # Per-run sample count consumed in current step
        self.samples_consumed_this_step: dict[int, int] = defaultdict(int)

        # Round-robin position (persists across pack() calls)
        self._round_robin_position: int = 0

        # Register creation hook to reset state when a run is replaced
        self.runs.register_creation_hook(self._on_run_created)

    def _on_run_created(self, idx: int, run_id: str) -> None:
        """Reset packer state for a run index when a new run is created.

        Called via creation hook when a run is deleted and a new run takes its place.
        Clears buffered samples and partial step progress for the index.
        """
        # Clear any buffered samples from the old run
        if idx in self.buffers:
            self.buffers[idx].clear()

        # Reset partial step progress
        if idx in self.samples_consumed_this_step:
            del self.samples_consumed_this_step[idx]

        # Reset receiver state (e.g., received step tracking)
        self.receiver.reset_run(idx)

    def _get_batch(self) -> None:
        """Receive batches from orchestrator and buffer samples per run."""
        self.runs.check_for_changes()
        batches = self.receiver.receive()

        for batch in batches:
            if batch.run_idx is None:
                self.logger.warning("Received batch with no run index")
                continue
            for sample in batch.examples:
                self.buffers[batch.run_idx].append((sample, batch.temperature))

    def _get_runs_with_full_batch(self) -> list[int]:
        """Get run indices that have at least batch_size samples buffered."""
        runs_with_full_batch = []
        for run_idx in self.runs.used_idxs:
            batch_size = self.runs.config[run_idx].batch_size
            if len(self.buffers[run_idx]) >= batch_size:
                runs_with_full_batch.append(run_idx)
        return runs_with_full_batch

    def _has_enough_tokens(self) -> bool:
        """Check if we have enough samples in buffer to pack a step

        When small_batch_granularity=False, requires at least one run to have batch_size samples before packing.
        When small_batch_granularity=True, we pack whenever we can make at least 1 micro batch for each data rank.
        """
        # When not using small batch granularity, require at least one full batch
        if not self.small_batch_granularity:
            return len(self._get_runs_with_full_batch()) > 0

        threshold = self.seq_len * self.dp_world_size
        tokens = 0
        samples = 1e-5  # Avoid division by zero

        for buffer in self.buffers.values():
            for sample, _ in buffer:
                tokens += len(sample.prompt_ids) + len(sample.completion_ids)
                samples += 1
                estimated_next_sample_tokens = tokens + tokens / samples
                if estimated_next_sample_tokens >= threshold:
                    return True
        return False

    def _select_samples_round_robin(self, token_budget: int) -> list[tuple[int, TrainingSample, float]]:
        """Select samples using round-robin from runs with buffered work.

        When small_batch_granularity=False, we ignore the token budget and select all the samples from a run with enough samples.
        When small_batch_granularity=True, we select samples from runs with buffered work until we have enough tokens to pack a step.
        """
        selected: list[tuple[int, TrainingSample, float]] = []
        tokens_collected = 0

        # For full batch mode, determine eligible runs once at the start
        # (so popping samples doesn't disqualify runs mid-selection)
        if not self.small_batch_granularity:
            run_idx = self._get_runs_with_full_batch()[0]
            while len(self.buffers[run_idx]) > 0:
                sample, temperature = self.buffers[run_idx].pop()
                selected.append((run_idx, sample, temperature))
            return selected

        while tokens_collected < token_budget:
            # Round-robin until we find a run with work
            for _ in range(len(self.buffers)):
                if len(self.buffers[self._round_robin_position]) != 0:
                    break
                self._round_robin_position = (self._round_robin_position + 1) % len(self.buffers)
            else:
                # TODO: We could probably make the logic safer. This is basically counting on _has_enough_tokens() to be correct.
                raise ValueError("No runs with work found. This should never happen.")
            run_idx = self._round_robin_position
            self._round_robin_position += 1

            while len(self.buffers[run_idx]) > 0:
                sample, temperature = self.buffers[run_idx][-1]
                tokens_collected += len(sample.prompt_ids) + len(sample.completion_ids)
                if tokens_collected > token_budget:
                    return selected
                selected.append((run_idx, sample, temperature))
                self.buffers[run_idx].pop()

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
            if (
                self.small_batch_granularity
                and time.time() - start_time > TIMEOUT_SECONDS
                and any(self.buffers.values())
            ):
                self.logger.warning("Timeout waiting for enough tokens to pack")
                break
            time.sleep(1)
            self._get_batch()

        token_budget = self.seq_len * self.dp_world_size
        selected_samples = self._select_samples_round_robin(token_budget)
        assert selected_samples, "No samples selected"

        # Group by run for prepare_batch (MultiLoRAMoE requires same run_idx in microbatch)
        samples_by_run: dict[int, list[tuple[TrainingSample, float]]] = defaultdict(list)
        for run_idx, sample, temperature in selected_samples:
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
