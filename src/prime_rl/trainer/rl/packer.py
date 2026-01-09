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

    def get_batch(self) -> None:
        """Receive batches from orchestrator and buffer samples per run."""
        self.runs.check_for_changes()
        batches = self.receiver.receive()

        for batch in batches:
            if batch.run_idx is None:
                # TODO: put a warning here
                continue
            for sample in batch.examples:
                self.buffers[batch.run_idx].append((sample, batch.temperature))

    def _get_runs_with_full_batch(self) -> list[int]:
        """Get run indices that have at least batch_size samples buffered."""
        runs_with_full_batch = []
        for run_idx in self.runs.used_idxs:
            if run_idx not in self.runs.config:
                continue
            batch_size = self.runs.config[run_idx].batch_size
            if len(self.buffers[run_idx]) >= batch_size:
                runs_with_full_batch.append(run_idx)
        return runs_with_full_batch

    def has_enough_tokens(self) -> bool:
        """Check if buffered samples have enough tokens to pack.

        When small_batch_granularity=False (default), requires at least one run
        to have batch_size samples before packing.
        When small_batch_granularity=True, packs whenever token threshold is met.
        """
        # When not using small batch granularity, require at least one full batch
        if not self.small_batch_granularity:
            if not self._get_runs_with_full_batch():
                return False

        threshold = self.seq_len * self.dp_world_size
        tokens = 0
        batches = 1e-5  # Avoid division by zero

        for run_idx, buffer in self.buffers.items():
            for sample, _ in buffer:
                tokens += len(sample.prompt_ids) + len(sample.completion_ids)
            if buffer:
                batches += 1
            estimated_next_batch_tokens = tokens + tokens / batches
            if estimated_next_batch_tokens >= threshold:
                return True
        return False

    def _select_samples_round_robin(self, token_budget: int) -> list[tuple[int, TrainingSample, float]]:
        """Select samples using round-robin from runs with buffered work.

        When small_batch_granularity=False (default), only selects from runs
        that have at least batch_size samples buffered at the start of selection.
        When small_batch_granularity=True, selects from any run with buffered work.
        """
        selected: list[tuple[int, TrainingSample, float]] = []
        tokens_collected = 0

        # For full batch mode, determine eligible runs once at the start
        # (so popping samples doesn't disqualify runs mid-selection)
        eligible_runs = set(self._get_runs_with_full_batch()) if not self.small_batch_granularity else None

        while tokens_collected < token_budget:
            if self.small_batch_granularity:
                # Select from any run with buffered work
                runs_with_work = [idx for idx in self.runs.used_idxs if self.buffers[idx]]
            else:
                # Only select from runs that had full batch_size at start AND still have samples
                runs_with_work = [idx for idx in self.runs.used_idxs if idx in eligible_runs and self.buffers[idx]]

            if not runs_with_work:
                break

            self._round_robin_position = self._round_robin_position % len(runs_with_work)
            run_idx = runs_with_work[self._round_robin_position]

            if self.buffers[run_idx]:
                sample, temperature = self.buffers[run_idx].popleft()
                selected.append((run_idx, sample, temperature))
                tokens_collected += len(sample.prompt_ids) + len(sample.completion_ids)

            self._round_robin_position += 1

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
        self.get_batch()
        start_time = time.time()

        while not self.has_enough_tokens():
            if time.time() - start_time > TIMEOUT_SECONDS and any(self.buffers.values()):
                self.logger.warning("Timeout waiting for enough tokens to pack")
                break
            time.sleep(1)
            self.get_batch()

        token_budget = self.seq_len * self.dp_world_size
        selected_samples = self._select_samples_round_robin(token_budget)

        if not selected_samples:
            return

        # Group by run for prepare_batch (MultiLoRAMoE requires same run_idx in microbatch)
        samples_by_run: dict[int, list[tuple[TrainingSample, float]]] = defaultdict(list)
        for run_idx, sample, temperature in selected_samples:
            samples_by_run[run_idx].append((sample, temperature))

        micro_batch_grid = [[] for _ in range(self.dp_world_size)]

        for run_idx, sample_temp_pairs in samples_by_run.items():
            samples = [s for s, _ in sample_temp_pairs]
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
