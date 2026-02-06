import shutil
import time
from abc import ABC, abstractmethod
from collections import deque

from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.batch import prepare_batch
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.transport import (
    MicroBatch,
    MicroBatchSender,
    TrainingSample,
    TransportConfigType,
    setup_micro_batch_sender,
    setup_training_batch_receiver,
)
from prime_rl.utils.billing import get_billing_client
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_rollout_dir

TIMEOUT_SECONDS = 0.1


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
        self.multi_run_manager = get_multi_run_manager()
        self.dp_world_size = dp_world_size
        self.seq_len = seq_len
        self.pad_to_multiple_of = pad_to_multiple_of
        self.tokenizer = tokenizer
        self.receiver = setup_training_batch_receiver(config)
        shutil.rmtree(get_rollout_dir(self.multi_run_manager.output_dir), ignore_errors=True)
        self.sender: MicroBatchSender = setup_micro_batch_sender(
            self.multi_run_manager.output_dir, dp_world_size, start_step, config
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
        assert self.multi_run_manager.max_runs == 1, "SinglePacker only supports one run"

    def pack(self):
        # Wait for batch to be available
        batches = []
        while len(batches) == 0:
            self.multi_run_manager.discover_runs()
            batches = self.receiver.receive()
            time.sleep(0.2)

        assert len(batches) == 1, "SinglePacker only supports one batch per step"
        batch = batches[0]

        self.multi_run_manager.ready_to_update[0] = True
        self.multi_run_manager.progress[0].step += 1
        micro_batch_grid = prepare_batch(
            rollouts=batch.examples,
            seq_len=self.seq_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            num_train_workers=self.dp_world_size,
            idxs=[0] * len(batch.examples),
            num_loras=self.multi_run_manager.max_runs,
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
        # Per-run buffer: stores (TrainingSample, step) tuples
        self.buffers: list[deque[tuple[TrainingSample, int]]] = [
            deque() for _ in range(self.multi_run_manager.max_runs)
        ]

        # Round-robin position (persists across pack() calls)
        self._round_robin_position: int = 0

        # Register forgotten hook for receiver reset (master only, called during discover_runs)
        # This must happen when a run is deleted to prevent stale data from remaining
        self.multi_run_manager.register_forgotten_hook(self._on_run_data_deleted)

    def _on_run_data_deleted(self, idx: int, run_id: str) -> None:
        """Reset run state when run data is deleted (master only)."""
        self.logger.debug(f"Packing is resetting run state for deleted run {idx}")
        self.receiver.reset_run(idx)

        # Reset run state
        self.buffers[idx].clear()

    def _validate_sample(self, sample: TrainingSample) -> tuple[bool, str | None]:
        """Validate a sample to ensure it won't crash the trainer."""
        sample_length = len(sample.prompt_ids) + len(sample.completion_ids)
        if len(sample.prompt_mask) != len(sample.prompt_ids):
            return (
                False,
                f"Run wrote a sample with prompt mask length != prompt ids length ({len(sample.prompt_mask)} != {len(sample.prompt_ids)})",
            )
        if len(sample.completion_mask) != len(sample.completion_ids):
            return (
                False,
                f"Run wrote a sample with completion mask length != completion ids length ({len(sample.completion_mask)} != {len(sample.completion_ids)})",
            )
        if len(sample.completion_logprobs) != len(sample.completion_ids):
            return (
                False,
                f"Run wrote a sample with completion logprobs length != completion ids length ({len(sample.completion_logprobs)} != {len(sample.completion_ids)})",
            )
        if len(sample.completion_temperatures) != len(sample.completion_ids):
            return (
                False,
                f"Run wrote a sample with completion temperatures length != completion ids length ({len(sample.completion_temperatures)} != {len(sample.completion_ids)})",
            )
        if sample_length == 0:
            return False, "Run wrote a sample with no tokens"
        if sample_length > self.seq_len:
            return (
                False,
                f"Run wrote a sample with length {sample_length} which exceeds max sequence length {self.seq_len}",
            )
        if sample.teacher_logprobs is not None and len(sample.teacher_logprobs) != sample_length:
            return (
                False,
                f"Run wrote a sample with teacher logprobs length != sample length ({len(sample.teacher_logprobs)} != {sample_length})",
            )
        return True, None

    def _get_batch(self) -> None:
        """Receive batches from orchestrator and buffer samples per run."""
        self.multi_run_manager.discover_runs()
        batches = self.receiver.receive()

        for batch in batches:
            if batch.run_idx is None:
                self.logger.warning("Received batch with no run index")
                continue
            if len(batch.examples) == 0:
                self.multi_run_manager.evict_run(batch.run_idx, "Run wrote a batch with no samples")
                continue
            for sample in batch.examples:
                valid, reason = self._validate_sample(sample)
                if not valid:
                    self.multi_run_manager.evict_run(batch.run_idx, f"Run wrote a sample with invalid data: {reason}")
                    break
                self.buffers[batch.run_idx].append((sample, batch.step))

        # This is necessary to forget evicted runs
        self.multi_run_manager.discover_runs()

    def _count_tokens(self, threshold: int | None = None) -> int:
        tokens = 0

        for run_idx in self.multi_run_manager.used_idxs:
            buffer = self.buffers[run_idx]
            current_step = self.multi_run_manager.progress[run_idx].step

            for sample, step in buffer:
                if step > current_step:
                    break
                tokens += len(sample.prompt_ids) + len(sample.completion_ids)
                if threshold is not None and tokens >= threshold:
                    return tokens
        return tokens

    def _has_enough_tokens(self) -> bool:
        """Check if we have enough samples in buffer to pack a step"""
        # When not using small batch granularity, require at least one full batch
        threshold = self.seq_len * self.dp_world_size
        return self._count_tokens(threshold) >= threshold

    def _select_samples_round_robin(self, token_budget: int) -> list[tuple[int, TrainingSample, int]]:
        """Select samples using round-robin from runs with buffered work."""
        selected: list[tuple[int, TrainingSample, int]] = []
        tokens_collected = 0

        while tokens_collected < token_budget:
            # Round-robin until we find a run with work for the current step
            for _ in range(len(self.buffers)):
                if len(self.buffers[self._round_robin_position]) > 0:
                    _, step = self.buffers[self._round_robin_position][0]
                    if step <= self.multi_run_manager.progress[self._round_robin_position].step:
                        break
                self._round_robin_position = (self._round_robin_position + 1) % len(self.buffers)
            else:
                # TODO: We could probably make the logic safer. This is basically counting on _has_enough_tokens() to be correct.
                # We also need to cover the timeout case here.
                break
            run_idx = self._round_robin_position
            self._round_robin_position = (self._round_robin_position + 1) % len(self.buffers)
            current_step = self.multi_run_manager.progress[run_idx].step

            while len(self.buffers[run_idx]) > 0:
                sample, step = self.buffers[run_idx][0]
                if step > current_step:
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
                selected.append((run_idx, sample, step))
                self.buffers[run_idx].popleft()

        return selected

    def _update_run_progress(self, run_idx: int, num_samples: int, num_tokens: int) -> None:
        """Update run progress; increment step when all samples from the current step have been consumed."""
        # HACK: This fixes the issue with branching rollouts having unpredictable batch size
        # However, it makes us unable to do incremental orchestrator rollouts
        # Removing the len(self.buffers[run_idx]) == 0 check would allow incremental orchestrator rollouts
        if (
            len(self.buffers[run_idx]) == 0
            or self.buffers[run_idx][0][1] > self.multi_run_manager.progress[run_idx].step
        ):
            self.multi_run_manager.progress[run_idx].step += 1
            self.multi_run_manager.ready_to_update[run_idx] = True

        self.multi_run_manager.progress[run_idx].total_tokens += num_tokens
        self.multi_run_manager.progress[run_idx].total_samples += num_samples

    def pack(self):
        """Pack samples from buffers using round-robin fair scheduling."""
        self._get_batch()
        start_time = time.time()

        while not self._has_enough_tokens():
            if time.time() - start_time > TIMEOUT_SECONDS and self._count_tokens() > 0:
                self.logger.warning("Timeout waiting for enough tokens to pack")
                break
            time.sleep(1)
            self._get_batch()

        token_budget = self.seq_len * self.dp_world_size
        selected_samples = self._select_samples_round_robin(token_budget)
        assert selected_samples, "No samples selected"

        # Group samples by run_idx - each microbatch must contain samples from only ONE run
        # because MultiLoRAGroupedExperts (MoE) only supports one adapter per microbatch
        samples_by_run: dict[int, list[TrainingSample]] = {}
        # Track (num_samples, num_tokens, input_tokens, output_tokens) per run
        per_run_stats: dict[int, tuple[int, int, int, int]] = {}
        for run_idx, sample, step in selected_samples:
            if run_idx not in samples_by_run:
                samples_by_run[run_idx] = []
            samples_by_run[run_idx].append(sample)

            input_tokens = len(sample.prompt_ids)
            output_tokens = len(sample.completion_ids)
            num_tokens = input_tokens + output_tokens
            if run_idx in per_run_stats:
                cur_samples, cur_tokens, cur_input, cur_output = per_run_stats[run_idx]
                per_run_stats[run_idx] = (
                    cur_samples + 1,
                    cur_tokens + num_tokens,
                    cur_input + input_tokens,
                    cur_output + output_tokens,
                )
            else:
                per_run_stats[run_idx] = (1, num_tokens, input_tokens, output_tokens)

        # Update progress and report usage for billing
        billing_client = get_billing_client()
        for run_idx, (num_samples, num_tokens, input_tokens, output_tokens) in per_run_stats.items():
            self._update_run_progress(run_idx, num_samples, num_tokens)

            # Report usage for billing (run_id from folder name)
            run_id = self.multi_run_manager.idx_2_id.get(run_idx)
            if run_id:
                step = self.multi_run_manager.progress[run_idx].step
                billing_client.report_usage(
                    run_id=run_id,
                    step=step,
                    tokens=num_tokens,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

        # Pack each run separately to ensure no mixing of runs in microbatches
        all_micro_batches: list[list[MicroBatch]] = [[] for _ in range(self.dp_world_size)]
        for run_idx in sorted(samples_by_run.keys()):
            run_samples = samples_by_run[run_idx]
            run_micro_batch_grid = prepare_batch(
                rollouts=run_samples,
                seq_len=self.seq_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                num_train_workers=self.dp_world_size,
                idxs=[run_idx] * len(run_samples),
                num_loras=self.multi_run_manager.max_runs,
            )
            # Merge into combined grid
            for worker_idx, worker_batches in enumerate(run_micro_batch_grid):
                all_micro_batches[worker_idx].extend(worker_batches)

        self.sender.send(all_micro_batches)


def setup_packer(
    dp_world_size: int,
    seq_len: int,
    pad_to_multiple_of: int,
    tokenizer: PreTrainedTokenizer,
    transport_config: TransportConfigType,
    start_step: int = 0,
) -> BasePacker:
    multi_run_manager = get_multi_run_manager()
    if multi_run_manager.max_runs == 1:
        return SinglePacker(dp_world_size, seq_len, pad_to_multiple_of, tokenizer, transport_config, start_step)
    else:
        return MultiPacker(dp_world_size, seq_len, pad_to_multiple_of, tokenizer, transport_config, start_step)
