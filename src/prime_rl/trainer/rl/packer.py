import shutil
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Literal

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
from prime_rl.utils.logger import ProgressTracker, get_logger
from prime_rl.utils.pathing import get_rollout_dir

TIMEOUT_SECONDS = 0.1


def _sample_num_tokens(sample: TrainingSample) -> int:
    return len(sample.prompt_ids) + len(sample.completion_ids)


def _resolve_batch_target(
    token_batch_size: int | None,
    rollout_batch_size: int | None,
    *,
    default_token_batch_size: int | None = None,
) -> tuple[int, Literal["tokens", "rollouts"]]:
    if token_batch_size is not None and rollout_batch_size is not None:
        raise ValueError("Only one of token_batch_size or rollout_batch_size can be set")

    if token_batch_size is None and rollout_batch_size is None:
        if default_token_batch_size is None:
            raise ValueError("Either token_batch_size or rollout_batch_size is required")
        token_batch_size = default_token_batch_size

    if token_batch_size is not None:
        return token_batch_size, "tokens"

    assert rollout_batch_size is not None
    return rollout_batch_size, "rollouts"


def _resolve_batch_target_from_orchestrator_config(orch_config) -> tuple[int, Literal["tokens", "rollouts"]]:
    return _resolve_batch_target(orch_config.token_batch_size, orch_config.batch_size)


def _resolve_batch_target_from_discovered_runs(
    default_token_batch_size: int,
) -> tuple[int, Literal["tokens", "rollouts"], bool]:
    """Resolve batch target from orchestrator configs already on disk.

    Returns (target, unit, is_provisional).  ``is_provisional`` is True when no
    runs were found and the caller received the default; the MultiPacker should
    adopt the first real run's config instead of evicting it.
    """
    multi_run_manager = get_multi_run_manager()
    multi_run_manager.discover_runs()

    run_targets = {
        _resolve_batch_target_from_orchestrator_config(multi_run_manager.config[run_idx])
        for run_idx in multi_run_manager.used_idxs
        if run_idx in multi_run_manager.config
    }

    if not run_targets:
        return default_token_batch_size, "tokens", True

    if len(run_targets) > 1:
        raise ValueError("All runs must share the same batching config.")

    target, unit = next(iter(run_targets))
    return target, unit, False


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
        token_batch_size: int | None = None,
        rollout_batch_size: int | None = None,
        start_step: int = 0,
    ):
        super().__init__(dp_world_size, seq_len, pad_to_multiple_of, tokenizer, config, start_step)
        assert self.multi_run_manager.max_runs == 1, "SinglePacker only supports one run"
        self.batch_target, self.batch_unit = _resolve_batch_target(token_batch_size, rollout_batch_size)
        # The rollout dir was cleaned in BasePacker.__init__, so the orchestrator
        # will write files starting from step_0. Override the receiver's lazy
        # initialization (which falls back to progress[0].step) to match.
        self.receiver.set_start_step(0, 0)

    def _sample_batch_units(self, sample: TrainingSample) -> int:
        if self.batch_unit == "tokens":
            return _sample_num_tokens(sample)
        return 1

    def pack(self):
        """Accumulate samples from streamed group rollouts until the batch budget is met."""
        accumulated_samples: list[TrainingSample] = []
        total_batch_units = 0

        pbar = ProgressTracker(
            total=self.batch_target,
            desc="Accumulating total tokens" if self.batch_unit == "tokens" else "Accumulating rollouts",
        )

        while total_batch_units < self.batch_target:
            self.multi_run_manager.discover_runs()
            batches = self.receiver.receive()

            if not batches:
                time.sleep(0.2)
                continue

            for batch in batches:
                for sample in batch.examples:
                    accumulated_samples.append(sample)
                    sample_batch_units = self._sample_batch_units(sample)
                    total_batch_units += sample_batch_units
                    pbar.update(sample_batch_units)

        pbar.close()

        self.multi_run_manager.ready_to_update[0] = True
        self.multi_run_manager.progress[0].step += 1
        micro_batch_grid = prepare_batch(
            rollouts=accumulated_samples,
            seq_len=self.seq_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            num_train_workers=self.dp_world_size,
            idxs=[0] * len(accumulated_samples),
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
        token_batch_size: int | None = None,
        rollout_batch_size: int | None = None,
        start_step: int = 0,
        batch_config_provisional: bool = False,
    ):
        super().__init__(dp_world_size, seq_len, pad_to_multiple_of, tokenizer, config, start_step)
        self.batch_target, self.batch_unit = _resolve_batch_target(
            token_batch_size,
            rollout_batch_size,
            default_token_batch_size=self.seq_len * self.dp_world_size,
        )
        self._batch_config_provisional = batch_config_provisional
        # Per-run buffer: stores (TrainingSample, step) tuples
        self.buffers: list[deque[tuple[TrainingSample, int]]] = [
            deque() for _ in range(self.multi_run_manager.max_runs)
        ]

        # Round-robin position (persists across pack() calls)
        self._round_robin_position: int = 0

        # Register forgotten hook for receiver reset (master only, called during discover_runs)
        # This must happen when a run is deleted to prevent stale data from remaining
        self.multi_run_manager.register_forgotten_hook(self._on_run_data_deleted)
        self.multi_run_manager.register_discovered_hook(self._on_run_discovered)

        # TrainingBatch.step counters are local to orchestrator process lifetime and
        # start at 0 on resume/restart. Align receiver state to step 0 for all current runs.
        for idx in self.multi_run_manager.used_idxs:
            self.receiver.set_start_step(idx, 0)

    def _on_run_data_deleted(self, idx: int, run_id: str) -> None:
        """Reset run state when run data is deleted (master only)."""
        self.logger.debug(f"Packing is resetting run state for deleted run {idx}")
        self.receiver.reset_run(idx)

        # Reset run state
        self.buffers[idx].clear()

    def _on_run_discovered(self, idx: int, run_id: str, config) -> None:
        """Align receiver state for newly discovered runs (master only)."""
        run_target, run_unit = _resolve_batch_target_from_orchestrator_config(config)
        if self._batch_config_provisional:
            self.logger.info(
                f"Adopting batch config from run {idx} ({run_id}): {run_unit}={run_target}"
            )
            self.batch_target = run_target
            self.batch_unit = run_unit
            self._batch_config_provisional = False
        elif run_target != self.batch_target or run_unit != self.batch_unit:
            self.multi_run_manager.evict_run(
                idx,
                f"Run batching config ({run_unit}={run_target}) does not match trainer batching "
                f"({self.batch_unit}={self.batch_target})",
            )
            return
        self.logger.debug(f"Packing is setting receiver start step to zero for discovered run {idx} ({run_id})")
        self.receiver.set_start_step(idx, 0)

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

    def _sample_batch_units(self, sample: TrainingSample) -> int:
        if self.batch_unit == "tokens":
            return _sample_num_tokens(sample)
        return 1

    def _count_total_batch_units(self, threshold: int | None = None) -> int:
        total_units = 0

        for run_idx in self.multi_run_manager.used_idxs:
            buffer = self.buffers[run_idx]
            current_step = self.multi_run_manager.progress[run_idx].step

            for sample, step in buffer:
                if step > current_step:
                    break
                total_units += self._sample_batch_units(sample)
                if threshold is not None and total_units >= threshold:
                    return total_units
        return total_units

    def _has_enough_batch_units(self) -> bool:
        """Check if we have enough total work in buffer to pack a step."""
        return self._count_total_batch_units(self.batch_target) >= self.batch_target

    def _select_samples_round_robin(self) -> list[tuple[int, TrainingSample, int]]:
        """Select samples using round-robin from runs with buffered work."""
        selected: list[tuple[int, TrainingSample, int]] = []
        collected_units = 0

        while collected_units < self.batch_target:
            # Round-robin until we find a run with work for the current step
            for _ in range(len(self.buffers)):
                if len(self.buffers[self._round_robin_position]) > 0:
                    _, step = self.buffers[self._round_robin_position][0]
                    if step <= self.multi_run_manager.progress[self._round_robin_position].step:
                        break
                self._round_robin_position = (self._round_robin_position + 1) % len(self.buffers)
            else:
                # TODO: We could probably make the logic safer. This is basically counting on _has_enough_batch_items() to be correct.
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
                sample_batch_units = self._sample_batch_units(sample)
                if collected_units > 0 and collected_units + sample_batch_units > self.batch_target:
                    return selected
                selected.append((run_idx, sample, step))
                self.buffers[run_idx].popleft()
                collected_units += sample_batch_units
                if collected_units >= self.batch_target:
                    return selected

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

        while not self._has_enough_batch_units():
            if time.time() - start_time > TIMEOUT_SECONDS and self._count_total_batch_units() > 0:
                self.logger.warning(f"Timeout waiting for enough {self.batch_unit} to pack")
                break
            time.sleep(1)
            self._get_batch()

        selected_samples = self._select_samples_round_robin()
        assert selected_samples, "No samples selected"

        # Group samples by run_idx - each microbatch must contain samples from only ONE run
        # because MultiLoRAGroupedExperts (MoE) only supports one adapter per microbatch
        samples_by_run: dict[int, list[TrainingSample]] = {}
        per_run_stats: dict[int, tuple[int, int]] = {}
        for run_idx, sample, step in selected_samples:
            if run_idx not in samples_by_run:
                samples_by_run[run_idx] = []
            samples_by_run[run_idx].append(sample)

            num_tokens = _sample_num_tokens(sample)
            if run_idx in per_run_stats:
                cur_samples, cur_tokens = per_run_stats[run_idx]
                per_run_stats[run_idx] = (cur_samples + 1, cur_tokens + num_tokens)
            else:
                per_run_stats[run_idx] = (1, num_tokens)

        for run_idx, (num_samples, num_tokens) in per_run_stats.items():
            self._update_run_progress(run_idx, num_samples, num_tokens)

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
    token_batch_size: int | None = None,
    rollout_batch_size: int | None = None,
    start_step: int = 0,
) -> BasePacker:
    multi_run_manager = get_multi_run_manager()
    batch_config_provisional = False
    if token_batch_size is None and rollout_batch_size is None:
        batch_target, batch_unit, batch_config_provisional = _resolve_batch_target_from_discovered_runs(
            default_token_batch_size=seq_len * dp_world_size
        )
        if batch_unit == "tokens":
            token_batch_size = batch_target
        else:
            rollout_batch_size = batch_target

    if multi_run_manager.max_runs == 1:
        return SinglePacker(
            dp_world_size,
            seq_len,
            pad_to_multiple_of,
            tokenizer,
            transport_config,
            token_batch_size=token_batch_size,
            rollout_batch_size=rollout_batch_size,
            start_step=start_step,
        )
    else:
        return MultiPacker(
            dp_world_size,
            seq_len,
            pad_to_multiple_of,
            tokenizer,
            transport_config,
            token_batch_size,
            rollout_batch_size,
            start_step,
            batch_config_provisional=batch_config_provisional,
        )
