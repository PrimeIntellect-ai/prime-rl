from pathlib import Path
from time import time

from prime_rl.trainer.runs import get_runs
from prime_rl.transport.base import MicroBatchReceiver, MicroBatchSender, TrainingBatchReceiver, TrainingBatchSender
from prime_rl.transport.types import MicroBatch, TrainingBatch
from prime_rl.utils.pathing import get_rollout_dir, get_step_path, sync_wait_for_path

BATCH_FILE_TMP_NAME = "rollouts.bin.tmp"
BATCH_FILE_NAME = "rollouts.bin"
LOG_FREQ_SECONDS = 10


class FileSystemTrainingBatchSender(TrainingBatchSender):
    """Filesystem-based training batch sender that writes batches to disk."""

    def __init__(self, output_dir: Path):
        super().__init__(output_dir)
        self.rollout_dir = get_rollout_dir(output_dir)

    def send(self, batch: TrainingBatch) -> None:
        """Send a batch by writing it to disk"""
        step_path = get_step_path(self.rollout_dir, batch.step)
        step_path.mkdir(parents=True, exist_ok=True)

        buffer = self.encoder.encode(batch)
        tmp_path = step_path / BATCH_FILE_TMP_NAME
        with open(tmp_path, "wb") as f:
            f.write(buffer)
        tmp_path.rename(step_path / BATCH_FILE_NAME)


class FileSystemTrainingBatchReceiver(TrainingBatchReceiver):
    """Filesystem-based training batch receiver that reads batches from multiple run directories."""

    def __init__(self) -> None:
        super().__init__()
        self.runs = get_runs()
        self._last_logged_paths: list[Path] | None = None
        self._last_logged_time = time()
        self._waiting_since: float | None = None
        # Track received steps per run independently of runs.progress[idx].step
        # This prevents duplicate reads when trainer step != orchestrator step
        self._received_steps: dict[int, int] = {}

    def _get_received_step(self, idx: int) -> int:
        """Get the next step to receive for a run, initializing from progress if needed."""
        if idx not in self._received_steps:
            # Initialize from runs.progress on first access (for checkpoint resume)
            self._received_steps[idx] = self.runs.progress[idx].step
        return self._received_steps[idx]

    def _get_batch_path(self, idx: int) -> Path:
        """Get the batch file path for a specific run at its next step to receive."""
        run_dir = self.runs.get_run_dir(idx)
        rollout_dir = get_rollout_dir(run_dir)
        step = self._get_received_step(idx)
        return get_step_path(rollout_dir, step) / BATCH_FILE_NAME

    def can_receive(self) -> bool:
        """Check if any run has a batch file available."""
        for idx in self.runs.used_idxs:
            if not self.runs.ready_to_update[idx] and self._get_batch_path(idx).exists():
                return True
        return False

    def receive(self) -> list[TrainingBatch]:
        """Read and return all available batches from all runs."""
        batches: list[TrainingBatch] = []
        now = time()

        # Track how long we've been waiting for any new batch to appear.
        if self.can_receive():
            self._waiting_since = None
        else:
            self._waiting_since = self._waiting_since or now

        current_paths = [self._get_batch_path(idx) for idx in self.runs.used_idxs]
        if current_paths != self._last_logged_paths or now - self._last_logged_time > LOG_FREQ_SECONDS:
            if len(current_paths) == 0:
                self.logger.debug(
                    "Did you set the output dir of the orchestrator to a run_* subdirectory of the trainer output dir?"
                )
            waiting_suffix = ""
            if self._waiting_since is not None:
                waiting_suffix = f" (waiting {now - self._waiting_since:.1f}s)"
            self.logger.debug(f"Looking for batches in {current_paths}{waiting_suffix}")
            self._last_logged_paths = current_paths
            self._last_logged_time = now
        for idx in list(self.runs.used_idxs):
            if self.runs.ready_to_update[idx]:
                continue
            batch_path = self._get_batch_path(idx)
            if batch_path.exists():
                try:
                    with open(batch_path, "rb") as f:
                        batch: TrainingBatch = self.decoder.decode(f.read())
                    batch.run_idx = idx
                    batches.append(batch)
                    # Increment received step to avoid reading the same file again
                    self._received_steps[idx] = self._get_received_step(idx) + 1
                except Exception as e:
                    self.logger.error(f"Error loading rollouts for run {idx}: {e}")
        return batches

    def reset_run(self, idx: int) -> None:
        """Reset received step tracking for a run index.

        Called when a run is deleted and a new run takes its place.
        The next access to _get_received_step will re-initialize from runs.progress.
        """
        if idx in self._received_steps:
            del self._received_steps[idx]


class FileSystemMicroBatchSender(MicroBatchSender):
    """Filesystem-based micro batch sender that writes micro batches to disk."""

    def __init__(self, output_dir: Path, data_world_size: int, current_step: int = 0):
        super().__init__(output_dir, data_world_size)
        self.rollout_dir = get_rollout_dir(output_dir)
        self.current_step = current_step

    def send(self, micro_batch_grid: list[list[MicroBatch]]) -> None:
        """Send grid of micro batches to the trainers."""
        # Validation
        assert len(micro_batch_grid) == self.data_world_size, "Number of micro batch lists must match data world size"
        for micro_batch_list in micro_batch_grid:
            assert len(micro_batch_list) == len(micro_batch_grid[0]), "All micro batch lists must have the same length"

        step_path = get_step_path(self.rollout_dir, self.current_step)
        step_path.mkdir(parents=True, exist_ok=True)

        for data_rank in range(self.data_world_size):
            buffer = self.encoder.encode(micro_batch_grid[data_rank])
            tmp_path = step_path / f"rank_{data_rank}.bin.tmp"
            with open(tmp_path, "wb") as f:
                f.write(buffer)
            tmp_path.rename(step_path / f"rank_{data_rank}.bin")
        self.current_step += 1


class FileSystemMicroBatchReceiver(MicroBatchReceiver):
    """Filesystem-based micro batch receiver that reads micro batches from disk."""

    def __init__(self, output_dir: Path, data_rank: int, current_step: int = 0):
        super().__init__(output_dir, data_rank)
        self.rollout_dir = get_rollout_dir(output_dir)
        self.current_step = current_step

    def _get_micro_batch_path(self) -> Path:
        return get_step_path(self.rollout_dir, self.current_step) / f"rank_{self.data_rank}.bin"

    def wait(self) -> None:
        """Wait for the micro batch file to appear on disk."""
        sync_wait_for_path(self._get_micro_batch_path())

    def can_receive(self) -> bool:
        """Check if the micro batch file exists."""
        return self._get_micro_batch_path().exists()

    def receive(self) -> list[MicroBatch]:
        """Read and return the micro batches from disk."""
        with open(self._get_micro_batch_path(), "rb") as f:
            micro_batches: list[MicroBatch] = self.decoder.decode(f.read())
        self.current_step += 1
        return micro_batches
