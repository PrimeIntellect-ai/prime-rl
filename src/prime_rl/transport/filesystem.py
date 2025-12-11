from pathlib import Path

from prime_rl.transport.base import MicroBatchReceiver, MicroBatchSender, TrainingBatchReceiver, TrainingBatchSender
from prime_rl.transport.types import MicroBatch, TrainingBatch
from prime_rl.utils.pathing import get_rollout_dir, get_step_path, sync_wait_for_path

BATCH_FILE_NAME = "rollouts.bin"


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
        with open(step_path / BATCH_FILE_NAME, "wb") as f:
            f.write(buffer)


class FileSystemTrainingBatchReceiver(TrainingBatchReceiver):
    """Filesystem-based training batch receiver that reads batches from disk."""

    def __init__(self, output_dir: Path, current_step: int = 0):
        super().__init__(output_dir)
        self.rollout_dir = get_rollout_dir(output_dir)
        self.current_step = current_step

    def _get_batch_path(self) -> Path:
        return get_step_path(self.rollout_dir, self.current_step) / BATCH_FILE_NAME

    def wait(self) -> None:
        """Wait for the batch file to appear on disk."""
        sync_wait_for_path(self._get_batch_path())

    def can_receive(self) -> bool:
        """Check if the batch file exists."""
        return self._get_batch_path().exists()

    def receive(self) -> TrainingBatch:
        """Read and return the batch from disk."""
        with open(self._get_batch_path(), "rb") as f:
            batch: TrainingBatch = self.decoder.decode(f.read())
        self.current_step += 1
        return batch


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
            with open(step_path / f"rank_{data_rank}.bin", "wb") as f:
                f.write(buffer)
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
