import asyncio
import time
from pathlib import Path

from prime_rl.transports.batch.base import BatchReceiver, BatchSender
from prime_rl.transports.batch.types import MicroBatch
from prime_rl.utils.pathing import get_batch_dir, get_step_path, sync_wait_for_path

REQUEST_POLL_INTERVAL_S = 0.1


def write_atomic(path: Path, data: bytes) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(data)
    tmp_path.rename(path)


class FileSystemBatchSender(BatchSender):
    """Filesystem-based micro batch sender that writes micro batches to disk. Every trainer rank
    drops a request marker holding the trainer world size into the step directory, and the step is
    published only once every rank's marker is there."""

    def __init__(self, output_dir: Path, data_world_size: int, current_step: int = 0):
        super().__init__(output_dir, data_world_size)
        self.batch_dir = get_batch_dir(output_dir)
        self.current_step = current_step

    async def send(self, micro_batch_grid: list[list[MicroBatch]]) -> float:
        """Send grid of micro batches to the trainers. Encode + write run in a worker thread so the
        orchestrator event loop stays responsive, and before the wait, so only the rename sits
        between the last request and the batch."""
        assert len(micro_batch_grid) == self.data_world_size, "Number of micro batch lists must match data world size"
        for micro_batch_list in micro_batch_grid:
            assert len(micro_batch_list) == len(micro_batch_grid[0]), "All micro batch lists must have the same length"

        step_path = get_step_path(self.batch_dir, self.current_step)
        step_path.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._encode_and_write, micro_batch_grid, step_path)
        await self._wait_for_requests(step_path)
        send_start = time.perf_counter()
        for data_rank in range(self.data_world_size):
            (step_path / f"rank_{data_rank}.bin.tmp").rename(step_path / f"rank_{data_rank}.bin")
        send_end = time.perf_counter()

        self.current_step += 1
        return send_end - send_start

    def _encode_and_write(self, micro_batch_grid: list[list[MicroBatch]], step_path: Path) -> None:
        for data_rank in range(self.data_world_size):
            (step_path / f"rank_{data_rank}.bin.tmp").write_bytes(self.encoder.encode(micro_batch_grid[data_rank]))

    async def _wait_for_requests(self, step_path: Path) -> None:
        """Block until every trainer rank requested the step. The markers hold the trainer world
        size, so the first one says how many to wait for."""
        while True:
            request_paths = list(step_path.glob("rank_*.request"))
            if request_paths and len(request_paths) == int(request_paths[0].read_text()):
                return
            await asyncio.sleep(REQUEST_POLL_INTERVAL_S)


class FileSystemBatchReceiver(BatchReceiver):
    """Filesystem-based micro batch receiver that reads micro batches from disk."""

    def __init__(self, output_dir: Path, data_rank: int, rank: int, world_size: int, current_step: int = 0):
        super().__init__(output_dir, data_rank, rank, world_size)
        self.batch_dir = get_batch_dir(output_dir)
        self.current_step = current_step

    def _get_micro_batch_path(self) -> Path:
        return get_step_path(self.batch_dir, self.current_step) / f"rank_{self.data_rank}.bin"

    def wait(self) -> None:
        """Request the current step, then wait for its micro batch file to appear on disk."""
        step_path = get_step_path(self.batch_dir, self.current_step)
        step_path.mkdir(parents=True, exist_ok=True)
        write_atomic(step_path / f"rank_{self.rank}.request", str(self.world_size).encode())
        sync_wait_for_path(self._get_micro_batch_path())

    def receive(self) -> list[MicroBatch]:
        """Read and return the micro batches from disk."""
        with open(self._get_micro_batch_path(), "rb") as f:
            micro_batches: list[MicroBatch] = self.decoder.decode(f.read())
        self.current_step += 1
        return micro_batches
