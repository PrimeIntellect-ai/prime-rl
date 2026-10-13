import asyncio
from pathlib import Path

from prime_rl.trainer.world import get_world
from prime_rl.transports.batch.base import BatchReceiver, BatchSender
from prime_rl.transports.batch.types import MicroBatch
from prime_rl.utils.pathing import get_batch_dir, get_step_path, sync_wait_for_path


def _write_atomic(path: Path, data: bytes) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        f.write(data)
    tmp_path.rename(path)


class FileSystemBatchSender(BatchSender):
    """Filesystem-based micro batch sender. Every trainer rank drops a request
    marker into a step directory before reading it; the sender writes step N
    only once every rank has requested step N-1, so at most one unread step
    sits on disk."""

    def __init__(self, output_dir: Path, data_world_size: int, current_step: int = 0):
        super().__init__(output_dir, data_world_size)
        self.batch_dir = get_batch_dir(output_dir)
        self.current_step = current_step
        self.start_step = current_step

    async def send(self, micro_batch_grid: list[list[MicroBatch]]) -> None:
        assert len(micro_batch_grid) == self.data_world_size, "Number of micro batch lists must match data world size"
        for micro_batch_list in micro_batch_grid:
            assert len(micro_batch_list) == len(micro_batch_grid[0]), "All micro batch lists must have the same length"

        if self.current_step > self.start_step:
            await self._wait_for_requests(self.current_step - 1)
        step_path = get_step_path(self.batch_dir, self.current_step)
        step_path.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._encode_and_write, micro_batch_grid, step_path)
        self.current_step += 1

    async def _wait_for_requests(self, step: int) -> None:
        """Block until every trainer rank has asked for ``step``. The markers
        carry the trainer world size, so the first one tells how many to expect."""
        step_path = get_step_path(self.batch_dir, step)
        self.logger.debug(f"Waiting for all ranks to request step {step}")
        while True:
            markers = list(step_path.glob("rank_*.request"))
            if markers and len(markers) == int(markers[0].read_text()):
                return
            await asyncio.sleep(0.1)

    def _encode_and_write(self, micro_batch_grid: list[list[MicroBatch]], step_path: Path) -> None:
        for data_rank in range(self.data_world_size):
            _write_atomic(step_path / f"rank_{data_rank}.bin", self.encoder.encode(micro_batch_grid[data_rank]))


class FileSystemBatchReceiver(BatchReceiver):
    """Filesystem-based micro batch receiver that reads micro batches from disk."""

    def __init__(self, output_dir: Path, data_rank: int, current_step: int = 0):
        super().__init__(output_dir, data_rank)
        self.world = get_world()
        self.batch_dir = get_batch_dir(output_dir)
        self.current_step = current_step

    def _get_micro_batch_path(self) -> Path:
        return get_step_path(self.batch_dir, self.current_step) / f"rank_{self.data_rank}.bin"

    def wait(self) -> None:
        step_path = get_step_path(self.batch_dir, self.current_step)
        step_path.mkdir(parents=True, exist_ok=True)
        _write_atomic(step_path / f"rank_{self.world.rank}.request", str(self.world.world_size).encode("utf-8"))
        sync_wait_for_path(self._get_micro_batch_path())

    def receive(self) -> list[MicroBatch]:
        with open(self._get_micro_batch_path(), "rb") as f:
            micro_batches: list[MicroBatch] = self.decoder.decode(f.read())
        self.current_step += 1
        return micro_batches
