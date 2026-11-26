import asyncio
import time
from pathlib import Path

import msgspec

from prime_rl.transports.batch.base import BatchReceiver, BatchRequest, BatchSender, last_requested_at
from prime_rl.transports.batch.types import StepBatch
from prime_rl.utils.pathing import get_step_path, sync_wait_for_path

REQUEST_POLL_INTERVAL_S = 0.1


def write_atomic(path: Path, data: bytes) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(data)
    tmp_path.rename(path)


class FileSystemBatchSender(BatchSender):
    """Filesystem-based sender that writes each step's samples to disk. Every trainer rank drops
    its request into the step directory, and the step is published only once every rank's
    request is there."""

    def __init__(self, output_dir: Path, current_step: int = 0, retain: bool = False):
        super().__init__(output_dir, retain)
        self.current_step = current_step

    async def send(self, batch: StepBatch) -> float:
        """Send a step's samples to the trainer. Encode + write run in a worker thread so the
        orchestrator event loop stays responsive, and before the wait, so only the rename sits
        between the last request and the batch."""
        await asyncio.to_thread(self._encode_and_write, batch)
        wait_start = time.perf_counter()
        committed = await self._wait_for_requests()
        send_start = time.perf_counter()
        self.last_wait_s = send_start - wait_start
        self.log.publish(self.current_step)
        send_end = time.perf_counter()
        if self.retain:
            await asyncio.to_thread(self.log.prune, committed)

        self.current_step += 1
        return send_end - send_start

    def _encode_and_write(self, batch: StepBatch) -> None:
        self.log.write(self.current_step, self.encoder.encode(batch))

    async def _wait_for_requests(self) -> int:
        """Block until every trainer rank requested the step; returns the step the trainer has
        committed. Each request holds the trainer world size, so the first one says how many to
        wait for."""
        step_path = get_step_path(self.log.dir, self.current_step)
        while True:
            requests = [
                msgspec.msgpack.decode(path.read_bytes(), type=BatchRequest)
                for path in step_path.glob("rank_*.request")
            ]
            if requests and len(requests) == requests[0].world_size:
                self.last_requested_at = last_requested_at(requests)
                return max(request.committed for request in requests)
            await asyncio.sleep(REQUEST_POLL_INTERVAL_S)


class FileSystemBatchReceiver(BatchReceiver):
    """Filesystem-based receiver that reads each step's samples from disk."""

    def __init__(self, output_dir: Path, rank: int, world_size: int, current_step: int = 0):
        super().__init__(output_dir, rank, world_size, current_step)

    def wait(self) -> None:
        """Request the current step, then wait for its file to appear on disk."""
        step_path = get_step_path(self.log.dir, self.current_step)
        step_path.mkdir(parents=True, exist_ok=True)
        write_atomic(step_path / f"rank_{self.rank}.request", msgspec.msgpack.encode(self.request()))
        sync_wait_for_path(self.log.path(self.current_step))

    def receive(self) -> StepBatch:
        """Read and return the step from disk."""
        batch: StepBatch = self.decoder.decode(self.log.read(self.current_step))
        self.current_step += 1
        return batch
