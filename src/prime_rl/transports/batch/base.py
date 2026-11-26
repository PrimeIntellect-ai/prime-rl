import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path

import msgspec

from prime_rl.transports.batch.types import StepBatch
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_all_ckpt_steps, get_batch_dir, get_step_path


class BatchRequest(msgspec.Struct):
    """A trainer rank asking for its micro batches of one step."""

    world_size: int
    rank: int
    step: int
    committed: int
    """The newest step a trainer checkpoint covers; a retaining sender releases the steps up to it."""
    requested_at: float = 0.0
    """Wall-clock time the rank asked (0: not reported). The trainer was busy until then."""


def last_requested_at(requests: list[BatchRequest]) -> float | None:
    """When the last of a step's ranks asked for it, None if any rank did not report it."""
    if not requests or any(request.requested_at <= 0 for request in requests):
        return None
    return max(request.requested_at for request in requests)


class BatchLog:
    """Shipped steps on disk, one file per step (``<batch_dir>/step_<N>/batch.bin``). A step
    is written ahead of its delivery and published with it, so every published step is one
    the trainer was handed."""

    def __init__(self, output_dir: Path):
        self.dir = get_batch_dir(output_dir)

    def path(self, step: int) -> Path:
        return get_step_path(self.dir, step) / "batch.bin"

    def write(self, step: int, payload: bytes) -> None:
        get_step_path(self.dir, step).mkdir(parents=True, exist_ok=True)
        self.path(step).with_suffix(".bin.tmp").write_bytes(payload)

    def publish(self, step: int) -> None:
        path = self.path(step)
        path.with_suffix(".bin.tmp").rename(path)

    def read(self, step: int) -> bytes | None:
        try:
            return self.path(step).read_bytes()
        except FileNotFoundError:
            return None

    def prune(self, through: int) -> None:
        """Delete the steps up to ``through``."""
        for step in get_all_ckpt_steps(self.dir):
            if step <= through:
                shutil.rmtree(get_step_path(self.dir, step), ignore_errors=True)


class BatchSender(ABC):
    """Base class for sending a step's samples from the orchestrator to the trainer ranks.

    Trainer ranks pull: a step goes out only once every trainer rank requested it, so the
    trainer only ever holds the batch it is training on, and the last step sent is always the
    one the trainer is on. Every rank gets the same bytes and plans its own micro batches, so
    the sender knows nothing of the trainer's layout."""

    def __init__(self, output_dir: Path, retain: bool = False):
        self.logger = get_logger()
        self.encoder = msgspec.msgpack.Encoder()
        self.output_dir = output_dir
        # Seconds the last send waited for the trainer to request its step, and when its last rank asked
        self.last_wait_s = 0.0
        self.last_requested_at: float | None = None
        self.retain = retain
        """Keep every delivered step in the ``BatchLog`` until a trainer checkpoint covers it."""
        self.log = BatchLog(output_dir)

    @abstractmethod
    async def send(self, batch: StepBatch) -> float:
        """Send a step's samples to the trainer once every trainer rank requested the step.
        Returns the seconds spent handing the step to the transport, excluding the wait."""
        pass

    def close(self) -> None:
        """Clean up any resources. Override if needed."""
        pass


class BatchReceiver(ABC):
    """Base class for receiving a step's samples from the orchestrator."""

    def __init__(self, output_dir: Path, rank: int, world_size: int, current_step: int):
        self.logger = get_logger()
        self.decoder = msgspec.msgpack.Decoder(type=StepBatch)
        self.output_dir = output_dir
        self.rank = rank
        self.world_size = world_size
        self.current_step = current_step
        self.committed = current_step - 1
        """The newest step a trainer checkpoint covers, sent with every request."""
        self.log = BatchLog(output_dir)

    def request(self) -> BatchRequest:
        return BatchRequest(self.world_size, self.rank, self.current_step, self.committed, requested_at=time.time())

    @abstractmethod
    def wait(self) -> None:
        """Request the current step from the orchestrator and block until it is available."""
        pass

    @abstractmethod
    def receive(self) -> StepBatch:
        """Receive the step ``wait`` requested."""
        pass

    def close(self) -> None:
        """Clean up any resources. Override if needed."""
        pass
