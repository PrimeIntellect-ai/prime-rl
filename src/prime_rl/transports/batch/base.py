import os
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
    offered: bool = False
    """The rank holds the step from its offer, so the sender hands it over without the bytes."""


def last_requested_at(requests: list[BatchRequest]) -> float | None:
    """When the last of a step's ranks asked for it, None if any rank did not report it."""
    if not requests or any(request.requested_at <= 0 for request in requests):
        return None
    return max(request.requested_at for request in requests)


class BatchLog:
    """Shipped steps on disk, one file per step (``<batch_dir>/step_<N>/batch.bin``). A step
    is written ahead of its delivery and published with it, so every published step is one
    the trainer was handed. An offered step is readable before that, as ``offer.bin``."""

    def __init__(self, output_dir: Path):
        self.dir = get_batch_dir(output_dir)

    def path(self, step: int) -> Path:
        return get_step_path(self.dir, step) / "batch.bin"

    def write(self, step: int, payload: bytes) -> None:
        get_step_path(self.dir, step).mkdir(parents=True, exist_ok=True)
        self.path(step).with_suffix(".bin.tmp").write_bytes(payload)

    def offer(self, step: int) -> None:
        """Make the written step readable ahead of its delivery, as the same file where the
        filesystem links (``offer.bin``)."""
        written, offer = self.path(step).with_suffix(".bin.tmp"), self.offer_path(step)
        linked = offer.with_name(".offer.bin.tmp")
        linked.unlink(missing_ok=True)
        try:
            os.link(written, linked)
        except OSError:
            shutil.copyfile(written, linked)
        linked.replace(offer)

    def offer_path(self, step: int) -> Path:
        return get_step_path(self.dir, step) / "offer.bin"

    def read_offer(self, step: int) -> bytes | None:
        try:
            return self.offer_path(step).read_bytes()
        except FileNotFoundError:
            return None

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

    Trainer ranks pull: a step counts as sent only once every trainer rank requested it, so the
    last step sent is always the one the trainer is on. The sender offers each step's bytes as
    soon as it has the step, so a rank can plan and build its micro batches while it still
    trains the step before; the request still marks when the trainer took the step. Every rank
    gets the same bytes and plans its own micro batches, so the sender knows nothing of the
    trainer's layout."""

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
        """Offer a step's samples to the trainer, and hand them over once every trainer rank
        requested the step. Returns the seconds spent handing the step over, excluding the wait."""
        pass

    def close(self) -> None:
        """Clean up any resources. Override if needed."""
        pass


class BatchReceiver(ABC):
    """Base class for receiving a step's samples from the orchestrator. ``poll`` picks up the
    current step as soon as the sender offers it; ``wait`` requests it, which is when the
    trainer takes it, and blocks until the sender handed it over."""

    poll_interval: float = 0.05
    """Seconds ``poll`` waits per call for a step that is not there yet."""

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
        self.payload: bytes | memoryview | None = None
        """The current step's bytes, once offered or handed over."""
        self.published = False
        """The current step was read from the log: the sender handed it over before."""
        self._batch: StepBatch | None = None

    def request(self) -> BatchRequest:
        return BatchRequest(
            self.world_size,
            self.rank,
            self.current_step,
            self.committed,
            requested_at=time.time(),
            offered=self.payload is not None,
        )

    @abstractmethod
    def poll(self) -> bool:
        """Whether the current step's bytes are here, without requesting the step: offered by the
        sender, or published in the log. Waits up to ``poll_interval`` for them."""
        pass

    @abstractmethod
    def wait(self) -> None:
        """Request the current step and block until the sender handed it over."""
        pass

    def batch(self) -> StepBatch:
        """The current step, once ``poll`` or ``wait`` has it."""
        if self._batch is None:
            assert self.payload is not None, "the current step is not here yet"
            self._batch = self.decoder.decode(self.payload)
        return self._batch

    def receive(self) -> StepBatch:
        """The step ``wait`` got, after which the receiver moves on to the next step."""
        batch = self.batch()
        self.payload, self.published, self._batch = None, False, None
        self.logger.debug(f"Received {len(batch.samples)} samples for step {self.current_step}")
        self.current_step += 1
        return batch

    def close(self) -> None:
        """Clean up any resources. Override if needed."""
        pass
