import shutil
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path

import msgspec
import numpy as np

from prime_rl.transports.batch.types import MicroBatch, encode_array
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_all_ckpt_steps, get_batch_dir, get_step_path


class BatchRequest(msgspec.Struct):
    """A trainer rank asking for its micro batches of one step."""

    world_size: int
    rank: int
    data_rank: int
    data_world_size: int
    step: int
    committed: int
    """The newest step a trainer checkpoint covers; a retaining sender releases the steps up to it."""


class BatchLog:
    """Shipped steps on disk, one file per data rank (``<batch_dir>/step_<N>/rank_<K>.bin``).
    A step is written ahead of its delivery and published with it, so every published step
    is one the trainer was handed."""

    def __init__(self, output_dir: Path):
        self.dir = get_batch_dir(output_dir)

    def path(self, step: int, data_rank: int) -> Path:
        return get_step_path(self.dir, step) / f"rank_{data_rank}.bin"

    def write(self, step: int, payloads: list[bytes]) -> None:
        """Write the step's unpublished files, replacing any written for another number of data ranks."""
        step_path = get_step_path(self.dir, step)
        step_path.mkdir(parents=True, exist_ok=True)
        for path in step_path.glob("rank_*.bin.tmp"):
            path.unlink()
        for data_rank, payload in enumerate(payloads):
            self.path(step, data_rank).with_suffix(".bin.tmp").write_bytes(payload)

    def publish(self, step: int, data_world_size: int) -> None:
        for data_rank in range(data_world_size):
            path = self.path(step, data_rank)
            path.with_suffix(".bin.tmp").rename(path)

    def read(self, step: int, data_rank: int) -> bytes | None:
        try:
            return self.path(step, data_rank).read_bytes()
        except FileNotFoundError:
            return None

    def data_world_size(self, step: int) -> int | None:
        """The number of data ranks the step was published for, None until it is published."""
        step_path = get_step_path(self.dir, step)
        if any(step_path.glob("rank_*.bin.tmp")):
            return None
        return len(list(step_path.glob("rank_*.bin"))) or None

    def prune(self, through: int) -> None:
        """Delete the steps up to ``through``."""
        for step in get_all_ckpt_steps(self.dir):
            if step <= through:
                shutil.rmtree(get_step_path(self.dir, step), ignore_errors=True)


def dummy_micro_batch(source: MicroBatch) -> MicroBatch:
    """A zero-loss copy of a micro batch, preserving its modality."""
    num_tokens = sum(source.sequence_lengths)
    has_routing = any(payload is not None and "routed_experts" in payload.fields for payload in source.payloads or [])
    return msgspec.structs.replace(
        source,
        advantages=np.zeros(num_tokens, dtype=np.float32),
        loss_mask=np.zeros(num_tokens, dtype=np.bool_),
        # ce/ref_kl membership is weight != 0 (independent of loss_mask), so the
        # streams must go too or the dummy would still train those tokens.
        rl_weights=None,
        ce_weights=None,
        ref_kl_weights=None,
        # Fully loss-masked: fetch only the routing rows the forward pass replays, since
        # replaying sampling masks would be pure wasted work.
        payloads=[msgspec.structs.replace(payload, fields=["routed_experts"]) for payload in source.payloads]
        if has_routing
        else None,
        # The copied identity would double-annotate the source's traces.
        trace_ids=None,
        branch_indices=None,
    )


def _trains_nothing(micro_batch: MicroBatch) -> bool:
    weights = (micro_batch.rl_weights, micro_batch.ce_weights, micro_batch.ref_kl_weights)
    return all(stream is None for stream in weights) and not np.frombuffer(micro_batch.loss_mask, np.bool_).any()


def redistribute(micro_batch_grid: list[list[MicroBatch]], data_world_size: int) -> list[list[MicroBatch]]:
    """Deal a step's micro batches to another number of data ranks.

    Every rank gets the same number of micro batches, multimodal ones first, so all ranks
    run the same modality at each index (see ``prepare_batch``); each modality is padded
    with dummies to a multiple of ``data_world_size``. Dummies of the old split are dropped.
    Within a modality the micro batches go out in rounds, largest first to the rank holding
    the fewest tokens."""
    micro_batches = [mb for rank in micro_batch_grid for mb in rank if not _trains_nothing(mb)]
    if not micro_batches:
        micro_batches = [micro_batch_grid[0][0]]
    grid: list[list[MicroBatch]] = [[] for _ in range(data_world_size)]
    for multimodal in (True, False):
        group = [mb for mb in micro_batches if (mb.mm_refs is not None) == multimodal]
        if not group:
            continue
        group += [dummy_micro_batch(group[0])] * (-len(group) % data_world_size)
        group.sort(key=lambda mb: -sum(mb.sequence_lengths))
        loads = [0] * data_world_size
        for start in range(0, len(group), data_world_size):
            ranks = sorted(range(data_world_size), key=lambda rank: loads[rank])
            for rank, micro_batch in zip(ranks, group[start : start + data_world_size]):
                grid[rank].append(micro_batch)
                loads[rank] += sum(micro_batch.sequence_lengths)
    return grid


class BatchSender(ABC):
    """Base class for sending micro batches from the orchestrator to the train workers.

    Trainer ranks pull: a step goes out only once every trainer rank requested it, so the
    trainer only ever holds the batch it is training on, and the last step sent is always the
    one the trainer is on. A trainer that asks with another number of data ranks gets the
    step dealt to that number, and ``data_world_size`` follows it."""

    def __init__(self, output_dir: Path, data_world_size: int, retain: bool = False):
        self.logger = get_logger()
        # Per-token streams are numpy arrays; each goes out as its raw bytes.
        self.encoder = msgspec.msgpack.Encoder(enc_hook=encode_array)
        self.output_dir = output_dir
        self.data_world_size = data_world_size
        self.current_step = 0
        # Seconds the last send waited for the trainer to request its step
        self.last_wait_s = 0.0
        self.retain = retain
        """Keep every delivered step in the ``BatchLog`` until a trainer checkpoint covers it."""
        self.log = BatchLog(output_dir)

    @abstractmethod
    async def send(self, micro_batch_grid: list[list[MicroBatch]]) -> float:
        """Send grid of micro batches to the trainers once every trainer rank requested the step.
        Returns the seconds spent handing the step to the transport, excluding the wait."""
        pass

    def deal(self, micro_batch_grid: list[list[MicroBatch]], buffers: list[bytes], data_world_size: int) -> list[bytes]:
        """The step's encoded micro batches for the number of data ranks the trainer asked
        with: ``buffers`` if that is what the step was packed for."""
        if data_world_size == self.data_world_size:
            return buffers
        self.logger.info(
            f"The trainer asked for step {self.current_step} with {data_world_size} data ranks, "
            f"not {self.data_world_size}: dealing its micro batches again"
        )
        self.data_world_size = data_world_size
        return [self.encode(micro_batches) for micro_batches in redistribute(micro_batch_grid, data_world_size)]

    def encode(self, micro_batches: list[MicroBatch]) -> bytes:
        """Byte-identical to ``self.encoder.encode(micro_batches)``, but encodes one
        micro batch at a time so a worker thread releases the GIL between them."""
        n = len(micro_batches)
        if n <= 1:
            return self.encoder.encode(micro_batches)
        if n < 16:
            header = bytes([0x90 | n])
        elif n < 2**16:
            header = b"\xdc" + n.to_bytes(2, "big")
        else:
            header = b"\xdd" + n.to_bytes(4, "big")
        with BytesIO() as payload:
            payload.write(header)
            for micro_batch in micro_batches:
                payload.write(self.encoder.encode(micro_batch))
            return payload.getvalue()

    def close(self) -> None:
        """Clean up any resources. Override if needed."""
        pass


class BatchReceiver(ABC):
    """Base class for receiving micro batches from the orchestrator."""

    def __init__(
        self, output_dir: Path, data_rank: int, data_world_size: int, rank: int, world_size: int, current_step: int
    ):
        self.logger = get_logger()
        self.decoder = msgspec.msgpack.Decoder(type=list[MicroBatch])
        self.output_dir = output_dir
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.rank = rank
        self.world_size = world_size
        self.current_step = current_step
        self.committed = current_step - 1
        """The newest step a trainer checkpoint covers, sent with every request."""
        self.log = BatchLog(output_dir)

    def request(self) -> BatchRequest:
        return BatchRequest(
            self.world_size, self.rank, self.data_rank, self.data_world_size, self.current_step, self.committed
        )

    def read_log(self) -> list[MicroBatch] | None:
        """This rank's micro batches of the current step if it is published, dealt to this
        trainer's number of data ranks if it was published for another."""
        published_for = self.log.data_world_size(self.current_step)
        if published_for is None:
            return None
        if published_for == self.data_world_size:
            return self.decoder.decode(self.log.read(self.current_step, self.data_rank))
        grid = [self.decoder.decode(self.log.read(self.current_step, rank)) for rank in range(published_for)]
        return redistribute(grid, self.data_world_size)[self.data_rank]

    @abstractmethod
    def wait(self) -> None:
        """Request the current step from the orchestrator and block until it is available."""
        pass

    @abstractmethod
    def receive(self) -> list[MicroBatch]:
        """Receive the micro batches ``wait`` requested."""
        pass

    def close(self) -> None:
        """Clean up any resources. Override if needed."""
        pass
