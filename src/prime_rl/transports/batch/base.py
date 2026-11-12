import shutil
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path

import msgspec
import numpy as np

from prime_rl.transports.batch.types import MICRO_BATCH_DTYPES, MicroBatch, encode_array
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_all_ckpt_steps, get_batch_dir, get_step_path


class BatchRequest(msgspec.Struct):
    """A trainer rank asking for its micro batches of one step."""

    world_size: int
    rank: int
    data_rank: int
    data_world_size: int
    micro_batch_multiple: int
    """The trainer splits every micro batch over its context-parallel ranks, so it needs their
    length to be a multiple of this."""
    step: int
    committed: int
    """The newest step a trainer checkpoint covers; a retaining sender releases the steps up to it."""

    @property
    def shape(self) -> tuple[int, int, int]:
        """The trainer's shape as far as its batches depend on it. Ranks of one trainer agree on it."""
        return self.world_size, self.data_world_size, self.micro_batch_multiple


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


# Backfill value per component weight stream when a packed sample doesn't
# carry it: absent rl means weight 1.0 on the loss mask, absent ce/ref_kl
# means no component (weight 0.0).
STREAM_FILL = {"rl_weights": 1.0, "ce_weights": 0.0, "ref_kl_weights": 0.0}


def pad_micro_batch(micro_batch: MicroBatch, pad_to_multiple_of: int) -> MicroBatch:
    """
    Pad a micro batch with loss-masked tokens up to the next multiple of `pad_to_multiple_of`.
    Args:
        micro_batch: The micro batch to pad.
        pad_to_multiple_of: The multiple to pad the sequence length up to.
    Returns:
        The padded micro batch.
    """

    padding_size = (pad_to_multiple_of - (len(micro_batch.input_ids) % pad_to_multiple_of)) % pad_to_multiple_of

    if len(micro_batch.env_names) != len(micro_batch.sequence_lengths):
        raise ValueError(
            f"MicroBatch.env_names must name every sequence: "
            f"env_names={len(micro_batch.env_names)}, sequences={len(micro_batch.sequence_lengths)}"
        )

    if not (pad_to_multiple_of > 1 and padding_size > 0):
        return micro_batch

    # Identical pad tokens route to the same MoE experts and can overload one EP rank, so text rows pad with their
    # own tokens. Multimodal rows keep a constant id: a repeated image placeholder would have no feature to receive.
    # `np.pad` appends zeros of the stream's own dtype unless given another value.
    tail = (0, padding_size)
    if micro_batch.mm_refs is None:
        # `np.resize` repeats the row end to end.
        pad_ids = np.resize(micro_batch.input_ids, padding_size)
        micro_batch.input_ids = np.concatenate([micro_batch.input_ids, pad_ids])
    else:
        micro_batch.input_ids = np.pad(micro_batch.input_ids, tail, constant_values=1)
    micro_batch.advantages = np.pad(micro_batch.advantages, tail)
    micro_batch.loss_mask = np.pad(micro_batch.loss_mask, tail)
    micro_batch.position_ids = np.concatenate([micro_batch.position_ids, np.arange(padding_size, dtype=np.int32)])
    # Replaced, not changed in place: a dummy shares these lists with the micro batch it copies.
    micro_batch.sequence_lengths = [*micro_batch.sequence_lengths[:-1], micro_batch.sequence_lengths[-1] + padding_size]
    micro_batch.seq_lens = [*micro_batch.seq_lens[:-1], micro_batch.seq_lens[-1] + padding_size]
    micro_batch.padding += padding_size
    micro_batch.inference_logprobs = np.pad(micro_batch.inference_logprobs, tail)
    # Use temperature 1.0 for padding tokens (doesn't matter since loss_mask is False)
    micro_batch.temperatures = np.pad(micro_batch.temperatures, tail, constant_values=1.0)
    if micro_batch.ref_logprobs is not None:
        micro_batch.ref_logprobs = np.pad(micro_batch.ref_logprobs, tail)
    # Padding is loss-masked, so no component trains it; fill every stream
    # with 0.0 (not the pack-boundary defaults) so a padded pure-ce batch
    # still reads as rl-empty to consumers that key off nonzero weights
    # (e.g. the per-env mismatch metrics).
    for stream_name in STREAM_FILL:
        stream = getattr(micro_batch, stream_name)
        if stream is not None:
            setattr(micro_batch, stream_name, np.pad(stream, tail))
    if micro_batch.mm_token_type_ids is not None:
        micro_batch.mm_token_type_ids = np.pad(micro_batch.mm_token_type_ids, tail)
    if micro_batch.staleness is not None:
        micro_batch.staleness = np.pad(micro_batch.staleness, tail, constant_values=-1)

    return micro_batch


def as_arrays(micro_batch: MicroBatch) -> MicroBatch:
    """A decoded micro batch with its per-token streams as numpy arrays, as the packer holds them."""
    streams = {name: getattr(micro_batch, name) for name in MICRO_BATCH_DTYPES}
    return msgspec.structs.replace(
        micro_batch,
        **{
            name: np.frombuffer(stream, MICRO_BATCH_DTYPES[name])
            for name, stream in streams.items()
            if stream is not None
        },
    )


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
    one the trainer is on. A trainer that asks with another number of data ranks or another
    micro batch multiple gets the step fitted to them, and ``data_world_size`` and
    ``micro_batch_multiple`` follow it."""

    def __init__(self, output_dir: Path, data_world_size: int, micro_batch_multiple: int = 1, retain: bool = False):
        self.logger = get_logger()
        # Per-token streams are numpy arrays; each goes out as its raw bytes.
        self.encoder = msgspec.msgpack.Encoder(enc_hook=encode_array)
        self.output_dir = output_dir
        self.data_world_size = data_world_size
        self.micro_batch_multiple = micro_batch_multiple
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

    def fit(self, micro_batch_grid: list[list[MicroBatch]], buffers: list[bytes], request: BatchRequest) -> list[bytes]:
        """The step's encoded micro batches for the shape the trainer asked with: ``buffers`` if
        that is what the step was packed for."""
        shape = (request.data_world_size, request.micro_batch_multiple)
        if shape == (self.data_world_size, self.micro_batch_multiple):
            return buffers
        self.logger.info(
            f"The trainer asked for step {self.current_step} with {shape[0]} data ranks and micro batches "
            f"in multiples of {shape[1]}, not {self.data_world_size} and {self.micro_batch_multiple}: "
            "fitting the step to it"
        )
        if request.data_world_size != self.data_world_size:
            micro_batch_grid = redistribute(micro_batch_grid, request.data_world_size)
        self.data_world_size, self.micro_batch_multiple = shape
        return [
            self.encode([pad_micro_batch(micro_batch, self.micro_batch_multiple) for micro_batch in micro_batches])
            for micro_batches in micro_batch_grid
        ]

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
        self,
        output_dir: Path,
        data_rank: int,
        data_world_size: int,
        micro_batch_multiple: int,
        rank: int,
        world_size: int,
        current_step: int,
    ):
        self.logger = get_logger()
        self.decoder = msgspec.msgpack.Decoder(type=list[MicroBatch])
        self.output_dir = output_dir
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.micro_batch_multiple = micro_batch_multiple
        self.rank = rank
        self.world_size = world_size
        self.current_step = current_step
        self.committed = current_step - 1
        """The newest step a trainer checkpoint covers, sent with every request."""
        self.log = BatchLog(output_dir)

    def request(self) -> BatchRequest:
        return BatchRequest(
            world_size=self.world_size,
            rank=self.rank,
            data_rank=self.data_rank,
            data_world_size=self.data_world_size,
            micro_batch_multiple=self.micro_batch_multiple,
            step=self.current_step,
            committed=self.committed,
        )

    def read_log(self) -> list[MicroBatch] | None:
        """This rank's micro batches of the current step if it is published, fitted to this
        trainer's number of data ranks and micro batch multiple if it was published for others."""
        published_for = self.log.data_world_size(self.current_step)
        if published_for is None:
            return None
        if published_for == self.data_world_size:
            micro_batches = self.decoder.decode(self.log.read(self.current_step, self.data_rank))
        else:
            grid = [self.decoder.decode(self.log.read(self.current_step, rank)) for rank in range(published_for)]
            micro_batches = redistribute(grid, self.data_world_size)[self.data_rank]
        return [
            pad_micro_batch(as_arrays(micro_batch), self.micro_batch_multiple)
            if sum(micro_batch.sequence_lengths) % self.micro_batch_multiple
            else micro_batch
            for micro_batch in micro_batches
        ]

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
