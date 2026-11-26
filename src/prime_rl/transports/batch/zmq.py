import asyncio
import time
from collections.abc import Collection
from pathlib import Path

import msgspec
import zmq
import zmq.asyncio

from prime_rl.configs.shared import ZMQTransportConfig
from prime_rl.transports.batch.log import BatchLog
from prime_rl.transports.batch.types import StepBatch
from prime_rl.utils.logger import get_logger

POLL_INTERVAL_MS = 5
"""Milliseconds ``BatchReceiver.poll`` waits per call for a step that is not there yet."""
LOG_POLL_INTERVAL_MS = 1000

OFFER, STEP = b"offer", b"step"
"""Message kinds to a rank: a step's bytes ahead of its request, and the step handed over (empty
for a rank that holds the offer)."""


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


class BatchSender:
    """Sends a step's samples from the orchestrator to the trainer ranks.

    Trainer ranks pull: a step counts as sent only once every trainer rank requested it, so the
    last step sent is always the one the trainer is on. The sender offers each step's bytes as
    soon as it has the step, so a rank can plan and build its micro batches while it still
    trains the step before; the request still marks when the trainer took the step. Every rank
    gets the same bytes and plans its own micro batches, so the sender knows nothing of the
    trainer's layout.

    One ROUTER socket in the orchestrator answers the DEALER socket of every trainer rank,
    replying to each rank on the connection it asked on, and offers each step on the
    connections the previous step's requests came on."""

    def __init__(self, output_dir: Path, current_step: int, transport: ZMQTransportConfig):
        self.logger = get_logger()
        self.encoder = msgspec.msgpack.Encoder()
        self.request_decoder = msgspec.msgpack.Decoder(type=BatchRequest)
        self.current_step = current_step
        self.retain = transport.retain
        """Keep every delivered step in the ``BatchLog`` until a trainer checkpoint covers it."""
        self.log = BatchLog(output_dir)
        # Seconds the last send waited for the trainer to request its step, and when its last rank asked
        self.last_wait_s = 0.0
        self.last_requested_at: float | None = None
        self.identities: dict[int, bytes] = {}
        """Each rank's connection, from its last request."""
        # Async context so ``send`` yields instead of blocking the orchestrator event loop
        self.context = zmq.asyncio.Context.instance()
        self.socket: zmq.asyncio.Socket = self.context.socket(zmq.ROUTER)
        # Replying to a rank that went away raises instead of dropping silently
        self.socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.socket.bind(f"tcp://{transport.host}:{transport.port}")
        self.logger.debug(f"Initialized ZMQ step sender (endpoint=tcp://{transport.host}:{transport.port})")

    async def send(self, batch: StepBatch) -> float:
        """Offer a step's samples to the trainer, and hand them over once every trainer rank
        requested the step. Returns the seconds spent handing the step over, excluding the wait.
        Encoding happens before the wait, so only the hand-off sits between the last request and
        the batch."""
        payload = await asyncio.to_thread(self.encoder.encode, batch)
        if self.retain:
            await asyncio.to_thread(self.log.write, self.current_step, payload)
        step = self.current_step.to_bytes(8, "big")
        # A rank that asked already waits for the step itself: an offer would send it twice.
        rank_to_request = await self._queued_requests()
        await self._offer(step, payload, asked=rank_to_request.keys())
        wait_start = time.perf_counter()
        rank_to_request = await self._wait_for_requests(rank_to_request)
        send_start = time.perf_counter()
        self.last_wait_s = send_start - wait_start
        if self.retain:
            self.log.publish(self.current_step)
        for identity, request in rank_to_request.values():
            try:
                await self.socket.send_multipart(
                    [identity, STEP, step, b"" if request.offered else payload], copy=False
                )
            except zmq.ZMQError as error:
                # A rank that died after asking: its replacement reads the published step.
                if not self.retain or error.errno != zmq.EHOSTUNREACH:
                    raise
                self.logger.warning(f"Trainer rank {request.rank} is gone; step {self.current_step} stays retained")
        self.identities = {request.rank: identity for identity, request in rank_to_request.values()}
        send_end = time.perf_counter()
        if self.retain:
            committed = max(request.committed for _, request in rank_to_request.values())
            await asyncio.to_thread(self.log.prune, committed)

        self.logger.debug(f"Sent step {self.current_step} to {len(rank_to_request)} ranks")
        self.current_step += 1
        return send_end - send_start

    async def _offer(self, step: bytes, payload: bytes, asked: Collection[int]) -> None:
        """Hand the step's bytes to every rank that has not ``asked`` for it yet. A rank gone
        since (a restarted trainer) gets the step when it asks."""
        for rank, identity in self.identities.items():
            if rank in asked:
                continue
            try:
                await self.socket.send_multipart([identity, OFFER, step, payload], copy=False)
            except zmq.ZMQError as error:
                if error.errno != zmq.EHOSTUNREACH:
                    raise
                self.logger.debug(f"Trainer rank {rank} is gone; it gets step {self.current_step} when it asks")

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
        finally:
            self.logger.debug("Closed ZMQ step sender")

    def _current_request(self, payload: bytes) -> BatchRequest | None:
        """The request, if it is for the current step. A rank requests the next step only after
        receiving this one, so a later step means the orchestrator and trainer resumed from
        different steps. A retaining sender skips requests for earlier steps (None): a restarted
        rank reads those from the log."""
        request = self.request_decoder.decode(payload)
        if request.step == self.current_step:
            return request
        if not self.retain or request.step > self.current_step:
            raise RuntimeError(
                f"A trainer rank requested step {request.step}, but the orchestrator is on step "
                f"{self.current_step}. Resume both from the same checkpoint step."
            )
        return None

    async def _receive_request(self) -> tuple[bytes, BatchRequest]:
        while True:
            identity, payload = await self.socket.recv_multipart()
            if (request := self._current_request(payload)) is not None:
                return identity, request

    async def _queued_requests(self) -> dict[int, tuple[bytes, BatchRequest]]:
        """The requests for the current step that arrived already, without waiting for more."""
        queued: dict[int, tuple[bytes, BatchRequest]] = {}
        while await self.socket.poll(timeout=0):
            identity, payload = await self.socket.recv_multipart()
            if (request := self._current_request(payload)) is not None:
                queued[request.rank] = (identity, request)
        return queued

    async def _wait_for_requests(
        self, rank_to_request: dict[int, tuple[bytes, BatchRequest]]
    ) -> dict[int, tuple[bytes, BatchRequest]]:
        """Collect one request per trainer rank for the current step, on top of those already
        here. Each request carries the trainer world size, so the first one says how many to wait
        for. A rank that asks again (a restarted trainer) replaces its earlier request."""
        if not rank_to_request:
            identity, request = await self._receive_request()
            rank_to_request = {request.rank: (identity, request)}
        world_size = next(iter(rank_to_request.values()))[1].world_size
        while len(rank_to_request) < world_size:
            identity, request = await self._receive_request()
            rank_to_request[request.rank] = (identity, request)
        self.last_requested_at = last_requested_at([request for _, request in rank_to_request.values()])
        return rank_to_request


class BatchReceiver:
    """Receives a step's samples from the orchestrator. ``poll`` picks up the current step as
    soon as the sender offers it; ``wait`` requests it, which is when the trainer takes it, and
    blocks until the sender handed it over.

    Every trainer rank owns one DEALER socket, on which the sender offers each step ahead of the
    rank's request, and requests each step before training it. With ``transport.retain``, a step
    the sender already published is read from the log instead, which is how a restarted trainer
    gets the steps since its checkpoint."""

    def __init__(
        self,
        output_dir: Path,
        rank: int,
        world_size: int,
        current_step: int,
        transport: ZMQTransportConfig,
    ):
        self.logger = get_logger()
        self.decoder = msgspec.msgpack.Decoder(type=StepBatch)
        self.request_encoder = msgspec.msgpack.Encoder()
        self.rank = rank
        self.world_size = world_size
        self.current_step = current_step
        self.committed = current_step - 1
        """The newest step a trainer checkpoint covers, sent with every request."""
        self.retain = transport.retain
        self.log = BatchLog(output_dir)
        self.payload: bytes | memoryview | None = None
        """The current step's bytes, once offered or handed over."""
        self.published = False
        """The current step was read from the log: the sender handed it over before."""
        self._batch: StepBatch | None = None
        self._log_checked: int | None = None
        self.context = zmq.Context.instance()
        self.socket: zmq.Socket = self.context.socket(zmq.DEALER)
        self.socket.connect(f"tcp://{transport.host}:{transport.port}")
        self.logger.debug(f"Initialized ZMQ step receiver (endpoint=tcp://{transport.host}:{transport.port})")

    def request(self) -> BatchRequest:
        return BatchRequest(
            self.world_size,
            self.rank,
            self.current_step,
            self.committed,
            requested_at=time.time(),
            offered=self.payload is not None,
        )

    def poll(self) -> bool:
        """Whether the current step's bytes are here, without requesting the step: offered by the
        sender, or published in the log. Waits up to ``POLL_INTERVAL_MS`` for them."""
        # A step is published before its request only for a restarted trainer: once per step is enough.
        if self.payload is None and self.retain and self._log_checked != self.current_step:
            self._log_checked = self.current_step
            self.payload = self.log.read(self.current_step)
            self.published = self.payload is not None
        if self.payload is None and self.socket.poll(timeout=POLL_INTERVAL_MS):
            kind, step, payload = self.socket.recv_multipart(copy=False)
            # Anything else is a late reply to a request for a step that was then read from the log.
            if kind.bytes == OFFER and int.from_bytes(step.bytes, "big") == self.current_step:
                self.payload = payload
        return self.payload is not None

    def wait(self) -> None:
        """Request the current step and block until the sender handed it over."""
        if self.published:
            return
        requested = False
        while True:
            if self.retain and self.payload is None:
                self.payload = self.log.read(self.current_step)
                if self.payload is not None:
                    self.published = True
                    return
            if not requested:
                self.socket.send(self.request_encoder.encode(self.request()))
                requested = True
            if self.socket.poll(timeout=LOG_POLL_INTERVAL_MS if self.retain else None):
                kind, step, payload = self.socket.recv_multipart(copy=False)
                if int.from_bytes(step.bytes, "big") != self.current_step:
                    continue  # a late reply to a request for a step that was then read from the log
                if kind.bytes == OFFER:
                    # Offered after this rank asked without it: the step itself follows.
                    if self.payload is None:
                        self.payload = payload
                    continue
                if len(payload):
                    self.payload = payload
                if self.payload is not None:
                    return

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
        try:
            self.socket.close(linger=0)
        finally:
            self.logger.debug("Closed ZMQ step receiver")
