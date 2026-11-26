import asyncio
import time
from pathlib import Path

import msgspec
import zmq
import zmq.asyncio

from prime_rl.configs.shared import ZMQTransportConfig
from prime_rl.transports.batch.base import BatchReceiver, BatchRequest, BatchSender, last_requested_at
from prime_rl.transports.batch.types import StepBatch

LOG_POLL_INTERVAL_MS = 1000

OFFER, STEP = b"offer", b"step"
"""Message kinds to a rank: a step's bytes ahead of its request, and the step handed over (empty
for a rank that holds the offer)."""


class ZMQBatchSender(BatchSender):
    def __init__(self, output_dir: Path, current_step: int, transport: ZMQTransportConfig):
        """ZMQ step sender. One ROUTER socket in the orchestrator answers the DEALER socket of
        every trainer rank, replying to each rank on the connection it asked on, and offers
        each step on the connections the previous step's requests came on."""
        super().__init__(output_dir, transport.retain)
        # Async context so ``send`` yields instead of blocking the orchestrator event loop
        self.context = zmq.asyncio.Context.instance()
        self.socket: zmq.asyncio.Socket = self.context.socket(zmq.ROUTER)
        # Replying to a rank that went away raises instead of dropping silently
        self.socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.socket.bind(f"tcp://{transport.host}:{transport.port}")
        self.request_decoder = msgspec.msgpack.Decoder(type=BatchRequest)
        self.identities: dict[int, bytes] = {}
        """Each rank's connection, from its last request."""
        self.current_step = current_step
        self.logger.debug(f"Initialized ZMQ step sender (endpoint=tcp://{transport.host}:{transport.port})")

    async def send(self, batch: StepBatch) -> float:
        """Send a step's samples to the trainer. Encoding happens before the wait, so only the
        hand-off sits between the last request and the batch."""
        payload = await asyncio.to_thread(self.encoder.encode, batch)
        if self.retain:
            await asyncio.to_thread(self.log.write, self.current_step, payload)
        step = self.current_step.to_bytes(8, "big")
        await self._offer(step, payload)
        wait_start = time.perf_counter()
        rank_to_request = await self._wait_for_requests()
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

    async def _offer(self, step: bytes, payload: bytes) -> None:
        """Hand the step's bytes to every rank ahead of its request. A rank gone since (a
        restarted trainer) gets the step when it asks."""
        for rank, identity in self.identities.items():
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

    async def _receive_request(self) -> tuple[bytes, BatchRequest]:
        """A rank requests the next step only after receiving this one, so any other step means the
        orchestrator and trainer resumed from different steps. A retaining sender skips requests
        for earlier steps: a restarted rank reads those from the log."""
        while True:
            identity, payload = await self.socket.recv_multipart()
            request = self.request_decoder.decode(payload)
            if request.step == self.current_step:
                return identity, request
            if not self.retain or request.step > self.current_step:
                raise RuntimeError(
                    f"A trainer rank requested step {request.step}, but the orchestrator is on step "
                    f"{self.current_step}. Resume both from the same checkpoint step."
                )

    async def _wait_for_requests(self) -> dict[int, tuple[bytes, BatchRequest]]:
        """Collect one request per trainer rank for the current step. Each request carries the
        trainer world size, so the first one says how many to wait for. A rank that asks again
        (a restarted trainer) replaces its earlier request."""
        identity, request = await self._receive_request()
        rank_to_request = {request.rank: (identity, request)}
        while len(rank_to_request) < request.world_size:
            identity, request = await self._receive_request()
            rank_to_request[request.rank] = (identity, request)
        self.last_requested_at = last_requested_at([request for _, request in rank_to_request.values()])
        return rank_to_request


class ZMQBatchReceiver(BatchReceiver):
    def __init__(
        self,
        output_dir: Path,
        rank: int,
        world_size: int,
        current_step: int,
        transport: ZMQTransportConfig,
    ):
        """ZMQ step receiver. Every trainer rank owns one DEALER socket, on which the sender offers
        each step ahead of the rank's request, and requests each step before training it. With
        ``transport.retain``, a step the sender already published is read from the log instead,
        which is how a restarted trainer gets the steps since its checkpoint."""
        super().__init__(output_dir, rank, world_size, current_step)
        self.context = zmq.Context.instance()
        self.socket: zmq.Socket = self.context.socket(zmq.DEALER)
        self.socket.connect(f"tcp://{transport.host}:{transport.port}")
        self.request_encoder = msgspec.msgpack.Encoder()
        self.retain = transport.retain
        self._log_checked: int | None = None
        self.logger.debug(f"Initialized ZMQ step receiver (endpoint=tcp://{transport.host}:{transport.port})")

    poll_interval = 0.005

    def poll(self) -> bool:
        # A step is published before its request only for a restarted trainer: once per step is enough.
        if self.payload is None and self.retain and self._log_checked != self.current_step:
            self._log_checked = self.current_step
            self.payload = self.log.read(self.current_step)
            self.published = self.payload is not None
        if self.payload is None and self.socket.poll(timeout=int(self.poll_interval * 1000)):
            kind, step, payload = self.socket.recv_multipart(copy=False)
            # Anything else is a late reply to a request for a step that was then read from the log.
            if kind.bytes == OFFER and int.from_bytes(step.bytes, "big") == self.current_step:
                self.payload = payload
        return self.payload is not None

    def wait(self) -> None:
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

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
        finally:
            self.logger.debug("Closed ZMQ step receiver")
