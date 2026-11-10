import asyncio
import time
from pathlib import Path

import msgspec
import zmq
import zmq.asyncio

from prime_rl.configs.shared import ZMQTransportConfig
from prime_rl.transports.batch.base import BatchReceiver, BatchRequest, BatchSender
from prime_rl.transports.batch.types import MicroBatch

LOG_POLL_INTERVAL_MS = 1000


class ZMQBatchSender(BatchSender):
    def __init__(
        self,
        output_dir: Path,
        data_world_size: int,
        current_step: int,
        transport: ZMQTransportConfig,
        micro_batch_multiple: int = 1,
    ):
        """ZMQ micro batch sender. One ROUTER socket in the orchestrator answers the DEALER socket of
        every trainer rank, replying to each rank on the connection it asked on."""
        super().__init__(output_dir, data_world_size, micro_batch_multiple, transport.retain)
        # Async context so ``send`` yields instead of blocking the orchestrator event loop
        self.context = zmq.asyncio.Context.instance()
        self.socket: zmq.asyncio.Socket = self.context.socket(zmq.ROUTER)
        # Replying to a rank that went away raises instead of dropping silently
        self.socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.socket.bind(f"tcp://{transport.host}:{transport.port}")
        self.request_decoder = msgspec.msgpack.Decoder(type=BatchRequest)
        self.current_step = current_step
        self.logger.debug(f"Initialized ZMQ micro batch sender (endpoint=tcp://{transport.host}:{transport.port})")

    async def send(self, micro_batch_grid: list[list[MicroBatch]]) -> float:
        """Send grid of micro batches to the trainers. Encoding happens before the wait, so only the
        hand-off sits between the last request and the batch."""
        assert len(micro_batch_grid) == self.data_world_size, "Number of micro batch lists must match data world size"
        for micro_batch_list in micro_batch_grid:
            assert len(micro_batch_list) == len(micro_batch_grid[0]), "All micro batch lists must have the same length"

        buffers = [await asyncio.to_thread(self.encode, micro_batch_list) for micro_batch_list in micro_batch_grid]
        if self.retain:
            await asyncio.to_thread(self.log.write, self.current_step, buffers)
        wait_start = time.perf_counter()
        rank_to_request = await self._wait_for_requests()
        _, request = next(iter(rank_to_request.values()))
        fitted = await asyncio.to_thread(self.fit, micro_batch_grid, buffers, request)
        if fitted is not buffers:
            buffers = fitted
            if self.retain:
                await asyncio.to_thread(self.log.write, self.current_step, buffers)
        send_start = time.perf_counter()
        self.last_wait_s = send_start - wait_start
        if self.retain:
            self.log.publish(self.current_step, self.data_world_size)
        step = self.current_step.to_bytes(8, "big")
        for identity, request in rank_to_request.values():
            try:
                await self.socket.send_multipart([identity, step, buffers[request.data_rank]], copy=False)
            except zmq.ZMQError as error:
                # A rank that died after asking: its replacement reads the published step.
                if not self.retain or error.errno != zmq.EHOSTUNREACH:
                    raise
                self.logger.warning(f"Trainer rank {request.rank} is gone; step {self.current_step} stays retained")
        send_end = time.perf_counter()
        if self.retain:
            committed = max(request.committed for _, request in rank_to_request.values())
            await asyncio.to_thread(self.log.prune, committed)

        self.logger.debug(f"Sent micro batch grid for step {self.current_step} to {len(rank_to_request)} ranks")
        self.current_step += 1
        return send_end - send_start

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
        finally:
            self.logger.debug("Closed ZMQ micro batch sender")

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
        trainer world size, which says how many to wait for. A rank that asks again (a restarted
        trainer) replaces its earlier request, and a request with another world size (a trainer
        restarted at another size) drops the ones collected so far."""
        rank_to_request: dict[int, tuple[bytes, BatchRequest]] = {}
        while True:
            identity, request = await self._receive_request()
            if any(other.world_size != request.world_size for _, other in rank_to_request.values()):
                rank_to_request = {}
            rank_to_request[request.rank] = (identity, request)
            if len(rank_to_request) == request.world_size:
                return rank_to_request


class ZMQBatchReceiver(BatchReceiver):
    def __init__(
        self,
        output_dir: Path,
        data_rank: int,
        data_world_size: int,
        micro_batch_multiple: int,
        rank: int,
        world_size: int,
        current_step: int,
        transport: ZMQTransportConfig,
    ):
        """ZMQ micro batch receiver. Every trainer rank owns one DEALER socket and requests each step
        before reading it. With ``transport.retain``, a step the sender already published is read
        from the log instead, which is how a restarted trainer gets the steps since its checkpoint."""
        super().__init__(output_dir, data_rank, data_world_size, micro_batch_multiple, rank, world_size, current_step)
        self.context = zmq.Context.instance()
        self.socket: zmq.Socket = self.context.socket(zmq.DEALER)
        self.socket.connect(f"tcp://{transport.host}:{transport.port}")
        self.request_encoder = msgspec.msgpack.Encoder()
        self.retain = transport.retain
        self.micro_batches: list[MicroBatch] | None = None
        self.logger.debug(f"Initialized ZMQ micro batch receiver (endpoint=tcp://{transport.host}:{transport.port})")

    def wait(self) -> None:
        requested = False
        while True:
            if self.retain:
                self.micro_batches = self.read_log()
                if self.micro_batches is not None:
                    return
            if not requested:
                self.socket.send(self.request_encoder.encode(self.request()))
                requested = True
            if self.socket.poll(timeout=LOG_POLL_INTERVAL_MS if self.retain else None):
                step, payload = self.socket.recv_multipart(copy=False)
                # A reply to a request for a step that was then read from the log arrives late.
                if int.from_bytes(step.bytes, "big") == self.current_step:
                    self.micro_batches = self.decoder.decode(payload)
                    return

    def receive(self) -> list[MicroBatch]:
        """The micro batches ``wait`` got from the orchestrator or the log."""
        micro_batches, self.micro_batches = self.micro_batches, None
        self.logger.debug(f"Received {len(micro_batches)} micro batches for step {self.current_step}")
        self.current_step += 1
        return micro_batches

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
        finally:
            self.logger.debug("Closed ZMQ micro batch receiver")
