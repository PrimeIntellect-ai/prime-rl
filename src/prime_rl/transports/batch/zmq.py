import time
from pathlib import Path

import msgspec
import zmq
import zmq.asyncio

from prime_rl.configs.shared import ZMQTransportConfig
from prime_rl.transports.batch.base import BatchReceiver, BatchSender
from prime_rl.transports.batch.types import MicroBatch


class BatchRequest(msgspec.Struct):
    """A trainer rank asking for its micro batches of one step."""

    world_size: int
    data_rank: int
    step: int


class ZMQBatchSender(BatchSender):
    def __init__(self, output_dir: Path, data_world_size: int, current_step: int, transport: ZMQTransportConfig):
        """ZMQ micro batch sender. One ROUTER socket in the orchestrator answers the DEALER socket of
        every trainer rank, replying to each rank on the connection it asked on."""
        super().__init__(output_dir, data_world_size)
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

        buffers = [self.encoder.encode(micro_batch_list) for micro_batch_list in micro_batch_grid]
        identity_to_request = await self._wait_for_requests()
        send_start = time.perf_counter()
        for identity, request in identity_to_request.items():
            await self.socket.send_multipart([identity, buffers[request.data_rank]], copy=False)
        send_end = time.perf_counter()

        self.logger.debug(f"Sent micro batch grid for step {self.current_step} to {len(identity_to_request)} ranks")
        self.current_step += 1
        return send_end - send_start

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
        finally:
            self.logger.debug("Closed ZMQ micro batch sender")

    async def _receive_request(self) -> tuple[bytes, BatchRequest]:
        """A rank requests the next step only after receiving this one, so any other step means the
        orchestrator and trainer resumed from different steps."""
        identity, payload = await self.socket.recv_multipart()
        request = self.request_decoder.decode(payload)
        if request.step != self.current_step:
            raise RuntimeError(
                f"A trainer rank requested step {request.step}, but the orchestrator is on step "
                f"{self.current_step}. Resume both from the same checkpoint step."
            )
        return identity, request

    async def _wait_for_requests(self) -> dict[bytes, BatchRequest]:
        """Collect one request per trainer rank for the current step. Each request carries the
        trainer world size, so the first one says how many to wait for."""
        identity, request = await self._receive_request()
        identity_to_request = {identity: request}
        while len(identity_to_request) < request.world_size:
            identity, request = await self._receive_request()
            identity_to_request[identity] = request
        return identity_to_request


class ZMQBatchReceiver(BatchReceiver):
    def __init__(
        self,
        output_dir: Path,
        data_rank: int,
        rank: int,
        world_size: int,
        current_step: int,
        transport: ZMQTransportConfig,
    ):
        """ZMQ micro batch receiver. Every trainer rank owns one DEALER socket and requests each step
        before reading it."""
        super().__init__(output_dir, data_rank, rank, world_size)
        self.context = zmq.Context.instance()
        self.socket: zmq.Socket = self.context.socket(zmq.DEALER)
        self.socket.connect(f"tcp://{transport.host}:{transport.port}")
        self.request_encoder = msgspec.msgpack.Encoder()
        self.current_step = current_step
        self.logger.debug(f"Initialized ZMQ micro batch receiver (endpoint=tcp://{transport.host}:{transport.port})")

    def wait(self) -> None:
        request = BatchRequest(
            world_size=self.world_size,
            data_rank=self.data_rank,
            step=self.current_step,
        )
        self.socket.send(self.request_encoder.encode(request))
        self.socket.poll(timeout=None)

    def receive(self) -> list[MicroBatch]:
        """Receive a micro batch from the orchestrator."""
        payload = self.socket.recv(copy=False)
        micro_batches: list[MicroBatch] = self.decoder.decode(payload)
        self.logger.debug(f"Received {len(micro_batches)} micro batches for step {self.current_step}")
        self.current_step += 1
        return micro_batches

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
        finally:
            self.logger.debug("Closed ZMQ micro batch receiver")
