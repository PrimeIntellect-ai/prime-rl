import asyncio
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
    rank: int
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
        self.last_sent: list[bytes] = []
        """The last step's encoded micro batches, for a restarted rank that had not saved them yet."""
        self.logger.debug(f"Initialized ZMQ micro batch sender (endpoint=tcp://{transport.host}:{transport.port})")

    async def send(self, micro_batch_grid: list[list[MicroBatch]]) -> float:
        """Send grid of micro batches to the trainers. Encoding happens before the wait, so only the
        hand-off sits between the last request and the batch."""
        assert len(micro_batch_grid) == self.data_world_size, "Number of micro batch lists must match data world size"
        for micro_batch_list in micro_batch_grid:
            assert len(micro_batch_list) == len(micro_batch_grid[0]), "All micro batch lists must have the same length"

        buffers = [await asyncio.to_thread(self.encode, micro_batch_list) for micro_batch_list in micro_batch_grid]
        requests = await self._wait_for_requests()
        send_start = time.perf_counter()
        for identity, request in requests.values():
            await self._reply(identity, buffers[request.data_rank])
        send_end = time.perf_counter()

        self.logger.debug(f"Sent micro batch grid for step {self.current_step} to {len(requests)} ranks")
        self.last_sent = buffers
        self.current_step += 1
        return send_end - send_start

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
        finally:
            self.logger.debug("Closed ZMQ micro batch sender")

    async def _reply(self, identity: bytes, buffer: bytes) -> None:
        try:
            await self.socket.send_multipart([identity, buffer], copy=False)
        except zmq.ZMQError as error:
            if error.errno != zmq.EHOSTUNREACH:
                raise
            # A restarted trainer replays the step from its replay log or asks for it again (``last_sent``)
            self.logger.warning("A trainer rank went away before it received its micro batches")

    async def _wait_for_requests(self) -> dict[int, tuple[bytes, BatchRequest]]:
        """Collect one request per trainer rank for the current step. Each request carries the
        trainer world size, so the first one says how many to wait for. A restarted trainer replays
        the steps it received from its own replay log; a rank that died before saving the last
        step asks for it again and gets it from ``last_sent``. A rank's newest request replaces
        the request of its previous worker."""
        requests: dict[int, tuple[bytes, BatchRequest]] = {}
        while True:
            identity, payload = await self.socket.recv_multipart()
            request = self.request_decoder.decode(payload)
            if request.step == self.current_step:
                requests[request.rank] = (identity, request)
                if len(requests) == request.world_size:
                    return requests
            elif request.step == self.current_step - 1 and self.last_sent:
                await self._reply(identity, self.last_sent[request.data_rank])
            else:
                raise RuntimeError(
                    f"A trainer rank requested step {request.step}, but the orchestrator is on step "
                    f"{self.current_step}. Resume both from the same checkpoint step."
                )


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

    def request(self) -> None:
        request = BatchRequest(
            world_size=self.world_size,
            rank=self.rank,
            data_rank=self.data_rank,
            step=self.current_step,
        )
        self.socket.send(self.request_encoder.encode(request))

    def poll(self, timeout: float) -> bool:
        return self.socket.poll(timeout=int(timeout * 1000)) != 0

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
