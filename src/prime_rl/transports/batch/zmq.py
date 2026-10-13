import asyncio
from pathlib import Path

import msgspec
import zmq
import zmq.asyncio

from prime_rl.configs.shared import ZMQTransportConfig
from prime_rl.trainer.world import get_world
from prime_rl.transports.batch.base import BatchReceiver, BatchSender
from prime_rl.transports.batch.types import MicroBatch


class BatchRequest(msgspec.Struct):
    """A trainer rank asking for its micro batches of one step."""

    rank: int
    world_size: int
    data_rank: int
    step: int


class ZMQBatchSender(BatchSender):
    def __init__(self, output_dir: Path, data_world_size: int, current_step: int, transport: ZMQTransportConfig):
        """ZMQ micro batch sender. One ROUTER socket in the orchestrator answers
        the DEALER sockets of all trainer ranks: a rank's micro batches for a
        step leave the orchestrator only once that rank has asked for the step,
        so a trainer never holds more than the batch it is training on.

        ``send`` parks the encoded step in ``_pending`` and returns; a background
        task serves requests from it. There is one such slot, so ``send`` blocks
        until every rank has picked up the previous step.
        """
        super().__init__(output_dir, data_world_size)
        self.context = zmq.asyncio.Context.instance()
        self.socket: zmq.asyncio.Socket = self.context.socket(zmq.ROUTER)
        # Replying to a rank that went away raises instead of dropping silently
        self.socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.socket.bind(f"tcp://{transport.host}:{transport.port}")
        self.logger.debug(f"Initialized ZMQ micro batch sender (endpoint=tcp://{transport.host}:{transport.port})")

        self._request_decoder = msgspec.msgpack.Decoder(type=BatchRequest)
        self._current_step = current_step
        self._world_size: int | None = None
        self._pending: tuple[int, list[bytes]] | None = None
        self._pending_served: set[int] = set()
        self._requests: dict[int, dict[bytes, BatchRequest]] = {}
        self._slot_free = asyncio.Event()
        self._slot_free.set()
        self._serve_task = asyncio.create_task(self._serve(), name="zmq_batch_sender")

    async def send(self, micro_batch_grid: list[list[MicroBatch]]) -> None:
        assert len(micro_batch_grid) == self.data_world_size, "Number of micro batch lists must match data world size"
        for micro_batch_list in micro_batch_grid:
            assert len(micro_batch_list) == len(micro_batch_grid[0]), "All micro batch lists must have the same length"

        buffers = [self.encoder.encode(micro_batch_list) for micro_batch_list in micro_batch_grid]
        await self._wait_for_slot()
        self.logger.debug(f"Queued micro batch grid for step {self._current_step}")
        self._slot_free.clear()
        self._pending = (self._current_step, buffers)
        self._pending_served = set()
        self._current_step += 1
        await self._flush()

    async def _wait_for_slot(self) -> None:
        """Block until the previous step was picked up by every rank. A dead
        serve task could never free the slot, so its failure is raised here."""
        waiter = asyncio.ensure_future(self._slot_free.wait())
        done, _ = await asyncio.wait({waiter, self._serve_task}, return_when=asyncio.FIRST_COMPLETED)
        if self._serve_task in done:
            waiter.cancel()
            self._serve_task.result()
            raise RuntimeError("ZMQ batch sender serve task exited unexpectedly")

    async def _serve(self) -> None:
        while True:
            identity, payload = await self.socket.recv_multipart()
            request = self._request_decoder.decode(payload)
            if self._world_size is None:
                self._world_size = request.world_size
            elif request.world_size != self._world_size:
                raise ValueError(f"Trainer world size changed from {self._world_size} to {request.world_size}")
            if request.step < self._current_step - (1 if self._pending is not None else 0):
                raise ValueError(f"Rank {request.rank} requested step {request.step}, already sent")
            self._requests.setdefault(request.step, {})[identity] = request
            await self._flush()

    async def _flush(self) -> None:
        """Answer every outstanding request for the pending step."""
        if self._pending is None:
            return
        step, buffers = self._pending
        requests = self._requests.get(step, {})
        for identity, request in list(requests.items()):
            await self.socket.send_multipart([identity, buffers[request.data_rank]], copy=False)
            del requests[identity]
            self._pending_served.add(request.rank)
        if self._world_size is not None and len(self._pending_served) == self._world_size:
            self.logger.debug(f"Sent micro batch grid for step {step} to all {self._world_size} ranks")
            self._requests.pop(step, None)
            self._pending = None
            self._slot_free.set()

    def close(self) -> None:
        try:
            self._serve_task.cancel()
            self.socket.close(linger=0)
        finally:
            self.logger.debug("Closed ZMQ micro batch sender")


class ZMQBatchReceiver(BatchReceiver):
    def __init__(self, output_dir: Path, data_rank: int, current_step: int, transport: ZMQTransportConfig):
        """ZMQ micro batch receiver. Every trainer rank owns one DEALER socket
        and asks the orchestrator for each step before reading it."""
        super().__init__(output_dir, data_rank)
        self.world = get_world()
        self.context = zmq.Context.instance()
        self.socket: zmq.Socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.IDENTITY, f"rank_{self.world.rank}".encode("utf-8"))
        self.socket.connect(f"tcp://{transport.host}:{transport.port}")
        self.logger.debug(f"Initialized ZMQ micro batch receiver (endpoint=tcp://{transport.host}:{transport.port})")

        self._request_encoder = msgspec.msgpack.Encoder()
        self._current_step = current_step

    def wait(self) -> None:
        request = BatchRequest(
            rank=self.world.rank, world_size=self.world.world_size, data_rank=self.data_rank, step=self._current_step
        )
        self.socket.send(self._request_encoder.encode(request))
        self.logger.debug(f"Requested micro batches for step {self._current_step}")
        self.socket.poll(timeout=None)

    def receive(self) -> list[MicroBatch]:
        payload = self.socket.recv(copy=False)
        micro_batches: list[MicroBatch] = self.decoder.decode(payload)
        self.logger.debug(f"Received {len(micro_batches)} micro batches for step {self._current_step}")
        self._current_step += 1
        return micro_batches

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
        finally:
            self.logger.debug("Closed ZMQ micro batch receiver")
