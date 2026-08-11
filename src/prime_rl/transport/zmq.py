from pathlib import Path

import zmq
import zmq.asyncio

from prime_rl.configs.shared import ZMQTransportConfig
from prime_rl.transport.base import MicroBatchReceiver, MicroBatchSender
from prime_rl.transport.types import MicroBatch


class ZMQMicroBatchSender(MicroBatchSender):
    def __init__(self, output_dir: Path, data_world_size: int, current_step: int, transport: ZMQTransportConfig):
        """ZMQ micro batch sender that publishes per-rank micro batches to the trainers.
        There is one sender (in the orchestrator process) for the entire data world.

        PUB never blocks: a message past a subscriber's HWM would be dropped
        silently. The orchestrator's dispatch gate bounds in-flight steps to
        TARGET_LAG + 1 (one message per rank per step), far below the HWM, so
        drops cannot happen in steady state; the READY barrier below covers
        slow joiners at startup.
        """
        super().__init__(output_dir, data_world_size)
        # Async context so ``send`` yields instead of blocking the orchestrator event loop
        self.context = zmq.asyncio.Context.instance()

        # Data channel (PUB)
        self.socket: zmq.asyncio.Socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, transport.hwm)
        self.socket.bind(f"tcp://{transport.host}:{transport.port}")

        # Control channel (PULL): READY announcements at startup and per-step
        # consumption acks, both pushed by the receivers.
        self.ready_socket: zmq.asyncio.Socket = self.context.socket(zmq.PULL)
        self.ready_socket.setsockopt(zmq.RCVHWM, transport.hwm)
        self.ready_socket.bind(f"tcp://{transport.host}:{transport.port + 1}")

        self._ready = False
        self._ready_ranks: set[int] = set()
        # Consumption acks per data rank. ``current_step`` floors the map so a
        # resumed run doesn't wait for acks of pre-resume steps.
        self._acked_step: dict[int, int] = {}
        self._ack_floor = current_step

        self.logger.info(
            f"ZMQ micro batch sender initialized: endpoint=tcp://{transport.host}:{transport.port} "
            f"ready_endpoint=tcp://{transport.host}:{transport.port + 1} hwm={transport.hwm}"
        )

        self._topic_prefix = b"data_rank|"

        self._current_step = current_step

    def _handle_control(self, msg: bytes) -> None:
        """Parse one control message: a READY announcement (``<rank>``) or a
        consumption ack (``ack|<rank>|<step>``)."""
        try:
            text = msg.decode("utf-8")
            if text.startswith("ack|"):
                _, rank, step = text.split("|")
                data_rank = int(rank)
                self._acked_step[data_rank] = max(self._acked_step.get(data_rank, 0), int(step))
            else:
                self._ready_ranks.add(int(text))
        except Exception:
            return

    @property
    def min_consumed_step(self) -> int:
        """The highest training step every data rank has acked (the start step
        before any acks)."""
        return min(
            (self._acked_step.get(rank, self._ack_floor) for rank in range(self.data_world_size)),
            default=self._ack_floor,
        )

    async def wait_for_consumed(self, step: int) -> None:
        """Block until every data rank has acked training step ``step``."""
        while self.min_consumed_step < step:
            self._handle_control(await self.ready_socket.recv())

    async def _wait_for_ready(self) -> None:
        if self._ready:
            return

        self.logger.debug(f"Waiting for {self.data_world_size} READY messages")

        # Block until all ranks have announced readiness
        while len(self._ready_ranks) < self.data_world_size:
            self._handle_control(await self.ready_socket.recv())

        self.logger.debug(f"All {self.data_world_size} ranks READY, starting PUB")
        self._ready = True

    async def send(self, micro_batch_grid: list[list[MicroBatch]]) -> None:
        """Send grid of micro batches to the trainers."""
        assert len(micro_batch_grid) == self.data_world_size, "Number of micro batch lists must match data world size"
        for micro_batch_list in micro_batch_grid:
            assert len(micro_batch_list) == len(micro_batch_grid[0]), "All micro batch lists must have the same length"

        # Ensure no slow-joiner drops for step 0 (and generally at startup)
        await self._wait_for_ready()

        self.logger.debug(f"Sending micro batch grid for step {self._current_step}")
        for data_rank in range(self.data_world_size):
            buffer = self.encoder.encode(micro_batch_grid[data_rank])
            topic = self._topic_prefix + str(data_rank).encode("utf-8") + b"|"
            await self.socket.send_multipart([topic, buffer], copy=False)
        self._current_step += 1

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
            self.ready_socket.close(linger=0)
        finally:
            self.logger.info("ZMQ micro batch sender closed")


class ZMQMicroBatchReceiver(MicroBatchReceiver):
    def __init__(self, output_dir: Path, data_rank: int, current_step: int, transport: ZMQTransportConfig):
        """ZMQ micro batch receiver that receives micro batches from the orchestrator. There is one receiver per data rank."""
        super().__init__(output_dir, data_rank)
        self.context = zmq.Context.instance()

        self.socket: zmq.Socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.RCVHWM, transport.hwm)
        self.socket.connect(f"tcp://{transport.host}:{transport.port}")

        self._topic = b"data_rank|" + str(data_rank).encode("utf-8") + b"|"
        self.socket.setsockopt(zmq.SUBSCRIBE, self._topic)

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

        # Control channel (PUSH): READY announcement at startup, per-step
        # consumption acks afterwards.
        self.ready_socket: zmq.Socket = self.context.socket(zmq.PUSH)
        self.ready_socket.setsockopt(zmq.SNDHWM, transport.hwm)
        self.ready_socket.connect(f"tcp://{transport.host}:{transport.port + 1}")

        # Announce readiness after connect+subscribe are set
        self.ready_socket.send(str(data_rank).encode("utf-8"))

        self.logger.info(
            f"ZMQ micro batch receiver initialized: endpoint=tcp://{transport.host}:{transport.port} "
            f"ready_endpoint=tcp://{transport.host}:{transport.port + 1} hwm={transport.hwm}"
        )

        self._current_step = current_step

    def wait(self) -> None:
        self.poller.poll(timeout=None)

    def can_receive(self) -> bool:
        events = dict(self.poller.poll(timeout=0))
        return self.socket in events

    def receive(self) -> list[MicroBatch]:
        """Receive a micro batch from the orchestrator."""
        _, payload = self.socket.recv_multipart(copy=False)
        micro_batches: list[MicroBatch] = self.decoder.decode(payload)
        self.logger.debug(f"Received {len(micro_batches)} micro batches for step {self._current_step}")
        self._current_step += 1
        return micro_batches

    def ack(self, step: int) -> None:
        """Announce that training step ``step`` finished consuming its batch."""
        self.ready_socket.send(f"ack|{self.data_rank}|{step}".encode("utf-8"))

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
            self.ready_socket.close(linger=0)
        finally:
            self.logger.info("ZMQ micro batch receiver closed")
