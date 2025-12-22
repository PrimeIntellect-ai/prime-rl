from pathlib import Path
from time import time

import zmq

from prime_rl.trainer.runs import get_runs
from prime_rl.transport.base import MicroBatchReceiver, MicroBatchSender, TrainingBatchReceiver, TrainingBatchSender
from prime_rl.transport.config import ZMQTransportConfig
from prime_rl.transport.types import MicroBatch, TrainingBatch

LOG_FREQ_SECONDS = 10


class ZMQTrainingBatchSender(TrainingBatchSender):
    """
    One PUSH socket; each message is multipart: [sender_id, payload]
    sender_id = output_dir.stem
    """

    def __init__(self, output_dir: Path, transport: ZMQTransportConfig):
        super().__init__(output_dir)

        self.context = zmq.Context.instance()
        self.socket: zmq.Socket = self.context.socket(zmq.PUSH)
        self.socket.setsockopt(zmq.SNDHWM, transport.hwm)
        self.socket.connect(f"tcp://{transport.host}:{transport.port}")

        self.sender_id = output_dir.stem.encode("utf-8")

        self.logger.info(
            f"ZMQ training batch sender initialized: output_dir={output_dir} "
            f"endpoint=tcp://{transport.host}:{transport.port} hwm={transport.hwm}"
        )

    def send(self, batch: TrainingBatch) -> None:
        payload = self.encoder.encode(batch)
        self.logger.debug(f"Sending batch {batch.step} to {self.sender_id}")
        self.socket.send_multipart([self.sender_id, payload], copy=False)

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
        finally:
            self.logger.info("ZMQ training batch sender closed")


class ZMQTrainingBatchReceiver(TrainingBatchReceiver):
    """
    One PULL socket bound to the shared endpoint.
    Receives multipart [sender_id, payload].

    Semantics vs filesystem version:
    - For each run idx, we return at most ONE batch per receive() call.
    - We always return the oldest pending batch (by step) for that idx.
    - Any newer batches received are buffered for future receive() calls.
    """

    def __init__(self, transport: ZMQTransportConfig):
        super().__init__()
        self.runs = get_runs()
        self._last_logged_time = time()
        self._last_logged_ids: list[str] | None = None

        self.context = zmq.Context.instance()
        self.socket: zmq.Socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVHWM, transport.hwm)
        self.socket.bind(f"tcp://{transport.host}:{transport.port}")

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

        # Pending batches per run_id, keyed by step -> batch.
        # We return the smallest step first; newer steps remain buffered.
        self._pending: dict[bytes, dict[int, TrainingBatch]] = {}

        self.logger.info(
            f"ZMQ training batch receiver initialized: endpoint=tcp://{transport.host}:{transport.port} hwm={transport.hwm}"
        )

    def can_receive(self) -> bool:
        events = dict(self.poller.poll(timeout=0))
        return self.socket in events

    def _drain_into_pending(self) -> None:
        """Drain all currently available ZMQ messages into the per-idx pending buffer (non-blocking)."""
        while True:
            events = dict(self.poller.poll(timeout=0))
            if self.socket not in events:
                break

            try:
                sender_id, payload = self.socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
            except zmq.Again:
                break

            try:
                batch: TrainingBatch = self.decoder.decode(payload)
                sender_id = bytes(sender_id)
            except Exception as e:
                self.logger.error(f"Error decoding rollouts for sender_id={sender_id!r}: {e}")
                continue

            self.logger.info(f"Received batch from {sender_id}")

            per_id_batches = self._pending.setdefault(sender_id, {})
            assert batch.step not in per_id_batches, (
                f"Step {batch.step} already in pending for {sender_id!r}, this should not happen: {per_id_batches}"
            )
            per_id_batches[batch.step] = batch

    def receive(self) -> list[TrainingBatch]:
        """Return at most one (oldest) pending batch per run idx; buffer newer ones for later calls."""
        batches: list[TrainingBatch] = []
        current_ids = [self.runs.idx_2_id[idx] for idx in self.runs.used_idxs]
        if current_ids != self._last_logged_ids or time() - self._last_logged_time > LOG_FREQ_SECONDS:
            if len(current_ids) == 0:
                self.logger.debug(
                    "Did you set the output dir of the orchestrator to a run_* subdirectory of the trainer output dir?"
                )
            self.logger.debug(f"Listening for batches from runs {current_ids}")
            self._last_logged_ids = current_ids
            self._last_logged_time = time()

        self._drain_into_pending()

        for idx in list(self.runs.used_idxs):
            run_id = self.runs.idx_2_id[idx].encode("utf-8")
            if self.runs.ready_to_update[idx]:
                continue

            per_id_batches = self._pending.get(run_id)
            if not per_id_batches:
                continue

            oldest_step = min(per_id_batches.keys())
            batch = per_id_batches.pop(oldest_step)
            if not per_id_batches:
                self._pending.pop(run_id, None)

            batch.run_idx = idx
            self.logger.debug(f"Received batch {batch.step} from {run_id!r}")
            batches.append(batch)

        return batches

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
        finally:
            self.context.term()
            self.logger.info("ZMQ training batch receiver closed")


class ZMQMicroBatchSender(MicroBatchSender):
    def __init__(self, output_dir: Path, data_world_size: int, current_step: int, transport: ZMQTransportConfig):
        """ZMQ micro batch sender that sends micro batches to the trainers through ZMQ transport. There is one sender for the entire data world."""
        super().__init__(output_dir, data_world_size)
        self.context = zmq.Context.instance()

        # Data channel (PUB)
        self.socket: zmq.Socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, transport.hwm)
        self.socket.bind(f"tcp://{transport.host}:{transport.port + 1}")

        # ready barrier socket, to avoid slow joiners dropping for step 0 (and generally at startup)
        self.ready_socket: zmq.Socket = self.context.socket(zmq.PULL)
        self.ready_socket.setsockopt(zmq.RCVHWM, transport.hwm)
        self.ready_socket.bind(f"tcp://{transport.host}:{transport.port + 2}")

        self._ready = False

        self.logger.info(
            f"ZMQ micro batch sender initialized: endpoint=tcp://{transport.host}:{transport.port + 1} "
            f"ready_endpoint=tcp://{transport.host}:{transport.port + 2} hwm={transport.hwm}"
        )

        self._topic_prefix = b"data_rank|"

        self._current_step = current_step

    def _wait_for_ready(self) -> None:
        if self._ready:
            return

        self.logger.debug(f"Waiting for {self.data_world_size} READY messages")
        ready_ranks: set[int] = set()

        # Block until all ranks have announced readiness
        while len(ready_ranks) < self.data_world_size:
            msg = self.ready_socket.recv()  # blocks
            try:
                rank = int(msg.decode("utf-8"))
            except Exception:
                continue
            ready_ranks.add(rank)

        self.logger.debug(f"All {self.data_world_size} ranks READY, starting PUB")
        self._ready = True

    def send(self, micro_batch_grid: list[list[MicroBatch]]) -> None:
        """Send grid of micro batches to the trainers."""
        assert len(micro_batch_grid) == self.data_world_size, "Number of micro batch lists must match data world size"
        for micro_batch_list in micro_batch_grid:
            assert len(micro_batch_list) == len(micro_batch_grid[0]), "All micro batch lists must have the same length"

        # Ensure no slow-joiner drops for step 0 (and generally at startup)
        self._wait_for_ready()

        self.logger.debug(f"Sending micro batch grid for step {self._current_step}")
        for data_rank in range(self.data_world_size):
            buffer = self.encoder.encode(micro_batch_grid[data_rank])
            topic = self._topic_prefix + str(data_rank).encode("utf-8") + b"|"
            self.socket.send_multipart([topic, buffer], copy=False)
        self._current_step += 1

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
            self.ready_socket.close(linger=0)
        finally:
            self.logger.info("ZMQ micro batch sender closed")


class ZMQMicroBatchReceiver(MicroBatchReceiver):
    def __init__(self, output_dir: Path, data_rank: int, current_step: int, transport: ZMQTransportConfig):
        """ZMQ micro batch receiver that receives micro batches from the sender. There is one receiver per data rank."""
        super().__init__(output_dir, data_rank)
        self.context = zmq.Context.instance()

        self.socket: zmq.Socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.RCVHWM, transport.hwm)
        self.socket.connect(f"tcp://{transport.host}:{transport.port + 1}")

        self._topic = b"data_rank|" + str(data_rank).encode("utf-8") + b"|"
        self.socket.setsockopt(zmq.SUBSCRIBE, self._topic)

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

        # ready barrier socket, to avoid slow joiners dropping for step 0 (and generally at startup)
        self.ready_socket: zmq.Socket = self.context.socket(zmq.PUSH)
        self.ready_socket.setsockopt(zmq.SNDHWM, transport.hwm)
        self.ready_socket.connect(f"tcp://{transport.host}:{transport.port + 2}")

        # Announce readiness after connect+subscribe are set
        self.ready_socket.send(str(data_rank).encode("utf-8"))

        self.logger.info(
            f"ZMQ micro batch receiver initialized: endpoint=tcp://{transport.host}:{transport.port + 1} "
            f"ready_endpoint=tcp://{transport.host}:{transport.port + 2} hwm={transport.hwm}"
        )

        self._current_step = current_step

    def wait(self) -> None:
        self.poller.poll(timeout=None)

    def can_receive(self) -> bool:
        events = dict(self.poller.poll(timeout=0))
        return self.socket in events

    def receive(self) -> list[MicroBatch]:
        """Receive a micro batch from the trainer."""
        _, payload = self.socket.recv_multipart(copy=False)
        micro_batches: list[MicroBatch] = self.decoder.decode(payload)
        self.logger.debug(f"Received {len(micro_batches)} micro batches for step {self._current_step}")
        self._current_step += 1
        return micro_batches

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
            self.ready_socket.close(linger=0)
        finally:
            self.logger.info("ZMQ micro batch receiver closed")
