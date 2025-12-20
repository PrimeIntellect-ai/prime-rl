from pathlib import Path
from time import time

import zmq

from prime_rl.trainer.runs import get_runs
from prime_rl.transport import MicroBatch, MicroBatchReceiver, MicroBatchSender, TrainingBatch
from prime_rl.transport.base import TrainingBatchReceiver, TrainingBatchSender
from prime_rl.transport.config import ZMQTransportConfig

LOG_FREQ_SECONDS = 10


class ZMQTrainingBatchSender(TrainingBatchSender):
    """
    One PUSH socket; each message is multipart: [sender_id, payload]
    sender_id = output_dir.stem
    """

    def __init__(self, output_dir: Path, transport: ZMQTransportConfig):
        super().__init__(output_dir)

        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUSH)

        self.socket.setsockopt(zmq.SNDHWM, transport.hwm)

        self.socket.connect(f"tcp://{transport.host}:{transport.port}")

        self.sender_id = output_dir.stem
        self.sender_id_bytes = self.sender_id.encode("utf-8")

        self.logger.info(
            f"ZMQ training batch sender initialized: output_dir={output_dir} "
            f"endpoint=tcp://{transport.host}:{transport.port} hwm={transport.hwm}"
        )

    def send(self, batch: TrainingBatch) -> None:
        payload = self.encoder.encode(batch)
        self.logger.warning(f"Sending batch {batch.step} to {self.sender_id}")
        self.socket.send_multipart([self.sender_id_bytes, payload])

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
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVHWM, transport.hwm)
        self.socket.bind(f"tcp://{transport.host}:{transport.port}")

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

        # Pending batches per run idx, keyed by step -> batch.
        # We return the smallest step first; newer steps remain buffered.
        self._pending: dict[int, dict[int, TrainingBatch]] = {}

        self.logger.info(
            f"ZMQ training batch receiver initialized: endpoint=tcp://{transport.host}:{transport.port} hwm={transport.hwm}"
        )

    def get_idx_from_sender_id(self, sender_id: bytes) -> int | None:
        return self.runs.id_2_idx.get(sender_id.decode("utf-8"))

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
                sender_id, payload = self.socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

            idx = self.get_idx_from_sender_id(sender_id)
            if idx is None:
                self.logger.warning(f"Dropping message for unknown sender_id={sender_id!r}")
                continue

            try:
                batch: TrainingBatch = self.decoder.decode(payload)
            except Exception as e:
                self.logger.error(f"Error decoding rollouts for sender_id={sender_id!r}: {e}")
                continue

            step = int(batch.step)
            per_idx = self._pending.setdefault(idx, {})
            assert step not in per_idx, f"Step {step} already in pending for idx, this should not happen: {per_idx}"
            per_idx[step] = batch

    def receive(self) -> list[TrainingBatch]:
        """Return at most one (oldest) pending batch per run idx; buffer newer ones for later calls."""
        batches: list[TrainingBatch] = []

        current_ids = [self.runs.get_run_dir(idx).stem for idx in self.runs.used_idxs]
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
            if self.runs.ready_to_update[idx]:
                continue

            per_idx = self._pending.get(idx)
            if not per_idx:
                continue

            oldest_step = min(per_idx.keys())
            batch = per_idx.pop(oldest_step)
            if not per_idx:
                self._pending.pop(idx, None)

            batch.run_idx = idx
            self.logger.debug(f"Received batch {batch.step} from {self.runs.idx_2_id[idx]}")
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
        """ZMQ micro batch sender that sends micro batches to the trainers through ZMQ transport."""
        super().__init__(output_dir, data_world_size)
        self.context = zmq.Context.instance()

        # Data channel (PUB)
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, transport.hwm)
        self.socket.bind(f"tcp://{transport.host}:{transport.port + 1}")

        # ready barrier socket, to avoid slow joiners dropping for step 0 (and generally at startup)
        self.ready_socket = self.context.socket(zmq.PULL)
        self.ready_socket.setsockopt(zmq.RCVHWM, transport.hwm)
        self.ready_socket.bind(f"tcp://{transport.host}:{transport.port + 2}")

        self._ready = False

        self.logger.info(
            f"ZMQ micro batch sender initialized: endpoint=tcp://{transport.host}:{transport.port + 1} "
            f"ready_endpoint=tcp://{transport.host}:{transport.port + 2} hwm={transport.hwm}"
        )

        self.current_step = current_step
        self._topic_prefix = b"data_rank|"

    def _wait_for_ready(self) -> None:
        if self._ready:
            return

        self.logger.debug(
            f"Waiting for {self.data_world_size} READY messages on port {self.ready_socket.LAST_ENDPOINT.decode(errors='ignore') if hasattr(self.ready_socket, 'LAST_ENDPOINT') else ''}"
        )
        ready_ranks: set[int] = set()

        # Block until all ranks have announced readiness (no graceful handling requested)
        while len(ready_ranks) < self.data_world_size:
            msg = self.ready_socket.recv()  # blocks
            try:
                rank = int(msg.decode("utf-8"))
            except Exception:
                continue
            ready_ranks.add(rank)

        self.logger.warning(f"All {self.data_world_size} ranks READY, starting PUB")
        self._ready = True

    def send(self, micro_batch_grid: list[list[MicroBatch]]) -> None:
        """Send grid of micro batches to the trainers."""
        assert len(micro_batch_grid) == self.data_world_size, "Number of micro batch lists must match data world size"
        for micro_batch_list in micro_batch_grid:
            assert len(micro_batch_list) == len(micro_batch_grid[0]), "All micro batch lists must have the same length"

        # Ensure no slow-joiner drops for step 0 (and generally at startup)
        self._wait_for_ready()

        self.logger.warning(f"Sending micro batch grid for step {self.current_step}")

        for data_rank in range(self.data_world_size):
            buffer = self.encoder.encode(micro_batch_grid[data_rank])
            topic = self._topic_prefix + str(data_rank).encode("utf-8") + b"|"
            self.socket.send_multipart([topic, buffer])

        self.current_step += 1

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
            self.ready_socket.close(linger=0)
        finally:
            self.logger.info("ZMQ micro batch sender closed")


class ZMQMicroBatchReceiver(MicroBatchReceiver):
    def __init__(self, output_dir: Path, data_rank: int, current_step: int, transport: ZMQTransportConfig):
        """ZMQ micro batch receiver that receives micro batches from the through ZMQ transport."""
        super().__init__(output_dir, data_rank)
        self.context = zmq.Context.instance()

        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.RCVHWM, transport.hwm)
        self.socket.connect(f"tcp://{transport.host}:{transport.port + 1}")

        self._topic = b"data_rank|" + str(data_rank).encode("utf-8") + b"|"
        self.socket.setsockopt(zmq.SUBSCRIBE, self._topic)

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

        # ready barrier socket, to avoid slow joiners dropping for step 0 (and generally at startup)
        self.ready_socket = self.context.socket(zmq.PUSH)
        self.ready_socket.setsockopt(zmq.SNDHWM, transport.hwm)
        self.ready_socket.connect(f"tcp://{transport.host}:{transport.port + 2}")

        # Announce readiness after connect+subscribe are set
        self.ready_socket.send(str(data_rank).encode("utf-8"))

        self.logger.info(
            f"ZMQ micro batch receiver initialized: endpoint=tcp://{transport.host}:{transport.port + 1} "
            f"ready_endpoint=tcp://{transport.host}:{transport.port + 2} hwm={transport.hwm}"
        )

        self.current_step = current_step

    def wait(self) -> None:
        self.logger.debug(f"Waiting for micro batch for step {self.current_step}")
        self.poller.poll(timeout=None)

    def can_receive(self) -> bool:
        events = dict(self.poller.poll(timeout=0))
        return self.socket in events

    def receive(self) -> list[MicroBatch]:
        """Receive a micro batch from the trainer."""
        _, payload = self.socket.recv_multipart()
        micro_batches: list[MicroBatch] = self.decoder.decode(payload)
        self.current_step += 1
        return micro_batches

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
            self.ready_socket.close(linger=0)
        finally:
            self.logger.info("ZMQ micro batch receiver closed")
