from pathlib import Path
from time import time
from typing import Optional, Tuple

import zmq

from prime_rl.trainer.runs import get_runs
from prime_rl.transport import TrainingBatch
from prime_rl.transport.base import TrainingBatchReceiver, TrainingBatchSender
from prime_rl.transport.config import ZMQTransportConfig

LOG_FREQ_SECONDS = 10


class ZMQTrainingBatchSender(TrainingBatchSender):
    """
    One PUSH socket; each message is multipart: [sender_id, payload]
    sender_id = run_dir.stem
    """

    def __init__(self, output_dir: Path, transport: ZMQTransportConfig):
        super().__init__(output_dir)

        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUSH)

        self.socket.setsockopt(zmq.SNDHWM, transport.hwm)

        # Single shared endpoint; receiver should bind, senders connect.
        self.socket.connect(f"tcp://{transport.host}:{transport.port}")

        self.sender_id = output_dir.stem
        self.sender_id_bytes = self.sender_id.encode("utf-8")

        self.logger.info(
            f"ZMQ training batch sender initialized: id={self.sender_id} "
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

            # Buffer by step; if duplicates arrive, keep the first (or overwrite—either is fine).
            step = int(batch.step)
            per_idx = self._pending.setdefault(idx, {})
            assert step not in per_idx, f"Step {step} already in pending for idx {idx}"
            if step not in per_idx:
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
            self.logger.warning(f"Received batch {batch.step} from {idx}")
            batches.append(batch)

        return batches

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
        finally:
            self.context.term()
            self.logger.info("ZMQ training batch receiver closed")
