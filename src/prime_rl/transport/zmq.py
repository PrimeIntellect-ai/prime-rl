import os
from pathlib import Path
from time import monotonic, sleep, time

import zmq

from prime_rl.configs.shared import ZMQTransportConfig
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.transport.base import MicroBatchReceiver, MicroBatchSender, TrainingBatchReceiver, TrainingBatchSender
from prime_rl.transport.types import MicroBatch, TrainingBatch

LOG_FREQ_SECONDS = 10


def _connect_host(transport: ZMQTransportConfig) -> str:
    if transport.host and transport.host != "0.0.0.0":
        return transport.host
    return os.environ.get("MASTER_ADDR", "localhost")


def _bind_host() -> str:
    return "0.0.0.0"


def _timeout_ms(seconds: int) -> int:
    return int(seconds * 1000)


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
        connect_host = _connect_host(transport)
        self.socket.connect(f"tcp://{connect_host}:{transport.port}")

        self.sender_id = output_dir.stem.encode("utf-8")

        self.logger.info(
            f"ZMQ training batch sender initialized: output_dir={output_dir} "
            f"endpoint=tcp://{connect_host}:{transport.port} hwm={transport.hwm}"
        )

    async def send(self, batch: TrainingBatch) -> None:
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
        self.multi_run_manager = get_multi_run_manager()
        self._last_logged_time = time()
        self._last_logged_ids: list[str] | None = None
        self._waiting_since: float | None = None

        self.context = zmq.Context.instance()
        self.socket: zmq.Socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVHWM, transport.hwm)
        bind_host = _bind_host()
        self.socket.bind(f"tcp://{bind_host}:{transport.port}")

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

        # Pending batches per run_id, keyed by step -> batch.
        # We return the smallest step first; newer steps remain buffered.
        self._pending: dict[bytes, dict[int, TrainingBatch]] = {}

        self.logger.info(
            f"ZMQ training batch receiver initialized: endpoint=tcp://{bind_host}:{transport.port} hwm={transport.hwm}"
        )

    def can_receive(self) -> bool:
        events = dict(self.poller.poll(timeout=0))
        return self.socket in events

    def _drain_into_pending(self) -> None:
        """Drain all currently available ZMQ messages into the per-idx pending buffer (non-blocking)."""
        while True:
            if not self.can_receive():
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

            per_id_batches = self._pending.setdefault(sender_id, {})
            assert batch.step not in per_id_batches, (
                f"Step {batch.step} already in pending for {sender_id!r}, this should not happen: {per_id_batches.keys()}"
            )
            per_id_batches[batch.step] = batch

    def receive(self) -> list[TrainingBatch]:
        """Return at most one (oldest) pending batch per run idx; buffer newer ones for later calls."""
        batches: list[TrainingBatch] = []
        now = time()

        # Bring any available messages into the pending buffer (non-blocking)
        self._drain_into_pending()

        # Track how long we've been waiting for any runnable batch.
        runnable_available = False
        for idx in self.multi_run_manager.used_idxs:
            if self.multi_run_manager.ready_to_update[idx]:
                continue
            run_id = self.multi_run_manager.idx_2_id[idx].encode("utf-8")
            if self._pending.get(run_id):
                runnable_available = True
                break

        if runnable_available:
            self._waiting_since = None
        else:
            self._waiting_since = self._waiting_since or now

        current_ids = [self.multi_run_manager.idx_2_id[idx] for idx in self.multi_run_manager.used_idxs]
        if current_ids != self._last_logged_ids or now - self._last_logged_time > LOG_FREQ_SECONDS:
            if len(current_ids) == 0:
                self.logger.debug(
                    "Did you set the output dir of the orchestrator to a run_* subdirectory of the trainer output dir?"
                )
            waiting_suffix = ""
            if self._waiting_since is not None:
                waiting_suffix = f" (waiting {now - self._waiting_since:.1f}s)"
            self.logger.debug(f"Listening for batches from runs {current_ids}{waiting_suffix}")
            self._last_logged_ids = current_ids
            self._last_logged_time = now

        for idx in list(self.multi_run_manager.used_idxs):
            run_id = self.multi_run_manager.idx_2_id[idx].encode("utf-8")
            if self.multi_run_manager.ready_to_update[idx]:
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
        self._ready_timeout_ms = _timeout_ms(transport.ready_timeout_seconds)
        self._publish_grace_seconds = transport.publish_grace_ms / 1000.0

        # Data channel (PUB)
        self.socket: zmq.Socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, transport.hwm)
        bind_host = _bind_host()
        self.socket.bind(f"tcp://{bind_host}:{transport.port + 1}")

        # ready barrier socket, to avoid slow joiners dropping for step 0 (and generally at startup)
        self.ready_socket: zmq.Socket = self.context.socket(zmq.PULL)
        self.ready_socket.setsockopt(zmq.RCVHWM, transport.hwm)
        self.ready_socket.bind(f"tcp://{bind_host}:{transport.port + 2}")
        self.ready_poller = zmq.Poller()
        self.ready_poller.register(self.ready_socket, zmq.POLLIN)

        self._ready = False

        self.logger.info(
            f"ZMQ micro batch sender initialized: endpoint=tcp://{bind_host}:{transport.port + 1} "
            f"ready_endpoint=tcp://{bind_host}:{transport.port + 2} hwm={transport.hwm}"
        )

        self._topic_prefix = b"data_rank|"

        self._current_step = current_step

    def _wait_for_ready(self) -> None:
        if self._ready:
            return

        self.logger.debug(f"Waiting for {self.data_world_size} READY messages")
        ready_ranks: set[int] = set()

        deadline = monotonic() + self._ready_timeout_ms / 1000.0
        while len(ready_ranks) < self.data_world_size:
            remaining_ms = max(0, int((deadline - monotonic()) * 1000))
            if remaining_ms == 0:
                missing = sorted(set(range(self.data_world_size)) - ready_ranks)
                raise TimeoutError(
                    f"Timed out waiting for ZMQ micro-batch READY messages from ranks {missing} "
                    f"after {self._ready_timeout_ms / 1000.0:.0f}s"
                )
            events = dict(self.ready_poller.poll(timeout=remaining_ms))
            if self.ready_socket not in events:
                continue
            msg = self.ready_socket.recv(flags=zmq.NOBLOCK)
            try:
                rank = int(msg.decode("utf-8"))
            except Exception:
                continue
            ready_ranks.add(rank)

        self.logger.debug(f"All {self.data_world_size} ranks READY, starting PUB")
        if self._publish_grace_seconds > 0:
            sleep(self._publish_grace_seconds)
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
            step = str(self._current_step).encode("utf-8")
            self.socket.send_multipart([topic, step, buffer], copy=False)
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
        self._recv_timeout_ms = _timeout_ms(transport.recv_timeout_seconds)

        self.socket: zmq.Socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.RCVHWM, transport.hwm)
        self.socket.setsockopt(zmq.RCVTIMEO, self._recv_timeout_ms)
        connect_host = _connect_host(transport)
        self.socket.connect(f"tcp://{connect_host}:{transport.port + 1}")

        self._topic = b"data_rank|" + str(data_rank).encode("utf-8") + b"|"
        self.socket.setsockopt(zmq.SUBSCRIBE, self._topic)

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

        # ready barrier socket, to avoid slow joiners dropping for step 0 (and generally at startup)
        self.ready_socket: zmq.Socket = self.context.socket(zmq.PUSH)
        self.ready_socket.setsockopt(zmq.SNDHWM, transport.hwm)
        self.ready_socket.connect(f"tcp://{connect_host}:{transport.port + 2}")

        # Announce readiness after connect+subscribe are set
        self.ready_socket.send(str(data_rank).encode("utf-8"))

        self.logger.info(
            f"ZMQ micro batch receiver initialized: endpoint=tcp://{connect_host}:{transport.port + 1} "
            f"ready_endpoint=tcp://{connect_host}:{transport.port + 2} hwm={transport.hwm} "
            f"recv_timeout_seconds={transport.recv_timeout_seconds}"
        )

        self._current_step = current_step

    def wait(self) -> None:
        events = dict(self.poller.poll(timeout=self._recv_timeout_ms))
        if self.socket not in events:
            raise TimeoutError(
                f"Timed out waiting for ZMQ micro-batch for data_rank={self.data_rank} "
                f"step={self._current_step} after {self._recv_timeout_ms / 1000.0:.0f}s"
            )

    def can_receive(self) -> bool:
        events = dict(self.poller.poll(timeout=0))
        return self.socket in events

    def receive(self) -> list[MicroBatch]:
        """Receive a micro batch from the trainer."""
        try:
            _, step_raw, payload = self.socket.recv_multipart(copy=False)
        except zmq.Again as exc:
            raise TimeoutError(
                f"Timed out receiving ZMQ micro-batch payload for data_rank={self.data_rank} "
                f"step={self._current_step} after {self._recv_timeout_ms / 1000.0:.0f}s"
            ) from exc
        step = int(bytes(step_raw).decode("utf-8"))
        if step != self._current_step:
            raise ValueError(
                f"Received ZMQ micro-batch for step {step}, expected {self._current_step} (data_rank={self.data_rank})"
            )
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
