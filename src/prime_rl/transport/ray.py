import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from prime_rl.configs.shared import RayTransportConfig
from prime_rl.transport.base import MicroBatchReceiver, MicroBatchSender, TrainingBatchReceiver, TrainingBatchSender
from prime_rl.transport.types import MicroBatch, TrainingBatch

LOG_FREQ_SECONDS = 10
POLL_INTERVAL_SECONDS = 0.05
ACTOR_SHUTDOWN_POLL_SECONDS = 0.1
ACTOR_SHUTDOWN_TIMEOUT_SECONDS = 5.0


class _RayTransportStore:
    def __init__(self, max_queued_items: int):
        self.max_queued_items = max_queued_items
        self.training_batches: list[tuple[str, bytes]] = []
        self.micro_batches: dict[tuple[int, int], bytes] = {}

    def put_training_batch(self, sender_id: str, payload: bytes) -> None:
        per_sender = sum(1 for sid, _ in self.training_batches if sid == sender_id)
        if per_sender >= self.max_queued_items:
            raise RuntimeError(
                f"Ray training batch queue is full for sender {sender_id!r} ({self.max_queued_items} items)"
            )
        self.training_batches.append((sender_id, payload))

    def drain_training_batches(self) -> list[tuple[str, bytes]]:
        batches = self.training_batches
        self.training_batches = []
        return batches

    def training_batch_count(self) -> int:
        return len(self.training_batches)

    def put_micro_batch(self, data_rank: int, step: int, payload: bytes) -> None:
        per_rank = sum(1 for (rank, _) in self.micro_batches if rank == data_rank)
        if per_rank >= self.max_queued_items:
            raise RuntimeError(
                f"Ray micro-batch queue is full for data_rank={data_rank} ({self.max_queued_items} items)"
            )
        key = (data_rank, step)
        if key in self.micro_batches:
            raise RuntimeError(f"Ray micro-batch already queued for data_rank={data_rank}, step={step}")
        self.micro_batches[key] = payload

    def has_micro_batch(self, data_rank: int, step: int) -> bool:
        return (data_rank, step) in self.micro_batches

    def pop_micro_batch(self, data_rank: int, step: int) -> bytes | None:
        return self.micro_batches.pop((data_rank, step), None)


def _require_ray():
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    try:
        import ray
    except ImportError as exc:
        raise ImportError(
            "Ray transport requires the optional 'ray' package. "
            "Install Ray before setting rollout_transport.type = 'ray'."
        ) from exc
    return ray


def _init_ray(config: RayTransportConfig):
    ray = _require_ray()
    if not ray.is_initialized():
        kwargs: dict[str, Any] = {"namespace": config.namespace}
        if config.address is not None:
            kwargs["address"] = config.address
        ray.init(**kwargs)
    return ray


def _get_transport_actor(config: RayTransportConfig):
    ray = _init_ray(config)
    try:
        return ray.get_actor(config.actor_name, namespace=config.namespace)
    except ValueError as exc:
        raise RuntimeError(
            f"Ray transport actor {config.actor_name!r} was not found in namespace {config.namespace!r}. "
            "Start the Ray-native rl launcher so it can create the shared transport actor before roles attach."
        ) from exc


def create_ray_transport_actor(config: RayTransportConfig):
    ray = _init_ray(config)
    try:
        stale_actor = ray.get_actor(config.actor_name, namespace=config.namespace)
    except ValueError:
        pass
    else:
        if not config.reclaim_stale_actor:
            raise RuntimeError(
                f"Ray transport actor {config.actor_name!r} already exists in namespace "
                f"{config.namespace!r}. Refusing to kill it because reclaim_stale_actor=false. "
                "On shared RayClusters set rollout_transport.actor_name (and matching trainer/orchestrator "
                "values) to a unique name per run, e.g. include the run id. To force the launcher to "
                "reclaim an orphaned actor, set rollout_transport.reclaim_stale_actor = true."
            )
        ray.kill(stale_actor, no_restart=True)
        deadline = time.time() + ACTOR_SHUTDOWN_TIMEOUT_SECONDS
        while time.time() < deadline:
            try:
                ray.get_actor(config.actor_name, namespace=config.namespace)
            except ValueError:
                break
            time.sleep(ACTOR_SHUTDOWN_POLL_SECONDS)
        else:
            raise RuntimeError(
                f"Timed out waiting for stale Ray transport actor {config.actor_name!r} "
                f"in namespace {config.namespace!r} to exit"
            )

    actor_cls = ray.remote(_RayTransportStore)
    return actor_cls.options(name=config.actor_name, namespace=config.namespace).remote(config.max_queued_items)


class RayTrainingBatchSender(TrainingBatchSender):
    """Ray actor based training batch sender."""

    def __init__(self, output_dir: Path, transport: RayTransportConfig):
        super().__init__(output_dir)
        self.ray = _init_ray(transport)
        self.actor = _get_transport_actor(transport)
        self.sender_id = output_dir.stem
        self.logger.info(
            f"Ray training batch sender initialized: actor={transport.actor_name} namespace={transport.namespace}"
        )

    def send(self, batch: TrainingBatch) -> None:
        payload = self.encoder.encode(batch)
        self.logger.debug(f"Sending batch {batch.step} to {self.sender_id}")
        self.ray.get(self.actor.put_training_batch.remote(self.sender_id, payload))


class RayTrainingBatchReceiver(TrainingBatchReceiver):
    """Ray actor based training batch receiver."""

    def __init__(self, transport: RayTransportConfig):
        super().__init__()
        from prime_rl.trainer.runs import get_multi_run_manager

        self.ray = _init_ray(transport)
        self.actor = _get_transport_actor(transport)
        self.multi_run_manager = get_multi_run_manager()
        self._pending: dict[str, dict[int, TrainingBatch]] = defaultdict(dict)
        self._last_logged_time = time.time()
        self._last_logged_ids: list[str] | None = None
        self._waiting_since: float | None = None
        self.logger.info(
            f"Ray training batch receiver initialized: actor={transport.actor_name} namespace={transport.namespace}"
        )

    def can_receive(self) -> bool:
        if self._has_runnable_pending_batch():
            return True
        return self.ray.get(self.actor.training_batch_count.remote()) > 0

    def _has_runnable_pending_batch(self) -> bool:
        for idx in self.multi_run_manager.used_idxs:
            if self.multi_run_manager.ready_to_update[idx]:
                continue
            run_id = self.multi_run_manager.idx_2_id[idx]
            if self._pending.get(run_id):
                return True
        return False

    def _drain_into_pending(self) -> None:
        for sender_id, payload in self.ray.get(self.actor.drain_training_batches.remote()):
            batch: TrainingBatch = self.decoder.decode(payload)
            per_id_batches = self._pending[sender_id]
            assert batch.step not in per_id_batches, (
                f"Step {batch.step} already in pending for {sender_id!r}, this should not happen: "
                f"{per_id_batches.keys()}"
            )
            per_id_batches[batch.step] = batch

    def receive(self) -> list[TrainingBatch]:
        batches: list[TrainingBatch] = []
        now = time.time()

        self._drain_into_pending()

        if self._has_runnable_pending_batch():
            self._waiting_since = None
        else:
            self._waiting_since = self._waiting_since or now

        current_ids = [self.multi_run_manager.idx_2_id[idx] for idx in self.multi_run_manager.used_idxs]
        if current_ids != self._last_logged_ids or now - self._last_logged_time > LOG_FREQ_SECONDS:
            waiting_suffix = ""
            if self._waiting_since is not None:
                waiting_suffix = f" (waiting {now - self._waiting_since:.1f}s)"
            self.logger.debug(f"Listening for Ray batches from runs {current_ids}{waiting_suffix}")
            self._last_logged_ids = current_ids
            self._last_logged_time = now

        for idx in list(self.multi_run_manager.used_idxs):
            if self.multi_run_manager.ready_to_update[idx]:
                continue

            run_id = self.multi_run_manager.idx_2_id[idx]
            per_id_batches = self._pending.get(run_id)
            if not per_id_batches:
                continue

            oldest_step = min(per_id_batches.keys())
            batch = per_id_batches.pop(oldest_step)
            if not per_id_batches:
                self._pending.pop(run_id, None)

            batch.run_idx = idx
            self.logger.debug(f"Received Ray batch {batch.step} from {run_id!r}")
            batches.append(batch)

        return batches


class RayMicroBatchSender(MicroBatchSender):
    """Ray actor based micro-batch sender."""

    def __init__(self, output_dir: Path, data_world_size: int, current_step: int, transport: RayTransportConfig):
        super().__init__(output_dir, data_world_size)
        self.ray = _init_ray(transport)
        self.actor = _get_transport_actor(transport)
        self.current_step = current_step
        self.logger.info(
            f"Ray micro-batch sender initialized: actor={transport.actor_name} namespace={transport.namespace}"
        )

    def send(self, micro_batch_grid: list[list[MicroBatch]]) -> None:
        assert len(micro_batch_grid) == self.data_world_size, "Number of micro batch lists must match data world size"
        for micro_batch_list in micro_batch_grid:
            assert len(micro_batch_list) == len(micro_batch_grid[0]), "All micro batch lists must have the same length"

        refs = []
        for data_rank in range(self.data_world_size):
            payload = self.encoder.encode(micro_batch_grid[data_rank])
            refs.append(self.actor.put_micro_batch.remote(data_rank, self.current_step, payload))
        self.ray.get(refs)
        self.logger.debug(f"Sent Ray micro-batch grid for step {self.current_step}")
        self.current_step += 1


class RayMicroBatchReceiver(MicroBatchReceiver):
    """Ray actor based micro-batch receiver."""

    def __init__(self, output_dir: Path, data_rank: int, current_step: int, transport: RayTransportConfig):
        super().__init__(output_dir, data_rank)
        self.ray = _init_ray(transport)
        self.actor = _get_transport_actor(transport)
        self.current_step = current_step
        self.logger.info(
            f"Ray micro-batch receiver initialized: actor={transport.actor_name} namespace={transport.namespace}"
        )

    def wait(self) -> None:
        while not self.can_receive():
            time.sleep(POLL_INTERVAL_SECONDS)

    def can_receive(self) -> bool:
        return self.ray.get(self.actor.has_micro_batch.remote(self.data_rank, self.current_step))

    def receive(self) -> list[MicroBatch]:
        payload = self.ray.get(self.actor.pop_micro_batch.remote(self.data_rank, self.current_step))
        if payload is None:
            raise RuntimeError(f"No Ray micro-batch available for data_rank={self.data_rank}, step={self.current_step}")
        micro_batches: list[MicroBatch] = self.decoder.decode(payload)
        self.logger.debug(f"Received {len(micro_batches)} Ray micro-batches for step {self.current_step}")
        self.current_step += 1
        return micro_batches
