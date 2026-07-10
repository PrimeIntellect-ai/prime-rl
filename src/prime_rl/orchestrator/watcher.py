"""WeightWatcher: polls broadcasts and applies policy updates behind the
dispatcher admission/drain barrier. Standalone async task; the orchestrator's
lead gate separately bounds sampling ahead of the trainer."""

from __future__ import annotations

import asyncio
import time

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.types import Policy, VersionObserver
from prime_rl.utils.async_utils import gather_shielded, safe_cancel
from prime_rl.utils.client import InferencePool
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import get_broadcast_dir, get_step_path, wait_for_path
from prime_rl.utils.utils import get_latest_ckpt_step


class WeightWatcher:
    """``await watcher.start()`` to drive the polling loop until ``stop()``."""

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        policy: Policy,
        inference: InferencePool,
        observers: list[VersionObserver],
        lora_name: str | None,
        ckpt_step: int = 0,
        poll_interval: float = 1.0,
    ) -> None:
        self.config = config
        self.policy = policy
        self.inference = inference
        self.observers = observers
        self.lora_name = lora_name
        self.ckpt_step = ckpt_step
        self.poll_interval = poll_interval

        self.last_update_weights_time: float = 0.0
        self.last_wait_for_ckpt_time: float = 0.0
        self.update_count: int = 0

        self.task: asyncio.Task | None = None
        self.update_lock = asyncio.Lock()
        self.stopped = asyncio.Event()

    async def start(self) -> None:
        self.task = asyncio.current_task()
        try:
            while not self.stopped.is_set():
                next_step = self.compute_next_ckpt_step()
                if next_step > self.ckpt_step:
                    await self.apply_policy_update(next_step)
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        self.stopped.set()
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None

    def compute_next_ckpt_step(self) -> int:
        """Next checkpoint to adopt — at least ``policy.version`` (we stay
        one step ahead of the trainer) plus anything fresher already
        published in ``broadcasts/``."""
        broadcast_dir = get_broadcast_dir(self.config.output_dir)
        latest_ckpt_step = get_latest_ckpt_step(broadcast_dir) or 0
        return max(self.policy.version, latest_ckpt_step)

    async def apply_policy_update(self, next_step: int) -> None:
        async with self.update_lock:
            if next_step <= self.ckpt_step:
                # Another caller raced us — bail without re-applying
                return

            broadcast_dir = get_broadcast_dir(self.config.output_dir)
            weights_path = get_step_path(broadcast_dir, next_step)
            stable_marker = weights_path / "STABLE"
            if not stable_marker.exists():
                get_logger().info(
                    f"Orchestrator paused: waiting for trainer to broadcast checkpoint {next_step}. "
                    "Training is progressing normally."
                )
                t0 = time.perf_counter()
                await wait_for_path(stable_marker)
                self.last_wait_for_ckpt_time = time.perf_counter() - t0
                get_logger().info(
                    f"Orchestrator resumed: checkpoint {next_step} ready (after {format_time(self.last_wait_for_ckpt_time)})"
                )

            # Establish the backend-specific application transition BEFORE
            # pausing inference engines. Dynamo fences new dispatch and drains
            # every mutable-policy request; the vLLM admin path retains its
            # configured off-policy cancellation window. Aborting a rollout
            # triggers vLLM's KV-connector cleanup (NIXL's
            # ``_reqs_not_processed``), which is only propagated to the workers
            # while the engine is stepping. If we drain after resume instead,
            # the aborts race with the flush of KV transfers that completed
            # during the pause and trip ``assert req_id in self.requests`` in
            # the decode scheduler's ``_update_from_kv_xfer_finished`` — killing
            # the engine and cascading to every DP rank. Draining first lets the
            # aborts settle under normal stepping. ``on_new_version`` (below)
            # still runs post-update for observers that need the live version.
            entered_observers: list[VersionObserver] = []
            try:
                for observer in self.observers:
                    # Include the observer before entry so a partially-applied
                    # barrier gets its failure rollback hook.
                    entered_observers.append(observer)
                    await observer.on_version_pending(next_step)
            except BaseException as exc:
                await self._notify_update_failed(entered_observers, next_step, exc)
                raise

            get_logger().debug(f"Updating weights to step {next_step}")
            t1 = time.perf_counter()
            try:
                await self.inference.update_weights(weights_path, lora_name=self.lora_name, step=next_step)
            except BaseException as exc:
                # Once an admin update starts, an error cannot prove that no
                # worker committed new weights (a resume failure is even later).
                # Keep every transition fence closed and let the component
                # failure terminate the run; reopening would stamp old-version
                # requests onto mixed or fully-updated workers.
                exc.add_note(
                    f"Policy update {next_step} may have mutated inference workers; mutable-policy admission "
                    "remains fail-closed"
                )
                raise
            self.last_update_weights_time = time.perf_counter() - t1
            self.update_count += 1
            get_logger().debug(f"Updated weights to step {next_step} in {format_time(self.last_update_weights_time)}")

            self.ckpt_step = next_step
            self.policy.version = next_step
            if self.lora_name is not None:
                self.inference.update_model_name(self.lora_name)
                self.policy.model_name = self.lora_name

            await self._notify_update_succeeded(entered_observers, next_step)

    @staticmethod
    async def _notify_update_succeeded(observers: list[VersionObserver], step: int) -> None:
        """Notify every observer, preserving cancellation until fences reopen."""
        fatal_error: BaseException | None = None
        # Complete observers in reverse order: the orchestrator first
        # re-evaluates its long-lived lead gate while the dispatcher's short
        # transition fence remains closed, then the dispatcher reopens
        # admission on the new version.
        for observer in reversed(observers):
            try:
                await observer.on_new_version(step)
            except BaseException as exc:
                # A success callback may have partially applied transition
                # state before failing. Its failure hook is also the idempotent
                # release fallback; continue through the remaining observers.
                try:
                    await observer.on_version_update_failed(step, exc)
                except BaseException as cleanup_error:
                    exc.add_note(
                        f"Observer {type(observer).__name__}.on_version_update_failed({step}) also failed: "
                        f"{cleanup_error!r}"
                    )
                    if not isinstance(cleanup_error, Exception) and fatal_error is None:
                        fatal_error = cleanup_error
                if isinstance(exc, Exception):
                    get_logger().warning(f"Observer {type(observer).__name__}.on_new_version({step}) raised: {exc!r}")
                elif fatal_error is None:
                    fatal_error = exc
                else:
                    fatal_error.add_note(
                        f"Observer {type(observer).__name__}.on_new_version({step}) also failed: {exc!r}"
                    )
        if fatal_error is not None:
            await WeightWatcher._notify_update_failed(observers, step, fatal_error)
            raise fatal_error

    @staticmethod
    async def _notify_update_failed(
        observers: list[VersionObserver],
        step: int,
        primary_error: BaseException,
    ) -> None:
        for observer in reversed(observers):
            results, cancellation = await gather_shielded(observer.on_version_update_failed(step, primary_error))
            failures = [result for result in results if isinstance(result, BaseException)]
            if cancellation is not None:
                failures.append(cancellation)
            for cleanup_error in failures:
                primary_error.add_note(
                    f"Observer {type(observer).__name__}.on_version_update_failed({step}) also failed: "
                    f"{cleanup_error!r}"
                )

    def gauges(self) -> dict[str, float]:
        return {
            "watcher/policy_version": float(self.policy.version),
            "watcher/update_count": float(self.update_count),
            "watcher/last_update_weights_time": self.last_update_weights_time,
            "watcher/last_wait_for_ckpt_time": self.last_wait_for_ckpt_time,
        }
