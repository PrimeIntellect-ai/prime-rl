"""WeightWatcher: discovers new policy versions via the transport's receiver,
advances ``Policy``, and notifies observers (dispatcher → staleness cancel).
Standalone async task; the orchestrator's barrier bounds the in-flight lead."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from prime_rl.orchestrator.types import Policy, VersionObserver
from prime_rl.transports.weights import WeightReceiver
from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.logger import format_time, get_logger


class WeightWatcher:
    """``await watcher.start()`` to drive the polling loop until ``stop()``."""

    def __init__(
        self,
        receiver: WeightReceiver,
        *,
        policy: Policy,
        observers: list[VersionObserver],
        ckpt_step: int = 0,
        poll_interval: float = 1.0,
    ) -> None:
        self.receiver = receiver
        self.policy = policy
        self.observers = observers
        self.ckpt_step = ckpt_step
        self.poll_interval = poll_interval

        self.last_update_weights_time: float = 0.0
        self.last_wait_for_ckpt_time: float = 0.0
        self.update_count: int = 0

        self.task: asyncio.Task | None = None
        self.update_lock = asyncio.Lock()
        self.stopped = asyncio.Event()
        self.update_hooks: list[Callable[[int], Awaitable[None]]] = []

    def on_update(self, hook: Callable[[int], Awaitable[None]]) -> None:
        """Register an async hook that runs after inference applies new weights."""
        self.update_hooks.append(hook)

    async def sync_startup(self, step: int, timeout: float) -> None:
        """Apply the startup policy and notify the registered update hooks."""
        async with self.update_lock:
            await self.receiver.sync_startup(step, timeout)
            self.ckpt_step = step
            self.policy.version = step
            await self._notify_update(step)

    async def start(self) -> None:
        self.task = asyncio.current_task()
        try:
            while not self.stopped.is_set():
                next_step = self.receiver.next_version(self.ckpt_step)
                if next_step > self.ckpt_step:
                    await self.apply_policy_update(next_step)
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        self.stopped.set()
        # Let an in-flight apply finish before dying: the trainer blocks inside
        # its in-memory broadcast until the apply completes, so cancelling
        # mid-apply would strand it. The orchestrator's global teardown budget
        # bounds this wait.
        async with self.update_lock:
            pass
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None

    async def apply_policy_update(self, next_step: int) -> None:
        async with self.update_lock:
            if next_step <= self.ckpt_step:
                # Another caller raced us — bail without re-applying
                return

            t0 = time.perf_counter()
            await self.receiver.wait_published(next_step, cancelled=self.stopped.is_set)
            self.last_wait_for_ckpt_time = time.perf_counter() - t0

            # Record the published version before notifying pending observers.
            # ``policy.version`` advances only after inference applies it.
            self.ckpt_step = next_step

            # Drain stale rollouts BEFORE pausing the inference engines.
            # Aborting a rollout triggers vLLM's KV-connector cleanup (NIXL's
            # ``_reqs_not_processed``), which is only propagated to the workers
            # while the engine is stepping. If we drain after resume instead,
            # the aborts race with the flush of KV transfers that completed
            # during the pause and trip ``assert req_id in self.requests`` in
            # the decode scheduler's ``_update_from_kv_xfer_finished`` — killing
            # the engine and cascading to every DP rank. Draining first lets the
            # aborts settle under normal stepping. ``on_new_version`` (below)
            # still runs post-update for observers that need the live version.
            for observer in self.observers:
                try:
                    await observer.on_version_pending(next_step)
                except Exception as exc:
                    get_logger().warning(
                        f"Observer {type(observer).__name__}.on_version_pending({next_step}) raised: {exc!r}"
                    )

            get_logger().debug(f"Updating inference weights to policy v{next_step}")
            t1 = time.perf_counter()
            await self.receiver.receive(next_step)
            self.last_update_weights_time = time.perf_counter() - t1
            self.update_count += 1
            self.policy.version = next_step
            get_logger().debug(
                f"Updated inference weights to policy v{next_step} in {format_time(self.last_update_weights_time)}"
            )

            await self._notify_update(next_step)

    async def _notify_update(self, step: int) -> None:
        for observer in self.observers:
            try:
                await observer.on_new_version(step)
            except Exception as exc:
                get_logger().warning(f"Observer {type(observer).__name__}.on_new_version({step}) raised: {exc!r}")

        for hook in self.update_hooks:
            await hook(step)

    def gauges(self) -> dict[str, float]:
        return {
            "watcher/policy_version": float(self.policy.version),
            "watcher/update_count": float(self.update_count),
            "watcher/last_update_weights_time": self.last_update_weights_time,
            "watcher/last_wait_for_ckpt_time": self.last_wait_for_ckpt_time,
        }
