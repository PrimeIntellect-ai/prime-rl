"""WeightWatcher: discovers new policy versions through the weight transport's
receiver, applies them to inference, and owns the policy version every other
component reads. Hooks fire around each update: ``on_version_pending`` before the
engines pause for the weight swap, ``on_new_version`` once the new weights are live."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable

from prime_rl.transports.weights import WeightReceiver
from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.logger import format_time, get_logger

VersionHook = Callable[[int], Awaitable[None]]

POLL_INTERVAL = 1.0
"""Seconds between checks for a newer published version."""


class WeightWatcher:
    """``await watcher.start()`` drives the polling loop until ``stop()``;
    ``apply(step)`` moves inference onto one version on demand."""

    def __init__(self, receiver: WeightReceiver) -> None:
        self.receiver = receiver
        self._version = 0
        # The newest version the receiver has published; ``version`` trails it until
        # inference has applied the weights.
        self.published = 0
        self.advanced = asyncio.Event()

        self.last_update_weights_time = 0.0
        self.last_wait_for_ckpt_time = 0.0
        self.update_count = 0

        self.task: asyncio.Task | None = None
        self.update_lock = asyncio.Lock()
        self.stopped = asyncio.Event()
        self._on_version_pending: list[VersionHook] = []
        self._on_new_version: list[VersionHook] = []

    def bind(
        self, *, on_version_pending: Iterable[VersionHook] = (), on_new_version: Iterable[VersionHook] = ()
    ) -> None:
        self._on_version_pending = list(on_version_pending)
        self._on_new_version = list(on_new_version)

    @property
    def version(self) -> int:
        """The policy version inference currently serves."""
        return self._version

    async def sync_startup(self, step: int, timeout: float) -> None:
        """Rendezvous with the trainer's startup broadcast and adopt its version."""
        async with self.update_lock:
            await self.receiver.sync_startup(step, timeout)
            self.published = step
            await self._advance(step)

    async def start(self) -> None:
        self.task = asyncio.current_task()
        try:
            while not self.stopped.is_set():
                next_step = self.receiver.next_version(self.published)
                if next_step > self.published:
                    await self.apply(next_step)
                await asyncio.sleep(POLL_INTERVAL)
        except asyncio.CancelledError:
            return
        finally:
            # Whoever awaits a version must not wait on a watcher that is gone.
            self.stopped.set()
            self.advanced.set()

    async def stop(self) -> None:
        self.stopped.set()
        # Let an in-flight apply finish before dying: the trainer blocks inside its
        # in-memory broadcast until the apply completes, so cancelling mid-apply would
        # strand it. The orchestrator's teardown budget bounds this wait.
        async with self.update_lock:
            pass
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None

    async def apply(self, step: int) -> None:
        """Move inference onto policy ``step``: drain stale work, swap weights, notify."""
        async with self.update_lock:
            if step <= self.published:
                return  # another caller raced us

            t0 = time.perf_counter()
            await self.receiver.wait_published(step, cancelled=self.stopped.is_set)
            self.last_wait_for_ckpt_time = time.perf_counter() - t0
            self.published = step

            # Stale rollouts drain BEFORE the engines pause. Aborting a rollout triggers
            # vLLM's KV-connector cleanup, which only propagates to the workers while the
            # engine is stepping; aborts after resume race the flush of KV transfers that
            # completed during the pause and crash the decode scheduler.
            for hook in self._on_version_pending:
                try:
                    await hook(step)
                except Exception as exc:
                    get_logger().warning(f"on_version_pending({step}) hook raised: {exc!r}")

            get_logger().debug(f"Updating inference weights to policy v{step}")
            t1 = time.perf_counter()
            await self.receiver.receive(step)
            self.last_update_weights_time = time.perf_counter() - t1
            self.update_count += 1
            get_logger().debug(
                f"Updated inference weights to policy v{step} in {format_time(self.last_update_weights_time)}"
            )
            await self._advance(step)

    async def _advance(self, step: int) -> None:
        """Publish the version, then run the post-swap hooks. Their errors propagate:
        an eval that failed to trigger or a gate that failed to move must end the run,
        not leave it waiting on work that will never be scheduled."""
        self._version = step
        self.advanced.set()
        for hook in self._on_new_version:
            await hook(step)

    async def wait_for(self, version: int, *, timeout: float | None = None, reason: str = "") -> bool:
        """Wait until inference serves at least ``version``. Returns False on timeout."""
        if self._version >= version:
            return True
        get_logger().info(f"Waiting for inference to apply policy v{version} {reason}".rstrip())

        async def wait() -> None:
            while self._version < version:
                if self.stopped.is_set():
                    raise RuntimeError(f"weight watcher stopped before inference applied policy v{version}")
                self.advanced.clear()
                if self._version >= version or self.stopped.is_set():
                    continue
                await self.advanced.wait()

        try:
            await asyncio.wait_for(wait(), timeout=timeout)
        except asyncio.TimeoutError:
            get_logger().warning(f"Inference did not apply policy v{version} within {timeout}s — proceeding anyway")
            return False
        return True

    def gauges(self) -> dict[str, float]:
        return {
            "watcher/policy_version": float(self._version),
            "watcher/update_count": float(self.update_count),
            "watcher/last_update_weights_time": self.last_update_weights_time,
            "watcher/last_wait_for_ckpt_time": self.last_wait_for_ckpt_time,
        }
