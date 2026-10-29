"""WeightWatcher: discovers new policy versions via the transport's receiver,
advances ``Policy``, and notifies observers (dispatcher, online evals).
Standalone async task; the orchestrator's barrier bounds the in-flight lead."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from prime_rl.orchestrator.types import Policy, VersionObserver
from prime_rl.transports.batch.recovery import PolicyState
from prime_rl.transports.weights import BroadcastAborted, WeightReceiver
from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.logger import format_time, get_logger

REJOIN_RETRY_INTERVAL_S = 5.0


class WeightWatcher:
    """``await watcher.start()`` to drive the polling loop until ``stop()``."""

    def __init__(
        self,
        receiver: WeightReceiver,
        *,
        policy: Policy,
        observers: list[VersionObserver],
        ckpt_step: int = 0,
        poll_interval: float = 0.1,
        trainer_recovery: bool = False,
    ) -> None:
        self.receiver = receiver
        self.policy = policy
        self.observers = observers
        self.ckpt_step = ckpt_step
        self.poll_interval = poll_interval
        # With trainer recovery, a failed update waits for a replacement trainer instead of
        # failing the run.
        self.trainer_recovery = trainer_recovery
        self.weights_valid = False
        """Inference holds the complete policy ``policy.version``. False until the startup sync
        and after an interrupted in-place update, which may have changed only some weights."""
        self.invalid_since: float | None = None
        """When an update failed and left inference without a complete policy, awaiting a
        replacement trainer; bounded by the receiver's transfer timeout."""
        self.reconnecting: asyncio.Task | None = None

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
            try:
                await self.receiver.sync_startup(step, timeout)
            except Exception as error:
                if not self.trainer_recovery or isinstance(error, TimeoutError):
                    raise
                # The trainer failed before its startup policy reached inference: hold train
                # scheduling until a replacement publishes a complete policy.
                get_logger().warning(f"Trainer failed before its startup broadcast of v{step} completed: {error!r}")
                self.invalid_since = time.monotonic()
                for observer in self.observers:
                    await observer.on_version_pending(step)
                return
            self.ckpt_step = step
            self.policy.version = step
            self.weights_valid = True
            await self._notify_update(step)

    async def reconcile_replacement(self, restored_step: int) -> PolicyState:
        """Prepare the weight channel for a replacement trainer that restored ``restored_step``.

        Waits out the replaced trainer's last update, drops its offers and returns the policy
        inference serves. The replacement recomputes policies up to that version without
        publishing them, unless inference needs its restored policy first.
        """
        self.receiver.abort()
        async with self.update_lock:
            self.receiver.clear_abort()
            if self.reconnecting is not None:
                await safe_cancel(self.reconnecting)
            await asyncio.to_thread(self.receiver.clear_offers)
            served = self.policy.version
            reload = not self.weights_valid or served < restored_step
            # Rolling inference back to the restored policy would end in-flight rollouts' policy
            # spans before they started: an incomplete policy is replaced by the same version.
            version = max(served, restored_step)
            self.ckpt_step = version - 1 if reload else version
            if self.invalid_since is not None:
                self.invalid_since = time.monotonic()
            # The replacement joins the transfer group after it is admitted; the next update
            # waits until inference joined too.
            self.reconnecting = asyncio.create_task(self.receiver.reconnect(), name="weight-reconnect")
        get_logger().info(
            f"Reconciled inference policy v{served} ({'complete' if self.weights_valid else 'incomplete'}) "
            f"with a replacement trainer restored at step {restored_step}"
            + (f"; awaiting its complete v{version}" if reload else "")
        )
        return PolicyState(version, reload)

    async def start(self) -> None:
        self.task = asyncio.current_task()
        try:
            while not self.stopped.is_set():
                timeout = self.receiver.config.timeout
                if self.invalid_since is not None and time.monotonic() - self.invalid_since > timeout:
                    raise RuntimeError(
                        f"Inference has held incomplete weights for {timeout}s without a trainer "
                        "publishing a complete policy"
                    )
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
        if self.reconnecting is not None:
            await safe_cancel(self.reconnecting)
            self.reconnecting = None
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None

    async def apply_policy_update(self, next_step: int) -> None:
        async with self.update_lock:
            if next_step <= self.ckpt_step:
                # Another caller raced us — bail without re-applying
                return
            if self.receiver.next_version(self.ckpt_step) != next_step:
                # A replacement's reconciliation held the lock, dropped the polled offer and may have
                # moved the position back; the next poll finds what the replacement publishes first.
                return

            t0 = time.perf_counter()
            try:
                await self.receiver.wait_published(next_step, cancelled=self.stopped.is_set)
            except BroadcastAborted as error:
                await self._abandon_update(next_step, error)
                return
            self.last_wait_for_ckpt_time = time.perf_counter() - t0

            # Record the published version before notifying pending observers.
            # ``policy.version`` advances only after inference applies it.
            self.ckpt_step = next_step

            # Let observers cancel rollouts BEFORE pausing the inference engines.
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
            try:
                if self.reconnecting is not None:
                    await self._rejoin()
                await self.receiver.receive(next_step)
            except Exception as error:
                if not self.trainer_recovery:
                    raise
                await self._abandon_update(next_step, error)
                return
            self.last_update_weights_time = time.perf_counter() - t1
            self.update_count += 1
            self.policy.version = next_step
            self.weights_valid = True
            self.invalid_since = None
            get_logger().debug(
                f"Updated inference weights to policy v{next_step} in {format_time(self.last_update_weights_time)}"
            )

            await self._notify_update(next_step)

    async def _rejoin(self) -> None:
        """Wait until inference joined the replacement's transfer session. A failed rejoin is retried
        until it succeeds, a newer replacement aborts it or the transfer timeout passes: inference would
        otherwise keep the replaced trainer's session, and the replacement would wait on its offer.
        An abort ends the wait at once, so the newer replacement's reconciliation gets the lock and
        cancels the rejoin instead of waiting out a rendezvous with a dead trainer."""
        deadline = time.monotonic() + self.receiver.config.timeout
        aborted = asyncio.create_task(self.receiver.aborted.wait())
        try:
            while True:
                await asyncio.wait({self.reconnecting, aborted}, return_when=asyncio.FIRST_COMPLETED)
                if self.receiver.aborted.is_set():
                    error = self.reconnecting.exception() if self.reconnecting.done() else None
                    raise BroadcastAborted("A newer replacement trainer superseded this rejoin") from error
                error = self.reconnecting.exception()
                if error is None:
                    self.reconnecting = None
                    return
                if self.stopped.is_set() or time.monotonic() >= deadline:
                    raise error
                get_logger().warning(f"Rejoining the replacement trainer's weight transfer failed, retrying: {error!r}")
                await asyncio.wait({aborted}, timeout=REJOIN_RETRY_INTERVAL_S)
                if self.receiver.aborted.is_set():
                    raise BroadcastAborted("A newer replacement trainer superseded this rejoin")
                self.reconnecting = asyncio.create_task(self.receiver.reconnect(), name="weight-reconnect")
        finally:
            await safe_cancel(aborted)

    async def _abandon_update(self, step: int, error: Exception) -> None:
        """Keep serving after an update failed with its trainer; a replacement continues. The failed
        offer is skipped until a replacement's reconciliation resets the position."""
        self.ckpt_step = step
        if self.reconnecting is not None and self.reconnecting.done():
            self.reconnecting = None
        if isinstance(error, BroadcastAborted):
            get_logger().warning(
                f"Abandoned policy v{step} of a replaced trainer; inference keeps "
                f"{'' if self.weights_valid else 'an incomplete '}v{self.policy.version}"
            )
            if not self.weights_valid:
                # An earlier update was interrupted in place: train scheduling stays held until a
                # replacement publishes a complete policy.
                return
            for observer in self.observers:
                try:
                    await observer.on_new_version(self.policy.version)
                except Exception as exc:
                    get_logger().warning(f"Observer {type(observer).__name__}.on_new_version raised: {exc!r}")
            return
        # The engines stay paused: an interrupted in-place update may have changed some weights.
        self.weights_valid = False
        self.invalid_since = time.monotonic()
        get_logger().opt(exception=error).error(
            f"Updating inference to policy v{step} failed; engines stay paused until a replacement trainer "
            "publishes a complete policy"
        )

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
            **{
                f"watcher/{phase}/{stat}": value
                for phase, times in self.receiver.admin_plane.update_times.items()
                for stat, value in (("min", min(times)), ("mean", sum(times) / len(times)), ("max", max(times)))
            },
        }
