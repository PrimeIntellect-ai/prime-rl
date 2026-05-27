"""IntervalLogger: wakes every N seconds, emits dispatcher gauges + lag metrics.

Single responsibility — async-native log emission to wandb on the
``_timestamp`` axis. Depends only on the dispatcher (for gauges) and the
policy (for ``policy.version``). The batcher's step-aligned ``monitor.log``
covers per-step metrics; this fills the gaps in between.
"""

from __future__ import annotations

import asyncio
import time

from prime_rl.orchestrator.dispatcher import RolloutDispatcher
from prime_rl.orchestrator.event_loop_lag import EventLoopLagMonitor
from prime_rl.orchestrator.types import Policy
from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.logger import get_logger

# Glob prefix used to declare wandb metrics keyed on ``_timestamp``. Matches
# the InferenceMetricsCollector pattern (``inference/*``) so all async-native
# gauges share a single time-axis convention.
TIME_AXIS_GLOBS = ("dispatcher/*", "policy/version", "event_loop_lag/*")


class IntervalLogger:
    """``await log_loop.start()`` runs until ``stop()``."""

    def __init__(
        self,
        *,
        dispatcher: RolloutDispatcher,
        policy: Policy,
        interval: float,
    ) -> None:
        self.dispatcher = dispatcher
        self.policy = policy
        self.interval = interval
        self.logger = get_logger()

        self.lag_monitor = EventLoopLagMonitor()
        self.lag_task: asyncio.Task | None = None
        self.stopped = asyncio.Event()
        self.wandb_metrics_defined = False

    async def start(self) -> None:
        # Start the event-loop lag sampler alongside the log loop.
        self.lag_task = asyncio.create_task(self.lag_monitor.run(), name="event_loop_lag")
        try:
            while not self.stopped.is_set():
                try:
                    await asyncio.wait_for(self.stopped.wait(), timeout=self.interval)
                    self.emit()
                    return
                except asyncio.TimeoutError:
                    pass
                self.emit()
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        self.stopped.set()
        if self.lag_task is not None:
            await safe_cancel(self.lag_task)
            self.lag_task = None

    def emit(self) -> None:
        gauges = self.dispatcher.gauges()
        gauges.update(self.lag_monitor.get_metrics())
        gauges["policy/version"] = float(self.policy.version)
        gauges["_timestamp"] = time.time()
        self.wandb_log(gauges)

    def wandb_log(self, payload: dict) -> None:
        """Direct ``wandb.log`` write, gated on wandb being enabled.

        Define the time-axis metrics lazily on first call so we don't need a
        wandb dependency at import time.
        """
        try:
            import wandb
        except Exception:
            return
        if wandb.run is None:
            return
        if not self.wandb_metrics_defined:
            for glob in TIME_AXIS_GLOBS:
                try:
                    wandb.define_metric(glob, step_metric="_timestamp")
                except Exception as exc:
                    self.logger.debug(f"wandb.define_metric({glob}) failed: {exc!r}")
            self.wandb_metrics_defined = True
        try:
            wandb.log(payload)
        except Exception as exc:
            self.logger.debug(f"IntervalLogger wandb.log failed: {exc!r}")
