"""IntervalLogger: wakes every N seconds, emits dispatcher gauges + lag metrics.

Lives on its own time axis (wandb ``_timestamp``) instead of step semantics —
the v2 design's "async-native logs" pillar. The batcher's per-step ``monitor.log``
covers step-aligned metrics (reward, seq_len, off-policy levels, filter rates);
this task fills the gaps between ships so the dispatcher's real-time state
(in-flight counts, sched_mode, queue depth, semaphore availability) is visible
even when batches are slow or absent.

Writes directly to wandb (with ``_timestamp`` as the step metric) rather than
going through ``monitor.log``, so it does not clobber step-aligned plots.

Concurrent with the rest of the pipeline — does not block dispatcher / batcher
work.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.event_loop_lag import EventLoopLagMonitor
from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector
from prime_rl.orchestrator_v2.policy import Policy
from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator_v2.batcher import TrainBatcher
    from prime_rl.orchestrator_v2.dispatcher import RolloutDispatcher


# Glob prefix used to declare wandb metrics keyed on ``_timestamp``. Matches
# the InferenceMetricsCollector pattern (``inference/*``) so all async-native
# gauges share a single time-axis convention.
_TIME_AXIS_GLOBS = ("dispatcher/*", "batcher/last_*", "policy/version", "event_loop_lag/*")


class IntervalLogger:
    """One ``asyncio.Task``. Reads gauges from dispatcher + lag monitor.

    Bypasses ``monitor.log`` for the interval emit and writes directly to wandb
    with ``_timestamp`` as the step metric so gauges plot on wall-clock time
    independent of training step (avoids overwriting per-step batcher logs).
    The ``InferenceMetricsCollector`` already does its own ``inference/*``
    metrics with the same pattern; we just keep a reference so the orchestrator
    can stop it on shutdown.
    """

    def __init__(
        self,
        *,
        config: OrchestratorConfig,
        dispatcher: "RolloutDispatcher",
        batcher: "TrainBatcher",
        policy: Policy,
        inference_metrics: InferenceMetricsCollector | None,
        monitor,
    ) -> None:
        self.config = config
        self.dispatcher = dispatcher
        self.batcher = batcher
        self.policy = policy
        self.inference_metrics = inference_metrics
        self.monitor = monitor
        self.interval = config.experimental.log_loop_interval
        self.logger = get_logger()

        self._lag_monitor = EventLoopLagMonitor()
        self._lag_task: asyncio.Task | None = None
        self._loop_task: asyncio.Task | None = None
        self._stopped = asyncio.Event()
        self._wandb_metrics_defined = False

    async def run(self) -> None:
        self._loop_task = asyncio.current_task()
        # Start the event-loop lag sampler alongside the log loop.
        self._lag_task = asyncio.create_task(self._lag_monitor.run(), name="event_loop_lag")
        try:
            while not self._stopped.is_set():
                try:
                    await asyncio.wait_for(self._stopped.wait(), timeout=self.interval)
                    self._emit()
                    return
                except asyncio.TimeoutError:
                    pass
                self._emit()
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        self._stopped.set()
        if self._lag_task is not None:
            await safe_cancel(self._lag_task)
            self._lag_task = None
        if self._loop_task is not None:
            await safe_cancel(self._loop_task)
            self._loop_task = None
        if self.inference_metrics is not None:
            await self.inference_metrics.stop()

    def _emit(self) -> None:
        gauges = self.dispatcher.gauges()
        gauges.update(self._lag_monitor.get_metrics())
        gauges["policy/version"] = float(self.policy.version)
        if self.batcher.last_batch_step is not None:
            gauges["batcher/last_step"] = float(self.batcher.last_batch_step)
            gauges["batcher/last_async_level"] = float(self.batcher.last_batch_step - self.policy.version)
        if self.batcher.last_batch_reward is not None:
            gauges["batcher/last_reward"] = float(self.batcher.last_batch_reward)
        if self.batcher.last_batch_size_shipped is not None:
            gauges["batcher/last_size_shipped"] = float(self.batcher.last_batch_size_shipped)
        gauges["_timestamp"] = time.time()
        self._wandb_log(gauges)

    def _wandb_log(self, payload: dict) -> None:
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
        if not self._wandb_metrics_defined:
            for glob in _TIME_AXIS_GLOBS:
                try:
                    wandb.define_metric(glob, step_metric="_timestamp")
                except Exception as exc:
                    self.logger.debug(f"wandb.define_metric({glob}) failed: {exc!r}")
            self._wandb_metrics_defined = True
        try:
            wandb.log(payload)
        except Exception as exc:
            self.logger.debug(f"IntervalLogger wandb.log failed: {exc!r}")
