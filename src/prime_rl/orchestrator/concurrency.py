"""Adaptive in-flight concurrency control for the rollout dispatcher.

The dispatcher caps concurrency with ``current_limit``; this controller adjusts
that limit at runtime to keep the inference server's GPU KV cache near a target
utilization. The hard part is that KV usage is a *lagging* signal — an admitted
rollout's KV footprint grows over its (possibly hour-long) lifetime, so the
committed load far exceeds the instantaneous reading. Ramping faster than that
materializes overshoots and triggers preemption.

The policy (``decide_limit``) is a gradient-gated AIMD:
  - **grow** only when KV is below target AND has *settled* (small per-tick rise)
    AND we're saturating the current limit — the settle gate paces the ramp to
    the rate at which in-flight rollouts' KV materializes, so we don't overshoot;
  - **back off** (multiplicative) on a KV high-water mark or rising preemptions;
  - otherwise **hold** (a deadband between target and high-water avoids
    oscillation).

The signal comes from polling vLLM's Prometheus ``/metrics`` (GPU
``kv_cache_usage_perc`` + ``num_preemptions``) on the decode engines.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from httpx import AsyncClient

from prime_rl.configs.orchestrator import AdaptiveConcurrencyConfig
from prime_rl.orchestrator.inference_metrics import build_metrics_endpoints, parse_prometheus_text
from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.logger import get_logger

# Fraction of the current limit that must be in flight before we grow it: if the
# limit isn't saturated, KV/permits aren't the bottleneck (env throughput is), so
# raising the ceiling wouldn't help.
SATURATION_FRACTION = 0.9


@dataclass(frozen=True)
class ControlSignal:
    kv_usage: float
    """Max GPU KV cache utilization across the (decode) engines, in [0, 1]."""
    preemption_rate: float
    """Engine preemptions per second since the previous poll."""
    waiting: float
    running: float


def decide_limit(
    *,
    current_limit: int,
    inflight: int,
    kv_usage: float,
    kv_delta: float,
    preemption_rate: float,
    max_inflight: int,
    config: AdaptiveConcurrencyConfig,
) -> tuple[int, str]:
    """Pure control law. Returns ``(new_limit, decision)`` where decision is one
    of ``"grow" | "backoff" | "hold"``. ``kv_delta`` is the per-tick change in the
    smoothed KV utilization."""
    lo, hi = config.min_inflight, max_inflight

    def clamp(n: int) -> int:
        return max(lo, min(hi, n))

    # Congested: KV past the high-water mark, or the engine is preempting. Lower
    # the ceiling toward what's actually running (KV can't be freed instantly, so
    # this just stops new admissions and lets the in-flight set drain).
    if kv_usage >= config.high_water_kv_usage or preemption_rate > config.preemption_rate_threshold:
        base = min(current_limit, max(inflight, lo))
        return clamp(int(base * config.backoff_factor)), "backoff"

    # Grow only when below target, usage has settled at the current concurrency,
    # and we're actually saturating the limit. The growth step tapers to ~1.0 as
    # usage approaches the target so we converge from below instead of overshooting.
    saturating = inflight >= SATURATION_FRACTION * current_limit
    settled = kv_delta <= config.settle_threshold
    if kv_usage < config.target_kv_usage and settled and saturating:
        headroom = (config.target_kv_usage - kv_usage) / config.target_kv_usage
        factor = 1.0 + (config.growth_factor - 1.0) * headroom
        return clamp(max(current_limit + 1, int(current_limit * factor))), "grow"

    return clamp(current_limit), "hold"


class ConcurrencyController:
    """Polls inference KV/preemption metrics and drives ``dispatcher.set_limit``."""

    def __init__(
        self,
        *,
        dispatcher,
        admin_clients: list[AsyncClient],
        roles: list[str | None] | None,
        max_inflight: int,
        config: AdaptiveConcurrencyConfig,
    ) -> None:
        self.dispatcher = dispatcher
        self.config = config
        self.max_inflight = max_inflight
        endpoints = build_metrics_endpoints(admin_clients, roles=roles)
        # KV pressure lives on the decode engines; restrict to them under P/D.
        decode = [e for e in endpoints if e.role == "decode"]
        self.endpoints = decode or endpoints

        self.ewma_kv: float | None = None
        self.kv_delta = 0.0
        self.preemption_rate = 0.0
        self._prev_preemptions: float | None = None
        self._prev_time: float | None = None
        self.last_decision = "init"

        self.stopped = asyncio.Event()
        self.task: asyncio.Task | None = None

    async def start(self) -> None:
        self.task = asyncio.create_task(self._run(), name="concurrency_controller")

    async def _run(self) -> None:
        try:
            while not self.stopped.is_set():
                try:
                    await self._tick()
                except Exception as e:
                    get_logger().debug(f"Concurrency controller tick failed: {e!r}")
                try:
                    await asyncio.wait_for(self.stopped.wait(), timeout=self.config.interval)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            return

    async def _tick(self) -> None:
        signal = await self._poll()
        if signal is None:
            return
        alpha = self.config.ewma_alpha
        prev = self.ewma_kv if self.ewma_kv is not None else signal.kv_usage
        ewma = alpha * signal.kv_usage + (1 - alpha) * prev
        self.kv_delta = ewma - prev
        self.ewma_kv = ewma
        self.preemption_rate = signal.preemption_rate

        before = self.dispatcher.current_limit
        new_limit, decision = decide_limit(
            current_limit=before,
            inflight=self.dispatcher.inflight_permits,
            kv_usage=ewma,
            kv_delta=self.kv_delta,
            preemption_rate=signal.preemption_rate,
            max_inflight=self.max_inflight,
            config=self.config,
        )
        self.last_decision = decision
        self.dispatcher.set_limit(new_limit)
        if new_limit != before:
            get_logger().debug(
                f"adaptive concurrency: {decision} {before}->{new_limit} - kv={ewma:.2f} "
                f"dkv={self.kv_delta:+.3f} preempt/s={signal.preemption_rate:.2f} "
                f"inflight={self.dispatcher.inflight_permits}"
            )

    async def _poll(self) -> ControlSignal | None:
        now = time.monotonic()

        async def fetch(endpoint) -> str | None:
            try:
                response = await endpoint.client.get("/metrics", timeout=5.0)
                response.raise_for_status()
                return response.text
            except Exception:
                return None

        texts = await asyncio.gather(*[fetch(endpoint) for endpoint in self.endpoints])
        rollups = [parse_prometheus_text(text) for text in texts if text is not None]
        if not rollups:
            return None

        kv_values = [value for rollup in rollups for value in rollup.values("kv_cache_usage_perc")]
        kv_usage = max(kv_values) if kv_values else 0.0
        preemptions = sum(rollup.summed("num_preemptions_total") for rollup in rollups)
        waiting = sum(rollup.summed("waiting_requests") for rollup in rollups)
        running = sum(rollup.summed("running_requests") for rollup in rollups)

        rate = 0.0
        if self._prev_preemptions is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 0:
                rate = max(0.0, (preemptions - self._prev_preemptions) / dt)
        self._prev_preemptions = preemptions
        self._prev_time = now

        return ControlSignal(kv_usage=kv_usage, preemption_rate=rate, waiting=waiting, running=running)

    def gauges(self) -> dict[str, float]:
        return {
            "concurrency/limit": float(self.dispatcher.current_limit),
            "concurrency/kv_usage": float(self.ewma_kv or 0.0),
            "concurrency/kv_delta": float(self.kv_delta),
            "concurrency/preemption_rate": float(self.preemption_rate),
        }

    @staticmethod
    def gauge_keys() -> list[str]:
        return ["concurrency/limit", "concurrency/kv_usage", "concurrency/kv_delta", "concurrency/preemption_rate"]

    async def stop(self) -> None:
        self.stopped.set()
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None
