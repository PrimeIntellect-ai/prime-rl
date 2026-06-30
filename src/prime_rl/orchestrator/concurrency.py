"""In-flight concurrency control for the rollout dispatcher.

The dispatcher caps concurrency with ``current_limit``; this controller adjusts
that limit to keep the inference server's GPU KV cache near a target utilization.
KV usage is a *lagging* signal — an admitted rollout's KV footprint grows over its
(possibly hour-long) lifetime, so committed load far exceeds the instantaneous
reading; ramping faster than that materializes overshoots and triggers preemption.

The policy (``decide_limit``) is a gradient-gated AIMD: grow only when KV is below
target AND has *settled* (small per-tick rise) AND the limit is saturated — the
settle gate paces the ramp to the rate at which in-flight rollouts' KV
materializes, so we converge from below instead of overshooting; back off on a KV
high-water mark or rising preemptions; otherwise hold (a deadband avoids
oscillation).

The KV/preemption signal is produced by ``InferenceMetricsCollector`` (the sole
``/metrics`` poller), which pushes a ``ControlSignal`` to ``update`` each poll —
the controller does not poll itself.
"""

from __future__ import annotations

from prime_rl.configs.orchestrator import ConcurrencyConfig
from prime_rl.orchestrator.metrics.inference import ControlSignal
from prime_rl.utils.logger import get_logger

# Fraction of the current limit that must be in flight before we grow it: if the
# limit isn't saturated, KV/permits aren't the bottleneck (env throughput is), so
# raising the ceiling wouldn't help.
SATURATION_FRACTION = 0.9


def decide_limit(
    *,
    current_limit: int,
    inflight: int,
    kv_usage: float,
    kv_delta: float,
    preemption_rate: float,
    min_inflight: int,
    max_inflight: int | None,
    config: ConcurrencyConfig,
) -> tuple[int, str]:
    """Pure control law. Returns ``(new_limit, decision)`` where decision is one
    of ``"grow" | "backoff" | "hold"``. ``kv_delta`` is the per-tick change in the
    smoothed KV utilization. ``min_inflight`` is the lower clamp (``group_size``);
    ``max_inflight`` is the hard upper clamp, or None for no clamp (bounded only by
    the KV target)."""
    lo = min_inflight

    def clamp(n: int) -> int:
        return max(lo, n if max_inflight is None else min(max_inflight, n))

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
    """Drives ``dispatcher.set_limit`` from inference-load samples. Not a poller —
    ``update`` is called by ``InferenceMetricsCollector`` on each metrics poll."""

    def __init__(self, *, dispatcher, min_inflight: int, max_inflight: int | None, config: ConcurrencyConfig) -> None:
        self.dispatcher = dispatcher
        self.config = config
        self.min_inflight = min_inflight
        self.max_inflight = max_inflight
        self.ewma_kv: float | None = None
        self.kv_delta = 0.0

    def update(self, signal: ControlSignal) -> None:
        alpha = self.config.ewma_alpha
        prev = self.ewma_kv if self.ewma_kv is not None else signal.kv_usage
        ewma = alpha * signal.kv_usage + (1 - alpha) * prev
        self.kv_delta = ewma - prev
        self.ewma_kv = ewma

        before = self.dispatcher.current_limit
        new_limit, decision = decide_limit(
            current_limit=before,
            inflight=self.dispatcher.inflight_permits,
            kv_usage=ewma,
            kv_delta=self.kv_delta,
            preemption_rate=signal.preemption_rate,
            min_inflight=self.min_inflight,
            max_inflight=self.max_inflight,
            config=self.config,
        )
        self.dispatcher.set_limit(new_limit)
        if new_limit != before:
            get_logger().debug(
                f"concurrency: {decision} {before}->{new_limit} - kv={ewma:.2f} "
                f"dkv={self.kv_delta:+.3f} preempt/s={signal.preemption_rate:.2f} "
                f"inflight={self.dispatcher.inflight_permits}"
            )
