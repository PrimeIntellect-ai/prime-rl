"""In-flight concurrency control for the rollout dispatcher.

The dispatcher caps concurrency with ``current_limit``; this controller adjusts it.
The control law (``decide_limit``) is a conservative AIMD biased to under- rather
than over-shoot: it backs off (multiplicative) on the first sign of congestion, and
grows gently (small multiplicative step) only when *every* gate passes — anything
ambiguous holds (fail-safe). Overshoot is far more costly here (runaway admission,
preemption thrash, leaked runtime sandboxes) than leaving a little throughput on the
table.

The grow *driver* is **inference generation throughput (tokens/sec)** scraped from the
decode engines: a continuous signal that rises with concurrency until the engine
saturates, then plateaus — and, unlike rollout completion rate, stays meaningful for
hour-long agentic rollouts. A grow requires throughput to have risen since the last
grow (gradient) AND a minimum wall-clock dwell since the last change (so the smoothed
TPS is averaged over enough ticks to be reliable). KV high-water and preemptions are
hard backoff guards; the KV target is only a permissive headroom gate. Pacing is
time- + signal-based with no dependence on rollout completions, so one hparam set
suits both fast single-turn and slow agentic rollouts.

Every tick logs (debug) the decision + per-gate breakdown so a "stuck" hold shows
exactly which gate is blocking growth.

``update`` is called by ``InferenceMetricsCollector`` on each ``/metrics`` poll.
"""

from __future__ import annotations

import statistics
import time
from collections import deque

from prime_rl.configs.orchestrator import ConcurrencyConfig
from prime_rl.orchestrator.metrics.inference import ControlSignal
from prime_rl.utils.logger import get_logger

# Fraction of the current limit that must be in flight before we grow it: if the limit
# isn't saturated, permits aren't the bottleneck — the dispatcher is throttled upstream
# (slow trainer / off-policy lag), so raising the ceiling wouldn't help.
SATURATION_FRACTION = 0.9


def decide_limit(
    *,
    current_limit: int,
    inflight: int,
    kv_usage: float,
    kv_delta: float,
    preemption_rate: float,
    seconds_since_change: float,
    throughput: float,
    throughput_ref: float,
    min_inflight: int,
    max_inflight: int | None,
    config: ConcurrencyConfig,
) -> tuple[int, str, dict]:
    """Pure, conservative control law -> ``(new_limit, decision, gates)`` with decision
    in ``"grow" | "backoff" | "hold"``. ``gates`` reports each grow precondition so a
    hold is explainable. ``throughput`` is the smoothed inference generation TPS and
    ``throughput_ref`` the TPS captured at the last grow (the gradient baseline)."""
    lo = min_inflight

    def clamp(n: int) -> int:
        return max(lo, n if max_inflight is None else min(max_inflight, n))

    gates = {
        # all must be true to grow; anything ambiguous -> hold (fail-safe)
        # actually using the current limit (else dispatcher is throttled upstream)
        "saturating": inflight >= SATURATION_FRACTION * current_limit,
        # enough wall-clock since the last change that smoothed TPS is reliable
        "enough_time": seconds_since_change >= config.min_seconds_between_grows,
        "settled": kv_delta <= config.settle_threshold,
        "below_target": kv_usage < config.target_kv_usage,
        # generation TPS rose since the last grow -> more concurrency still pays off
        "improving": throughput_ref <= 0.0 or throughput >= throughput_ref * (1.0 + config.min_throughput_gain),
        "seconds_since_change": seconds_since_change,
        "congested": kv_usage >= config.high_water_kv_usage or preemption_rate > config.preemption_rate_threshold,
    }

    # Hair-trigger backoff: any congestion (KV high-water or preemptions) -> cut the
    # ceiling toward what's actually running. Cheaper to re-grow than overshoot.
    if gates["congested"]:
        base = min(current_limit, max(inflight, lo))
        return clamp(int(base * config.backoff_factor)), "backoff", gates

    # Grow only when below KV target, usage has settled, the min-time dwell has elapsed,
    # AND the median-TPS gradient is still improving (more concurrency keeps paying off).
    if (
        gates["saturating"]
        and gates["enough_time"]
        and gates["settled"]
        and gates["below_target"]
        and gates["improving"]
    ):
        return clamp(max(current_limit + 1, int(current_limit * config.growth_factor))), "grow", gates

    return clamp(current_limit), "hold", gates


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
        # Throughput = MEDIAN total generation TPS over the current segment (since the
        # limit last changed). Per-poll TPS is very noisy/spiky at high concurrency, so a
        # median over the segment's samples is robust to outliers (vs an EWMA or mean
        # that a single burst drags); ``throughput_ref`` is that median at the last grow.
        # The window is capped so a long hold at the knee can't grow it unbounded, and
        # sized to cover the min-time dwell; it's cleared on each limit change.
        self.throughput = 0.0
        self.throughput_ref = 0.0
        self._tps_window: deque[float] = deque(
            maxlen=max(8, int(2 * config.min_seconds_between_grows / config.interval))
        )
        self._t_at_change: float | None = None

    def update(self, signal: ControlSignal) -> None:
        alpha = self.config.ewma_alpha
        prev = self.ewma_kv if self.ewma_kv is not None else signal.kv_usage
        ewma = alpha * signal.kv_usage + (1 - alpha) * prev
        self.kv_delta = ewma - prev
        self.ewma_kv = ewma

        # Segment-MEDIAN total generation throughput (tokens/s): median over the polls
        # since the limit last changed, robust to the per-poll TPS spikes (the min-time
        # gate ensures enough samples before this is acted on).
        self._tps_window.append(signal.tps)
        self.throughput = statistics.median(self._tps_window)

        now = time.monotonic()
        if self._t_at_change is None:
            self._t_at_change = now
        seconds_since_change = now - self._t_at_change

        before = self.dispatcher.current_limit
        inflight = self.dispatcher.inflight_permits
        new_limit, decision, g = decide_limit(
            current_limit=before,
            inflight=inflight,
            kv_usage=ewma,
            kv_delta=self.kv_delta,
            preemption_rate=signal.preemption_rate,
            seconds_since_change=seconds_since_change,
            throughput=self.throughput,
            throughput_ref=self.throughput_ref,
            min_inflight=self.min_inflight,
            max_inflight=self.max_inflight,
            config=self.config,
        )
        self.dispatcher.set_limit(new_limit)
        if new_limit != before:
            # On grow, the old segment's median TPS is the bar the new segment must beat;
            # on backoff, re-baseline so growth is re-earned from the cut level.
            self.throughput_ref = self.throughput if decision == "grow" else 0.0
            self._tps_window.clear()
            self._t_at_change = now

        # Every tick: decision + per-gate breakdown. On a "hold" the False gates are
        # exactly what's blocking growth.
        get_logger().debug(
            f"concurrency: {decision} {before}->{new_limit} inflight={inflight}/{before} | "
            f"saturated={g['saturating']} time={g['enough_time']} settled={g['settled']} below_target={g['below_target']} improving={g['improving']} | "
            f"kv={ewma:.3f} dkv={self.kv_delta:+.4f} tps={self.throughput:.0f} ref={self.throughput_ref:.0f} "
            f"preempt/s={signal.preemption_rate:.2f}"
        )
