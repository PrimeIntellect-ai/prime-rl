"""ConcurrencyController: adaptive cap on in-flight units of inference work.

A unit is whatever the dispatcher admits against one permit — the controller
treats it as a black box. There is no workload or architecture model: the
controller runs bracketed throughput probes and keeps the largest cap within
five percent of the local throughput maximum:

- **Probe** neighboring caps while the engines are clear and the cap binds
  admission. Each candidate is measured between two incumbent windows to
  reject slow workload drift. Upward probes are accepted while they preserve
  near-peak throughput; downward probes must materially improve it.
- **Trim** above ``KV_USAGE_SOFT_CAP``: lower the cap to what the engines
  hold at ``KV_USAGE_TARGET`` and let completions drain the pool (soft). If
  usage still climbs past ``KV_USAGE_HARD_CAP`` — units maturing in place
  outpace completions — also cancel the excess, youngest first (hard).
- **Cut** on overload: preemptions hard-cut and cancel excess work. A
  persistent capacity queue only lowers the admission cap and lets the pool
  drain naturally; queue pressure can be caused by replica imbalance and is
  not sufficient evidence that active work must be discarded.

The cap starts at ``initial_inflight`` (else the pessimistic bound
``KV capacity / max_model_len``) and is clamped to
``[min_inflight, max_inflight]`` throughout.

The controller is a pure state machine — it owns no tasks or clients. The
metrics collector pushes ``observe(samples)`` every poll; the dispatcher
reports ``record_episode(...)`` per completed unit for turnover observability
and consumes the cap via the hooks bound in :meth:`bind`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from typing import Literal

from prime_rl.configs.orchestrator import ConcurrencyConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import format_num

PROBE_FACTOR = 1.25
"""Multiplicative distance between the incumbent and a probe cap."""

PROBE_THROUGHPUT_TOLERANCE = 0.05
"""Near-peak tolerance for upward probes and minimum gain for downward probes."""

PROBE_WINDOW_POLLS = 3
"""Qualified throughput polls in each baseline or probe window."""

PROBE_SETTLE_POLLS = 1
"""Qualified polls discarded after reaching a newly applied cap."""

PROBE_COOLDOWN_POLLS = 6
"""Qualified incumbent polls between unsuccessful local searches."""

BINDING_FRACTION = 0.9
"""The cap counts as binding when inflight reaches this fraction of it."""

KV_USAGE_GROW = 0.6
"""Probe upward only while every decode engine's KV usage is below this."""

KV_USAGE_SOFT_CAP = 0.8
"""Above this usage, soft-trim: lower the cap and let completions drain the
pool naturally — no work is cancelled."""

KV_USAGE_HARD_CAP = 0.9
"""Above this usage, hard-trim: in-place context growth is outpacing
completions despite the closed gate, so also cancel the pool excess before
thrash onset (a cliff, not a slope)."""

KV_USAGE_TARGET = 0.7
"""A trim resizes to inflight * target / usage — below the trigger, so pool growth has headroom before the next trim."""

KV_TRIM_COOLDOWN_POLLS = 6
"""Polls between kv-headroom trims, letting each trim propagate before the next is sized."""

QUEUE_RATIO = 0.5
"""Overloaded once capacity-queued requests exceed this fraction of running requests for the persistence window."""

QUEUE_PERSISTENCE_POLLS = 6
"""Consecutive polls of queue overload before a soft cut; filters natural turn-completion bursts."""

QUEUE_CUT_FRACTION = 0.9
"""A soft queue cut targets this fraction of the in-flight pool. Pool units,
not engine requests: agentic episodes idle between turns, so engine
``running`` undercounts the pool by the duty cycle and cutting to it
over-cuts."""

PREEMPTION_CUT_FRACTION = 0.8
"""A preemption cut targets this fraction of the in-flight pool."""

ESCALATED_CUT_FRACTION = 0.5
"""Cut fraction when overload survives a full drain."""

ESCALATION_GRACE_POLLS = 6
"""Polls after a drain completes during which a repeat overload cuts at the
escalated fraction; past the grace window the system is considered recovered
and escalation resets."""

Signal = Literal["clear", "soft", "hard"]
"""Engine pressure, in increasing severity."""

SEVERITY: dict[Signal, int] = {"clear": 0, "soft": 1, "hard": 2}


@dataclass(frozen=True)
class EngineLoadSample:
    """Per-engine load facts for one ``/metrics`` poll. Raw values only —
    thresholds and verdicts live in the controller."""

    engine_id: str
    role: str | None
    kv_capacity_tokens: int | None
    max_model_len: int | None
    kv_usage: float
    running: int
    waiting: int
    # Requests queued specifically for KV capacity (None if the engine does
    # not report the by-reason breakdown; fall back to ``waiting``)
    waiting_capacity: int | None
    preemptions_delta: int
    generation_tokens_per_s: float | None = None


ProbePhase = Literal["baseline", "probe", "confirm"]
ProbeDirection = Literal["up", "down"]


class ConcurrencyController:
    def __init__(self, config: ConcurrencyConfig, *, fallback_cost: int) -> None:
        self.config = config
        self.floor = config.min_inflight
        self.fallback_cost = fallback_cost
        """Pessimistic per-unit cost for the starting cap when the engine reports no max context."""

        self.cap = float(config.initial_inflight or self.floor)
        self.max_inflight = self.clamp(self.cap)
        self.bootstrapped = config.initial_inflight is not None
        self.engine_max_len: int | None = None
        self.capacity_by_engine: dict[str, int] = {}

        self.turnover = 0.0
        self.signal: Signal = "clear"
        self.prev_waiting: dict[str, int] = {}
        self.prev_capacity_waiting: dict[str, int] = {}
        self.queue_overload_polls = 0
        self.trim_cooldown = 0
        self.escalation_grace = 0
        # After a cut, ignore further cuts until inflight has drained below
        # the new cap — the overload during drain is stale
        self.draining = False
        self.escalated = False

        self.incumbent = self.max_inflight
        self.probe_phase: ProbePhase = "baseline"
        self.probe_direction: ProbeDirection = "up"
        self.probe_cap: int | None = None
        self.baseline_before: float | None = None
        self.probe_throughput: float | None = None
        self.throughput_samples: list[float] = []
        self.settle_polls = 0
        self.probe_cooldown = 0
        self.throughput_engine_ids: frozenset[str] | None = None

        self.set_limit: Callable[[int], None] | None = None
        self.get_inflight: Callable[[], int] | None = None
        self.on_overload: Callable[[int], None] | None = None

    def bind(
        self,
        *,
        set_limit: Callable[[int], None],
        get_inflight: Callable[[], int],
        on_overload: Callable[[int], None] | None = None,
    ) -> None:
        """Attach the outbound hooks. The dispatcher is constructed with this
        controller's initial cap, so no ``set_limit`` fires here.
        For hard cuts, ``on_overload`` receives the unit excess so the
        dispatcher can cancel in-flight work instead of just blocking
        admission. Soft cuts do not invoke it."""
        self.set_limit = set_limit
        self.get_inflight = get_inflight
        self.on_overload = on_overload

    # ── inbound hooks ────────────────────────────────────────────────────────

    def record_episode(self, env_name: str, kind: str, tokens: int, duration: float) -> None:
        """Advance the scale-independent pipeline-turnover clock."""
        # The dispatcher releases the permit before this hook fires, so add
        # the completing episode back to the turnover denominator.
        inflight = (self.get_inflight() if self.get_inflight is not None else 0) + 1
        fraction = 1 / inflight
        self.turnover += fraction

    def observe(self, samples: list[EngineLoadSample]) -> None:
        """Per-poll engine load push from the metrics collector: classify the
        signal, advance throughput probes, and apply trims and cuts."""
        if not samples:
            return
        for sample in samples:
            if sample.kv_capacity_tokens and sample.role != "prefill":
                self.capacity_by_engine[sample.engine_id] = sample.kv_capacity_tokens
            if sample.max_model_len:
                self.engine_max_len = max(self.engine_max_len or 0, sample.max_model_len)
        # Overload signals read decode engines only: in P/D deployments a
        # prefill queue or prefill preemption is normal flow, not KV pressure
        samples = [sample for sample in samples if sample.role != "prefill"]
        if not samples:
            return

        worst: Signal = "clear"
        max_usage = 0.0
        total_running = 0
        total_queued = 0
        preempted = False
        capacity_queue_persisted = False
        for sample in samples:
            if sample.preemptions_delta > 0:
                preempted = True
                worst = "hard"
            if sample.waiting > 0 and self.prev_waiting.get(sample.engine_id, 0) > 0:
                worst = max(worst, "soft", key=SEVERITY.__getitem__)
            capacity_waiting = sample.waiting_capacity if sample.waiting_capacity is not None else sample.waiting
            if capacity_waiting > 0 and self.prev_capacity_waiting.get(sample.engine_id, 0) > 0:
                capacity_queue_persisted = True
            max_usage = max(max_usage, sample.kv_usage)
            total_running += sample.running
            total_queued += sample.waiting_capacity if sample.waiting_capacity is not None else sample.waiting
        self.prev_waiting = {sample.engine_id: sample.waiting for sample in samples}
        self.prev_capacity_waiting = {
            sample.engine_id: sample.waiting_capacity if sample.waiting_capacity is not None else sample.waiting
            for sample in samples
        }

        if total_running > 0 and total_queued > QUEUE_RATIO * total_running:
            self.queue_overload_polls += 1
        else:
            self.queue_overload_polls = 0
        queue_overload = self.queue_overload_polls >= QUEUE_PERSISTENCE_POLLS
        if queue_overload or max_usage > KV_USAGE_GROW:
            worst = max(worst, "soft", key=SEVERITY.__getitem__)
        self.signal = worst

        inflight = self.get_inflight() if self.get_inflight is not None else 0
        # Release the drain only once the engines have settled too: cuts
        # cancel episodes, so dispatcher inflight drops below the cap almost
        # immediately while the engines are still churning through their own
        # backlog — preemption deltas from that stale churn must not trigger
        # follow-up cuts (observed: an escalated-cut cascade 1024 -> 8, one
        # halving per poll).
        if self.draining and inflight <= self.max_inflight and not preempted and not queue_overload:
            self.draining = False
            self.escalation_grace = ESCALATION_GRACE_POLLS
        if not self.draining and self.escalated:
            # Escalation persists only through the grace window after a
            # drain: a repeat overload inside it cuts harder, anything later
            # means the system recovered (steady state post-cut is SOFT, so
            # gating the reset on a CLEAR poll would latch escalation forever)
            self.escalation_grace -= 1
            if self.escalation_grace <= 0:
                self.escalated = False
        self.trim_cooldown = max(0, self.trim_cooldown - 1)
        # First capacity observation without a user-set start: derive the
        # pessimistic starting cap, once
        if not self.bootstrapped and self.capacity is not None:
            self.bootstrapped = True
            cost = float(self.engine_max_len or self.fallback_cost)
            self.cap = self.clamp(self.capacity / cost)
            get_logger().info(
                f"Derived initial max inflight {int(self.cap)} - {format_num(self.capacity, precision=1)} "
                f"KV cache tokens / {format_num(cost, precision=1)} tokens per episode"
            )
            self.apply_limit(int(self.cap), reason=None)
            self.reset_optimizer(self.max_inflight)

        if self.draining:
            return

        if preempted:
            fraction = ESCALATED_CUT_FRACTION if self.escalated else None
            target = int(self.clamp(inflight * (fraction or PREEMPTION_CUT_FRACTION)))
            self.resize_down(target, inflight, reason="preemptions")
            self.draining = True
            self.escalated = True
            self.reset_optimizer(target, direction="down")
            return

        if queue_overload:
            self.queue_overload_polls = 0
            target = int(self.clamp(inflight * QUEUE_CUT_FRACTION))
            self.resize_down(target, inflight, reason="queue overload", cancel=False)
            self.draining = True
            return

        if max_usage > KV_USAGE_SOFT_CAP and inflight > 0 and self.trim_cooldown == 0:
            hard = max_usage > KV_USAGE_HARD_CAP
            target = int(self.clamp(inflight * KV_USAGE_TARGET / max_usage))
            self.resize_down(
                target,
                inflight,
                reason=f"kv headroom (usage {max_usage:.2f}, {'hard' if hard else 'soft'} trim)",
                cancel=hard,
            )
            self.trim_cooldown = KV_TRIM_COOLDOWN_POLLS
            self.reset_optimizer(target, direction="down")
            return

        if total_queued > 0 and self.probe_phase != "baseline":
            self.abort_probe(inflight, reason="engine queue during throughput probe")
            return
        if max_usage > KV_USAGE_GROW and self.probe_phase != "baseline":
            self.abort_probe(inflight, reason=f"kv headroom during throughput probe (usage {max_usage:.2f})")
            return
        if capacity_queue_persisted and self.probe_phase == "baseline" and self.incumbent > self.floor:
            target = int(self.clamp(math.floor(self.incumbent / PROBE_FACTOR)))
            self.resize_down(target, inflight, reason="persistent engine queue", cancel=False)
            self.reset_optimizer(target, direction="down")
            return

        throughput_values = [sample.generation_tokens_per_s for sample in samples]
        engine_ids = frozenset(sample.engine_id for sample in samples)
        if self.throughput_engine_ids is None:
            self.throughput_engine_ids = engine_ids
        elif engine_ids != self.throughput_engine_ids:
            self.throughput_engine_ids = engine_ids
            self.reset_optimizer(self.max_inflight, cooldown=PROBE_COOLDOWN_POLLS)
            return
        throughput_ready = all(value is not None for value in throughput_values)
        cap_reached = inflight >= BINDING_FRACTION * self.max_inflight and inflight <= self.max_inflight
        qualified = (
            worst == "clear"
            and total_queued == 0
            and throughput_ready
            and sum(value or 0.0 for value in throughput_values) > 0
            and cap_reached
        )
        if qualified:
            self.observe_throughput(sum(value or 0.0 for value in throughput_values), inflight)

    # ── internals ────────────────────────────────────────────────────────────

    def resize_down(self, target: int, inflight: int, *, reason: str, cancel: bool = True) -> None:
        """Lower the cap (never raise) and, with ``cancel``, shed the
        in-flight excess; without it, admission stays blocked until the pool
        drains below the new cap on its own."""
        target = min(target, self.max_inflight)
        self.cap = float(target)
        self.apply_limit(target, reason=reason)
        if cancel and self.on_overload is not None and inflight > target:
            self.on_overload(inflight - target)

    def observe_throughput(self, throughput: float, inflight: int) -> None:
        """Advance one qualified window of the bracketed throughput search."""
        if self.settle_polls > 0:
            self.settle_polls -= 1
            return
        if self.probe_phase == "baseline" and self.probe_cooldown > 0:
            self.probe_cooldown -= 1
            return

        self.throughput_samples.append(throughput)
        if len(self.throughput_samples) < PROBE_WINDOW_POLLS:
            return
        window_throughput = median(self.throughput_samples)
        self.throughput_samples.clear()

        if self.probe_phase == "baseline":
            self.baseline_before = window_throughput
            self.start_probe(inflight)
            return
        if self.probe_phase == "probe":
            self.probe_throughput = window_throughput
            self.probe_phase = "confirm"
            self.cap = float(self.incumbent)
            self.apply_limit(self.incumbent, reason="throughput probe confirmation")
            self.settle_polls = PROBE_SETTLE_POLLS
            return

        assert self.baseline_before is not None
        assert self.probe_throughput is not None
        assert self.probe_cap is not None
        baseline = (self.baseline_before + window_throughput) / 2
        gain = self.probe_throughput / baseline - 1
        direction = self.probe_direction
        probe_cap = self.probe_cap
        if direction == "up":
            accepted = gain >= -PROBE_THROUGHPUT_TOLERANCE
        else:
            accepted = gain >= PROBE_THROUGHPUT_TOLERANCE

        if accepted:
            self.incumbent = probe_cap
            self.cap = float(probe_cap)
            self.apply_limit(
                probe_cap,
                reason=f"throughput probe {direction} accepted ({gain:+.1%})",
            )
            self.probe_direction = direction
            self.probe_cooldown = 0
        else:
            self.cap = float(self.incumbent)
            self.apply_limit(
                self.incumbent,
                reason=f"throughput probe {direction} rejected ({gain:+.1%})",
            )
            self.probe_direction = "down" if direction == "up" else "up"
            self.probe_cooldown = PROBE_COOLDOWN_POLLS
        self.probe_phase = "baseline"
        self.probe_cap = None
        self.baseline_before = None
        self.probe_throughput = None
        self.settle_polls = PROBE_SETTLE_POLLS

    def start_probe(self, inflight: int) -> None:
        direction = self.probe_direction
        if direction == "up":
            target = int(self.clamp(math.ceil(self.incumbent * PROBE_FACTOR)))
            if target == self.incumbent:
                direction = "down"
        if direction == "down":
            target = int(self.clamp(math.floor(self.incumbent / PROBE_FACTOR)))
            if target == self.incumbent:
                direction = "up"
                target = int(self.clamp(math.ceil(self.incumbent * PROBE_FACTOR)))
        if target == self.incumbent:
            self.probe_cooldown = PROBE_COOLDOWN_POLLS
            self.baseline_before = None
            return

        self.probe_direction = direction
        self.probe_cap = target
        self.probe_phase = "probe"
        self.cap = float(target)
        if target < self.incumbent:
            self.resize_down(target, inflight, reason="throughput probe down", cancel=False)
        else:
            self.apply_limit(target, reason="throughput probe up")
        self.settle_polls = PROBE_SETTLE_POLLS

    def abort_probe(self, inflight: int, *, reason: str) -> None:
        target = self.incumbent
        if self.probe_direction == "down" and self.probe_cap is not None:
            target = self.probe_cap
        if self.max_inflight > target:
            self.resize_down(target, inflight, reason=reason, cancel=False)
        else:
            self.cap = float(target)
            self.apply_limit(target, reason=reason)
        self.reset_optimizer(target, cooldown=PROBE_COOLDOWN_POLLS, direction="down")

    def reset_optimizer(
        self,
        incumbent: int,
        *,
        cooldown: int = 0,
        direction: ProbeDirection = "up",
    ) -> None:
        self.incumbent = incumbent
        self.probe_phase = "baseline"
        self.probe_direction = direction
        self.probe_cap = None
        self.baseline_before = None
        self.probe_throughput = None
        self.throughput_samples.clear()
        self.settle_polls = PROBE_SETTLE_POLLS
        self.probe_cooldown = cooldown

    def apply_limit(self, n_max: int, *, reason: str | None) -> None:
        if n_max == self.max_inflight:
            return
        if reason is not None:
            verb = "Increased" if n_max > self.max_inflight else "Decreased"
            get_logger().info(
                f"{verb} concurrency {self.max_inflight} -> {n_max} ({reason}) - "
                f"turnover={self.turnover:.1f} signal={self.signal}"
            )
        self.max_inflight = n_max
        if self.set_limit is not None:
            self.set_limit(n_max)

    def clamp(self, n_max: float) -> float:
        ceiling = self.config.max_inflight or math.inf
        return min(max(n_max, float(self.floor)), float(ceiling))

    @property
    def capacity(self) -> int | None:
        """Total KV tokens across decode engines; None until the first poll."""
        return sum(self.capacity_by_engine.values()) or None

    # ── observability ────────────────────────────────────────────────────────

    def gauges(self) -> dict[str, float]:
        return {
            "concurrency/max_inflight": float(self.max_inflight),
            "concurrency/turnover": self.turnover,
            "concurrency/capacity": float(self.capacity or 0),
            "concurrency/signal": float(SEVERITY[self.signal]),
            "concurrency/incumbent": float(self.incumbent),
            "concurrency/probe_phase": float({"baseline": 0, "probe": 1, "confirm": 2}[self.probe_phase]),
        }
