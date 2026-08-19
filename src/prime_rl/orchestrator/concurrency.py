"""ConcurrencyController: adaptive cap on in-flight units of inference work.

A unit is whatever the dispatcher admits against one permit — the controller
treats it as a black box. There is no cost model: the controller measures KV
pressure off the engines and reacts at the pipeline's own pace (AIMD):

- **Grow** the cap multiplicatively per pipeline turnover (each completion
  advances the turnover by ``1/inflight``) while the engines are clear, KV
  usage is below ``KV_USAGE_GROW``, and the cap binds admission.
- **Trim** above ``KV_USAGE_TRIGGER``: lower the cap to what the engines
  hold at ``KV_USAGE_TARGET`` and let completions drain the pool (soft). If
  usage still climbs past ``KV_USAGE_CANCEL`` — units maturing in place
  outpace completions — also cancel the excess, youngest first (hard).
- **Cut** on overload: preemptions (single-shot loads) or a persistent
  capacity queue (agentic loads never preempt — admission control parks
  excess load in the waiting queue). Cut, cancel the excess, then freeze
  cuts until the pool drains below the new cap.

The cap starts at ``initial_inflight`` (else the pessimistic bound
``KV capacity / max_model_len``) and is clamped to
``[min_inflight, max_inflight]`` throughout.

The controller is a pure state machine — it owns no tasks or clients. The
metrics collector pushes ``observe(samples)`` every poll; the dispatcher
reports ``record_episode(...)`` per completed unit and consumes the cap via
the hooks bound in :meth:`bind`.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from prime_rl.configs.orchestrator import ConcurrencyConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import format_num

TURNOVER_GROWTH = 1.25
"""Cap growth per pipeline turnover while the engines are clear and the cap binds."""

BINDING_FRACTION = 0.9
"""The cap counts as binding when inflight reaches this fraction of it."""

KV_USAGE_GROW = 0.6
"""Grow only while every decode engine's KV usage is below this."""

KV_USAGE_TRIGGER = 0.8
"""Above this usage, soft-trim: lower the cap and let completions drain the
pool naturally — no work is cancelled."""

KV_USAGE_CANCEL = 0.9
"""Above this usage, hard-trim: in-place context growth is outpacing
completions despite the closed gate, so also cancel the pool excess before
thrash onset (a cliff, not a slope)."""

KV_USAGE_TARGET = 0.7
"""A trim resizes to inflight * target / usage — below the trigger, so pool growth has headroom before the next trim."""

KV_TRIM_COOLDOWN_POLLS = 6
"""Polls between kv-headroom trims, letting each trim propagate before the next is sized."""

QUEUE_RATIO = 0.5
"""HARD once capacity-queued requests exceed this fraction of running requests for the persistence window."""

QUEUE_PERSISTENCE_POLLS = 6
"""Consecutive polls of queue overload before the HARD cut; filters natural turn-completion bursts."""

QUEUE_CUT_FRACTION = 0.9
"""A queue cut targets this fraction of the in-flight pool. Pool units, not
engine requests: agentic episodes idle between turns, so engine ``running``
undercounts the pool by the duty cycle and cutting to it over-cuts."""

PREEMPTION_CUT_FRACTION = 0.8
"""A preemption cut targets this fraction of the in-flight pool."""

ESCALATED_CUT_FRACTION = 0.5
"""Cut fraction when overload survives a full drain."""

ESCALATION_GRACE_POLLS = 6
"""Polls after a drain completes during which a repeat overload cuts at the
escalated fraction; past the grace window the system is considered recovered
and escalation resets."""

GROWTH_GATE_TTL_S = 15.0
"""The growth gate expires this long after the poll that set it — if the
metrics path stalls (an API server too loaded to answer is exactly the state
runaway growth creates), growth freezes instead of compounding blind."""


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
        # Growth gate from the last poll, consumed by per-completion growth
        self.can_grow = False
        self.can_grow_until = 0.0
        self.prev_waiting: dict[str, int] = {}
        self.queue_overload_polls = 0
        self.trim_cooldown = 0
        self.escalation_grace = 0
        # After a cut, ignore further cuts until inflight has drained below
        # the new cap — the overload during drain is stale
        self.draining = False
        self.escalated = False

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
        ``on_overload`` receives the unit excess on a downward resize so the
        dispatcher can cancel in-flight work instead of just blocking
        admission."""
        self.set_limit = set_limit
        self.get_inflight = get_inflight
        self.on_overload = on_overload

    # ── inbound hooks ────────────────────────────────────────────────────────

    def record_episode(self, env_name: str, kind: str, tokens: int, duration: float) -> None:
        """One completed unit (from the dispatcher). Advances the turnover
        clock and, while the last poll was green, grows the cap — so growth
        is paced by the pipeline itself at any concurrency scale."""
        # The dispatcher releases the permit before this hook fires, so the
        # completing episode is already off the count — add it back, or the
        # binding check is unsatisfiable for caps below 10 (0.9 * cap > cap - 1)
        inflight = (self.get_inflight() if self.get_inflight is not None else 0) + 1
        fraction = 1 / inflight
        self.turnover += fraction
        if tokens <= 0:
            # Error markers consumed no engine capacity: a slow-failing error
            # storm keeps the pool binding while idle engines read CLEAR, and
            # growing on it floods the engines once the environment recovers
            return
        if (
            self.can_grow
            and time.monotonic() < self.can_grow_until
            and inflight >= BINDING_FRACTION * self.max_inflight
        ):
            self.cap = self.clamp(self.cap * TURNOVER_GROWTH**fraction)
            self.apply_limit(int(self.cap), reason=None)

    def observe(self, samples: list[EngineLoadSample]) -> None:
        """Per-poll engine load push from the metrics collector: classify the
        signal, gate growth, and apply trims and cuts."""
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
        for sample in samples:
            if sample.preemptions_delta > 0:
                preempted = True
                worst = "hard"
            if sample.waiting > 0 and self.prev_waiting.get(sample.engine_id, 0) > 0:
                worst = max(worst, "soft", key=SEVERITY.__getitem__)
            max_usage = max(max_usage, sample.kv_usage)
            total_running += sample.running
            total_queued += sample.waiting_capacity if sample.waiting_capacity is not None else sample.waiting
        self.prev_waiting = {sample.engine_id: sample.waiting for sample in samples}

        if total_running > 0 and total_queued > QUEUE_RATIO * total_running:
            self.queue_overload_polls += 1
        else:
            self.queue_overload_polls = 0
        queue_overload = self.queue_overload_polls >= QUEUE_PERSISTENCE_POLLS
        if queue_overload or max_usage > KV_USAGE_GROW:
            worst = max(worst, "hard" if queue_overload else "soft", key=SEVERITY.__getitem__)
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
        self.can_grow = worst == "clear" and total_queued == 0 and not self.draining
        self.can_grow_until = time.monotonic() + GROWTH_GATE_TTL_S

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

        if self.draining:
            return

        if preempted or queue_overload:
            fraction = ESCALATED_CUT_FRACTION if self.escalated else None
            if queue_overload:
                self.queue_overload_polls = 0
                target = int(self.clamp(inflight * (fraction or QUEUE_CUT_FRACTION)))
                reason = "queue overload"
            else:
                target = int(self.clamp(inflight * (fraction or PREEMPTION_CUT_FRACTION)))
                reason = "preemptions"
            self.resize_down(target, inflight, reason=reason)
            self.draining = True
            self.escalated = True
            return

        if max_usage > KV_USAGE_TRIGGER and inflight > 0 and self.trim_cooldown == 0:
            hard = max_usage > KV_USAGE_CANCEL
            target = int(self.clamp(inflight * KV_USAGE_TARGET / max_usage))
            self.resize_down(
                target,
                inflight,
                reason=f"kv headroom (usage {max_usage:.2f}, {'hard' if hard else 'soft'} trim)",
                cancel=hard,
            )
            self.trim_cooldown = KV_TRIM_COOLDOWN_POLLS

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
        }
