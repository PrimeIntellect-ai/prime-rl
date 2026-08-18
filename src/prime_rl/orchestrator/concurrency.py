"""ConcurrencyController: adaptive cap on in-flight units of inference work.

A unit is whatever the dispatcher admits against one permit — the controller
treats it as a black box. There is no cost model: the controller measures KV
pressure off the engines and reacts at the pipeline's own pace (AIMD):

- **Grow** the cap multiplicatively per pipeline turnover (each completion
  advances the turnover by ``1/inflight``) while the engines are clear, KV
  usage is below ``KV_USAGE_GROW``, and the cap binds admission.
- **Trim** above ``KV_USAGE_TRIGGER``: resize cap and pool to what the
  engines hold at ``KV_USAGE_TARGET`` and cancel the excess, youngest first.
  Cap moves alone cannot relieve pressure from units maturing in place.
- **Cut** on overload: preemptions (single-shot loads) or a persistent
  capacity queue (agentic loads never preempt — admission control parks
  excess load in the waiting queue). Cut, cancel the excess, then freeze
  cuts until the pool drains below the new cap.

The cap starts at ``initial_inflight`` (else the pessimistic bound
``KV capacity / max_model_len``) and is clamped to
``[floor, config.max_inflight]`` throughout.

The controller is a pure state machine — it owns no tasks or clients. The
metrics collector pushes ``observe(samples)`` every poll; the dispatcher
reports ``record_episode(...)`` per completed unit and consumes the cap via
the hooks bound in :meth:`bind`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

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
"""Above this usage, trim the cap and the in-flight pool."""

KV_USAGE_TARGET = 0.7
"""A trim resizes to inflight * target / usage — below the trigger, so pool growth has headroom before the next trim."""

KV_TRIM_COOLDOWN_POLLS = 6
"""Polls between kv-headroom trims, letting each trim propagate before the next is sized."""

QUEUE_RATIO = 0.5
"""HARD once capacity-queued requests exceed this fraction of running requests for the persistence window."""

QUEUE_PERSISTENCE_POLLS = 6
"""Consecutive polls of queue overload before the HARD cut; filters natural turn-completion bursts."""

QUEUE_CUT_FRACTION = 0.9
"""A queue cut targets this fraction of what the engines are serving."""

PREEMPTION_CUT_FRACTION = 0.8
"""A preemption cut targets this fraction of the in-flight pool."""

ESCALATED_CUT_FRACTION = 0.5
"""Cut fraction when overload survives a full drain."""


class Signal(Enum):
    CLEAR = auto()
    SOFT = auto()
    HARD = auto()


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
    def __init__(self, config: ConcurrencyConfig, *, floor: int, fallback_cost: int) -> None:
        self.config = config
        self.floor = floor
        self.fallback_cost = fallback_cost
        """Pessimistic per-unit cost for the starting cap when the engine reports no max context."""

        self.cap = float(config.initial_inflight or floor)
        self.max_inflight = self.clamp(self.cap)
        self.bootstrapped = config.initial_inflight is not None
        self.engine_max_len: int | None = None
        self.capacity_by_engine: dict[str, int] = {}

        self.turnover = 0.0
        self.signal = Signal.CLEAR
        # Growth gate from the last poll, consumed by per-completion growth
        self.can_grow = False
        self.prev_waiting: dict[str, int] = {}
        self.queue_overload_polls = 0
        self.trim_cooldown = 0
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
        inflight = self.get_inflight() if self.get_inflight is not None else 0
        fraction = 1 / max(inflight, self.floor, 1)
        self.turnover += fraction
        if self.can_grow and inflight >= BINDING_FRACTION * self.max_inflight:
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

        worst = Signal.CLEAR
        max_usage = 0.0
        total_running = 0
        total_queued = 0
        preempted = False
        for sample in samples:
            if sample.preemptions_delta > 0:
                preempted = True
                worst = Signal.HARD
            if sample.waiting > 0 and self.prev_waiting.get(sample.engine_id, 0) > 0:
                worst = max(worst, Signal.SOFT, key=lambda s: s.value)
            if sample.role != "prefill":
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
            worst = max(worst, Signal.HARD if queue_overload else Signal.SOFT, key=lambda s: s.value)
        self.signal = worst

        inflight = self.get_inflight() if self.get_inflight is not None else 0
        if self.draining and inflight <= self.max_inflight:
            self.draining = False
            if worst == Signal.CLEAR:
                self.escalated = False
        self.trim_cooldown = max(0, self.trim_cooldown - 1)
        self.can_grow = worst == Signal.CLEAR and total_queued == 0 and not self.draining

        # First capacity observation without a user-set start: derive the
        # pessimistic starting cap, once
        if not self.bootstrapped and self.capacity is not None:
            self.bootstrapped = True
            cost = float(self.engine_max_len or self.fallback_cost)
            self.cap = self.clamp(self.capacity / cost)
            get_logger().info(
                f"Derived initial max inflight {int(self.cap)} - {format_num(self.capacity, precision=1)} "
                f"KV cache tokens / {format_num(cost, precision=1)} tokens per unit"
            )
            self.apply_limit(int(self.cap), reason=None)

        if self.draining:
            return

        if preempted or queue_overload:
            fraction = ESCALATED_CUT_FRACTION if self.escalated else None
            if queue_overload:
                self.queue_overload_polls = 0
                target = int(self.clamp(total_running * (fraction or QUEUE_CUT_FRACTION)))
                reason = "queue overload"
            else:
                target = int(self.clamp(inflight * (fraction or PREEMPTION_CUT_FRACTION)))
                reason = "preemptions"
            self.resize_down(target, inflight, reason=reason)
            self.draining = True
            self.escalated = True
            return

        if max_usage > KV_USAGE_TRIGGER and inflight > 0 and self.trim_cooldown == 0:
            target = int(self.clamp(inflight * KV_USAGE_TARGET / max_usage))
            self.resize_down(target, inflight, reason=f"kv headroom (usage {max_usage:.2f})")
            self.trim_cooldown = KV_TRIM_COOLDOWN_POLLS

    # ── internals ────────────────────────────────────────────────────────────

    def resize_down(self, target: int, inflight: int, *, reason: str) -> None:
        """Lower the cap (never raise) and cancel the in-flight excess."""
        target = min(target, self.max_inflight)
        self.cap = float(target)
        self.apply_limit(target, reason=reason)
        if self.on_overload is not None and inflight > target:
            self.on_overload(inflight - target)

    def apply_limit(self, n_max: int, *, reason: str | None) -> None:
        if n_max == self.max_inflight:
            return
        if reason is not None:
            verb = "Increased" if n_max > self.max_inflight else "Decreased"
            get_logger().info(
                f"{verb} concurrency {self.max_inflight} -> {n_max} ({reason}) - "
                f"turnover={self.turnover:.1f} signal={self.signal.name.lower()}"
            )
        self.max_inflight = n_max
        if self.set_limit is not None:
            self.set_limit(n_max)

    def clamp(self, n_max: float) -> float:
        ceiling = self.config.max_inflight or math.inf
        return min(max(n_max, float(self.floor), float(self.config.min_inflight or 1)), float(ceiling))

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
            "concurrency/signal": float(
                {Signal.CLEAR: 0, Signal.SOFT: 1, Signal.HARD: 2}[self.signal],
            ),
        }
