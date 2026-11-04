"""ConcurrencyController: adaptive cap on in-flight units of inference work.

A unit is whatever the dispatcher admits against one permit — the controller
treats it as a black box. There is no cost model: the controller measures load
off the engines and reacts at the pipeline's own pace (AIMD).

Engines are pooled per role (prefill, decode, or colocated) and the worst role
binds, so in P/D deployments a saturated prefill stage holds the cap as much as
a saturated decode stage. Per poll the controller reads:

- **saturation**: queued over running requests, where queued is every waiting
  request — waiting for capacity or deferred (e.g. waiting on a KV load).
- **preemption rate**: preemptions per finished request over the window, on
  decode (or colocated) engines.
- **KV usage**: the p90 over decode (or colocated) engines of the GPU-KV gauge
  (the HiSparse host pool gauge where exported). The max over engines spikes
  whenever a few long contexts run on one engine at once.

and acts:

- **Grow** the cap multiplicatively per pipeline turnover (each completion
  advances the turnover by ``1/inflight``) while nothing is queued and the cap
  binds admission. The multiplier tapers linearly with KV usage until the
  soft-cap boundary.
- **Trim** above ``KV_USAGE_SOFT_CAP``: lower the cap to what the engines
  hold at ``KV_USAGE_TARGET`` and let completions drain the pool (soft). If
  usage still climbs past ``KV_USAGE_HARD_CAP`` — units maturing in place
  outpace completions — also cancel the excess, youngest first (hard).
- **Cut** on overload: a preemption rate above ``PREEMPTION_RATE_CLIFF`` or a
  persistent queue (agentic loads never preempt — admission control parks
  excess load in the waiting queue). Cut, cancel the excess, then freeze
  cuts until the pool drains below the new cap.

With ``config.recompute`` set and a measured recompute ratio (the share of
each call's previously computed prefix the engines prefilled again), waste —
that ratio plus, in P/D, the share of prefills whose KV lease expired before
decode pulled it — replaces KV usage for the growth taper and the trims. The KV
gauge counts only running requests, so it misses the evicted caches of episodes
idle between turns. Growth is then paced by finished model calls
(``CALL_GROWTH_MAX`` per turn turnover) rather than whole episodes, saturation
up to ``QUEUE_TOLERANCE`` (turns arriving together) does not hold it back, waste
brakes only while the engines are busy, and preemptions trim without cancelling.
The persistent-queue cut stays as the backstop.

While the engines are paused for a weight update the controller freezes, and
the queue that builds up behind the pause is ignored for a persistence window
after it.

The cap starts at ``initial_inflight`` (else the pessimistic bound
``KV capacity / max_model_len``) and is clamped to
``[min_inflight, max_inflight]`` throughout, unless a manual target pins it.

The controller is a pure state machine — it owns no tasks or clients. The
metrics collector pushes ``observe(samples)`` every poll; the dispatcher
reports ``record_episode(...)`` per completed unit, ``record_calls(...)`` per
model call and ``pause()``/``resume()`` around weight updates, and consumes the
cap via the hooks bound in :meth:`bind`.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from prime_rl.configs.orchestrator import ConcurrencyConfig
from prime_rl.control.types import ConfigUpdate, Reconfigurable
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import format_num

TURNOVER_GROWTH_MAX = 1.2
"""Cap growth per pipeline turnover at zero GPU-KV utilization."""

BINDING_FRACTION = 0.9
"""The cap counts as binding when inflight reaches this fraction of it."""

KV_USAGE_SOFT_CAP = 0.8
"""Above this usage, soft-trim: lower the cap and let completions drain the
pool naturally — no work is cancelled."""

KV_USAGE_HARD_CAP = 0.9
"""Above this usage, hard-trim: in-place context growth is outpacing
completions despite the closed gate, so also cancel the pool excess before
thrash onset (a cliff, not a slope)."""

KV_USAGE_TARGET = 0.7
"""A trim resizes to inflight * target / usage — below the trigger, so pool growth has headroom before the next trim."""

KV_USAGE_QUANTILE = 0.9
"""KV usage is this quantile over a role's engines: one hot engine under sticky routing is a placement problem, not
pool pressure, and trimming the whole pool for it under-utilizes the rest."""

KV_TRIM_COOLDOWN_POLLS = 6
"""Polls between kv-headroom trims, letting each trim propagate before the next is sized."""

QUEUE_RATIO = 0.5
"""HARD once saturation (queued over running requests) exceeds this for the persistence window."""

QUEUE_PERSISTENCE_POLLS = 6
"""Consecutive polls of queue overload before the HARD cut; filters natural turn-completion bursts."""

QUEUE_CUT_FRACTION = 0.9
"""A queue cut targets this fraction of the in-flight pool. Pool units, not
engine requests: agentic episodes idle between turns, so engine ``running``
undercounts the pool by the duty cycle and cutting to it over-cuts."""

PREEMPTION_RATE_CLIFF = 0.01
"""HARD once a role's preemptions exceed this fraction of its finished requests over the window. A rate, not "any
engine preempted": at 64 engines a 1% per-engine chance per poll fires on half the polls."""

PREEMPTION_CUT_FRACTION = 0.8
"""A preemption cut targets this fraction of the in-flight pool."""

ESCALATED_CUT_FRACTION = 0.5
"""Cut fraction when overload survives a full drain."""

ESCALATION_GRACE_POLLS = 6
"""Polls after a drain completes during which a repeat overload cuts at the
escalated fraction; past the grace window the system is considered recovered
and escalation resets."""

LOAD_WINDOW_POLLS = 12
"""Polls the recompute ratio, the preemption and expired-lease rates and the busy share are measured over."""

RECOMPUTE_MIN_PREFIX_TOKENS = 100_000
"""Prefix tokens the window needs before its recompute ratio counts as measured."""

RECOMPUTE_TRIM_FRACTION = 0.9
"""A waste trim lowers the cap to this fraction of the in-flight pool."""

EXPIRED_LEASE_TRIM = 0.02
"""With the recompute ratio measured: while busy, trim once this share of prefills had their KV lease expire before
decode pulled it (P/D). Each one is a whole prefill thrown away; healthy P/D expires none."""

GROWTH_GATE_TTL_S = 15.0
"""The growth gate expires this long after the poll that set it — if the
metrics path stalls (an API server too loaded to answer is exactly the state
runaway growth creates), growth freezes instead of compounding blind."""

CALL_GROWTH_MAX = 1.1
"""With the recompute ratio measured: the most the cap grows per turn turnover (each finished model call advances it
by ``1/inflight``). Agentic episodes run many calls, so this paces growth by turns rather than whole episodes."""

CEILING_PROBE = 0.1
"""With the recompute ratio measured: near the cap of the last preemption trim, growth slows to this fraction of its
full rate, rising back to full as the cap moves away below it (CUBIC-style), and stays at it above it to keep probing."""

QUEUE_TOLERANCE = 0.1
"""With the recompute ratio measured: saturation up to this is the normal burst of turns arriving together, so it
neither marks the signal SOFT nor blocks growth. Without it, any queued request does."""

BUSY_QUEUE_FRACTION = 0.25
"""With the recompute ratio measured: waste brakes growth (and trims) only while the engines are busy, i.e. some
queue showed up in at least this fraction of the window's polls. On idle engines a recomputed prefix only costs
spare prefill compute, so admitting more episodes still raises throughput."""

COLOCATED = "colocated"
"""Role of engines that both prefill and decode (no P/D role)."""


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
    # All waiting requests: queued for capacity plus deferred (KV loads, grammar compilation)
    waiting: int
    # The capacity-queued part of ``waiting`` (None if the engine does not report the by-reason breakdown)
    waiting_capacity: int | None
    preemptions_delta: int
    finished_delta: int = 0
    # Prefills whose KV lease expired before decode pulled it (NIXL, prefill engines)
    expired_kv_delta: int = 0
    # HiSparse host KV pool usage, where the engine keeps its KV there
    host_kv_usage: float | None = None


@dataclass(frozen=True)
class RoleLoad:
    """One poll's engines of one role, pooled."""

    running: int
    queued: int
    preemptions: int
    finished: int
    expired: int
    kv_usage: float

    @property
    def saturation(self) -> float:
        return self.queued / max(self.running, 1)


class ConcurrencyController(Reconfigurable):
    def __init__(self, config: ConcurrencyConfig, *, fallback_cost: int) -> None:
        self.config = config
        self.floor = config.min_inflight
        self.fallback_cost = fallback_cost
        """Pessimistic per-unit cost for the starting cap when the engine reports no max context."""

        self.cap = float(config.initial_inflight or self.floor)
        self.max_inflight = self.clamp(self.cap)
        self.bootstrapped = config.initial_inflight is not None
        # A manual cap that overrides the band and the adaptive loop until cleared
        self.target: int | None = None
        self.engine_max_len: int | None = None
        self.capacity_by_engine: dict[str, int] = {}

        self.turnover = 0.0
        self.signal: Signal = "clear"
        self.growth_multiplier = 1.0
        # Growth gate from the last poll, consumed by per-completion growth
        self.can_grow = False
        self.can_grow_until = 0.0
        self.queue_overload_polls = 0
        self.trim_cooldown = 0
        self.escalation_grace = 0
        # After a cut, ignore further cuts until inflight has drained below
        # the new cap — the overload during drain is stale
        self.draining = False
        self.escalated = False
        # Engines paused for a weight update, and the polls after resume whose queue is the pause's backlog
        self.paused = False
        self.resume_skip = 0
        # Pooled load per role, per poll of the window
        self.load_polls: deque[dict[str, RoleLoad]] = deque(maxlen=LOAD_WINDOW_POLLS)
        self.saturation_by_role: dict[str, float] = {}
        self.saturation = 0.0
        self.prev_saturated = False
        self.preemption_rate = 0.0
        self.expired_lease_rate = 0.0
        self.kv_usage = 0.0
        # Per-poll (prefix, recompute) call tokens, and the ratio over the window
        self.recompute_polls: deque[tuple[int, int]] = deque(maxlen=LOAD_WINDOW_POLLS)
        # Per poll of the same window: whether any request was queued
        self.queued_polls: deque[bool] = deque(maxlen=LOAD_WINDOW_POLLS)
        self.pending_prefix = 0
        self.pending_recompute = 0
        self.recompute_ratio: float | None = None
        self.recompute_trim_wait = 0
        self.call_growth_multiplier = 1.0
        # The cap at the last preemption trim under recompute control: growth slows near it
        self.preemption_ceiling: float | None = None

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

    def record_episode(self, tokens: int) -> None:
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
        if self.recompute_control:
            return  # growth is paced by calls instead (record_calls)
        self._grow(inflight, self.growth_multiplier**fraction)

    def record_calls(self, prefix: int, recompute: int) -> None:
        """Prefix and recompute tokens of a finished model call (from the dispatcher's
        live deltas), folded into the recompute ratio at the next poll. With the ratio
        measured, each call also grows the cap by its share of a turn turnover."""
        self.pending_prefix += prefix
        self.pending_recompute += recompute
        if self.recompute_control:
            inflight = self.get_inflight() if self.get_inflight is not None else 0
            if inflight > 0:
                self._grow(inflight, self.call_growth_multiplier ** (self.ceiling_rate() / inflight))

    def pause(self) -> None:
        """The engines are about to pause for a weight update: freeze growth, trims and cuts. Requests arriving
        meanwhile wait in the engines' queues, which say nothing about load."""
        self.paused = True

    def resume(self) -> None:
        """The engines resumed: skip the queue persistence window, which holds the backlog of the pause."""
        self.paused = False
        self.queue_overload_polls = 0
        self.prev_saturated = False
        self.resume_skip = QUEUE_PERSISTENCE_POLLS

    @property
    def recompute_control(self) -> bool:
        """Whether the measured recompute ratio drives growth and trims (``config.recompute`` set and measured)."""
        return self.config.recompute is not None and self.recompute_ratio is not None

    @property
    def waste(self) -> float | None:
        """Share of prefill work done twice: the recompute ratio plus the expired-lease rate. None until the
        recompute ratio is measured."""
        if self.recompute_ratio is None:
            return None
        return self.recompute_ratio + self.expired_lease_rate

    @property
    def engines_busy(self) -> bool:
        """Whether a queue showed up in ``BUSY_QUEUE_FRACTION`` of the window's polls: only then does
        recompute compete with other requests for prefill compute."""
        return bool(self.queued_polls) and sum(self.queued_polls) >= BUSY_QUEUE_FRACTION * len(self.queued_polls)

    def ceiling_rate(self) -> float:
        """Fraction of the full growth rate near the last preemption ceiling: ``CEILING_PROBE`` at or above it,
        rising linearly to 1 at half of it and below."""
        if self.preemption_ceiling is None:
            return 1.0
        distance = 1.0 - self.cap / self.preemption_ceiling
        return min(1.0, max(CEILING_PROBE, distance / 0.5))

    def _grow(self, inflight: int, factor: float) -> None:
        if (
            self.target is None
            and not self.paused
            and self.can_grow
            and time.monotonic() < self.can_grow_until
            and inflight >= BINDING_FRACTION * self.max_inflight
        ):
            self.cap = self.clamp(self.cap * factor)
            self.apply_limit(int(self.cap), reason=None)

    def observe(self, samples: list[EngineLoadSample]) -> None:
        """Per-poll engine load push from the metrics collector: classify the
        signal, gate growth, and apply trims and cuts."""
        if self.paused:
            return
        self.recompute_polls.append((self.pending_prefix, self.pending_recompute))
        self.pending_prefix = self.pending_recompute = 0
        self.recompute_ratio = recompute_ratio(self.recompute_polls)
        # Recompute-aware control while the ratio is measured
        recompute = self.config.recompute if self.recompute_ratio is not None else None
        if not samples:
            return
        for sample in samples:
            # The running set lives on decode (or colocated) engines; prefill engines only pass it through
            if sample.kv_capacity_tokens and sample.role != "prefill":
                self.capacity_by_engine[sample.engine_id] = sample.kv_capacity_tokens
            if sample.max_model_len:
                self.engine_max_len = max(self.engine_max_len or 0, sample.max_model_len)

        loads = role_loads(samples)
        self.load_polls.append(loads)
        self.saturation_by_role = {role: load.saturation for role, load in loads.items()}
        self.saturation = max(self.saturation_by_role.values())
        # Preemption and the KV gauge concern the running set, which lives on decode (or colocated) engines. A
        # preempted prefill re-queues, which prefill saturation sees; prefill KV holds transfers awaiting decode.
        running_set = [role for role in loads if role != "prefill"]
        self.preemption_rate = max(
            (window_rate(self.load_polls, role, "preemptions") for role in running_set), default=0.0
        )
        self.expired_lease_rate = max(window_rate(self.load_polls, role, "expired") for role in loads)
        self.kv_usage = max((loads[role].kv_usage for role in running_set), default=0.0)
        preempted = self.preemption_rate > PREEMPTION_RATE_CLIFF
        settled = all(loads[role].preemptions == 0 for role in running_set)

        # After a weight update the queue is the pause's backlog, not load
        queue_signals = self.resume_skip == 0
        self.resume_skip = max(0, self.resume_skip - 1)
        if queue_signals and self.saturation > QUEUE_RATIO:
            self.queue_overload_polls += 1
        else:
            self.queue_overload_polls = 0
        queue_overload = self.queue_overload_polls >= QUEUE_PERSISTENCE_POLLS
        saturated = self.saturation > (QUEUE_TOLERANCE if recompute is not None else 0.0)
        if queue_signals:
            self.queued_polls.append(self.saturation > 0)

        worst: Signal = "hard" if preempted or queue_overload else "clear"
        if worst == "clear":
            if recompute is not None:
                braking = self.engines_busy and self.waste >= recompute.grow_below
            else:
                braking = self.kv_usage > KV_USAGE_SOFT_CAP
            if braking or (saturated and self.prev_saturated):
                worst = "soft"
        self.prev_saturated = saturated and queue_signals
        self.signal = worst

        inflight = self.get_inflight() if self.get_inflight is not None else 0
        # Release the drain only once the engines have settled too: cuts
        # cancel episodes, so dispatcher inflight drops below the cap almost
        # immediately while the engines are still churning through their own
        # backlog — preemptions from that stale churn must not trigger
        # follow-up cuts (observed: an escalated-cut cascade 1024 -> 8, one
        # halving per poll).
        if self.draining and inflight <= self.max_inflight and settled and not queue_overload:
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
        self.recompute_trim_wait = max(0, self.recompute_trim_wait - 1)
        self.can_grow = worst == "clear" and not saturated and not self.draining
        if recompute is not None:
            headroom = 1.0 - self.waste / recompute.grow_below if self.engines_busy else 1.0
        else:
            headroom = 1.0 - self.kv_usage / KV_USAGE_SOFT_CAP
        self.growth_multiplier = 1.0 + (TURNOVER_GROWTH_MAX - 1.0) * max(0.0, headroom)
        self.call_growth_multiplier = 1.0 + (CALL_GROWTH_MAX - 1.0) * max(0.0, headroom)
        self.can_grow_until = time.monotonic() + GROWTH_GATE_TTL_S
        if self.target is not None:
            return

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
            # Preemptions while draining are the stale churn of the cut: measure afresh after it
            self.load_polls.clear()
            return

        if preempted and not queue_overload and recompute is not None:
            # Preemption only recomputes the preempted requests' prefill; cancelling episodes would
            # waste their earlier turns. Lower the cap, remember where, and let the pool drain.
            if self.recompute_trim_wait == 0 and 0 < inflight <= self.max_inflight:
                self.preemption_ceiling = self.cap
                self.resize_down(
                    int(self.clamp(inflight * PREEMPTION_CUT_FRACTION)),
                    inflight,
                    reason=f"preemption rate {self.preemption_rate:.3f}",
                    cancel=False,
                )
                self.recompute_trim_wait = LOAD_WINDOW_POLLS
                self.load_polls.clear()
            return

        if preempted or queue_overload:
            fraction = ESCALATED_CUT_FRACTION if self.escalated else None
            if queue_overload:
                self.queue_overload_polls = 0
                target = int(self.clamp(inflight * (fraction or QUEUE_CUT_FRACTION)))
                reason = f"queue overload (saturation {self.saturation:.2f})"
            else:
                target = int(self.clamp(inflight * (fraction or PREEMPTION_CUT_FRACTION)))
                reason = f"preemption rate {self.preemption_rate:.3f}"
            self.resize_down(target, inflight, reason=reason)
            self.draining = True
            self.escalated = True
            self.load_polls.clear()
            return

        if recompute is None and self.kv_usage > KV_USAGE_SOFT_CAP and inflight > 0 and self.trim_cooldown == 0:
            hard = self.kv_usage > KV_USAGE_HARD_CAP
            target = int(self.clamp(inflight * KV_USAGE_TARGET / self.kv_usage))
            self.resize_down(
                target,
                inflight,
                reason=f"kv headroom (usage {self.kv_usage:.2f}, {'hard' if hard else 'soft'} trim)",
                cancel=hard,
            )
            self.trim_cooldown = KV_TRIM_COOLDOWN_POLLS
        elif (
            recompute is not None
            and self.engines_busy
            and (self.waste > recompute.trim_above or self.expired_lease_rate > EXPIRED_LEASE_TRIM)
            and 0 < inflight <= self.max_inflight
            and self.recompute_trim_wait == 0
        ):
            # Recompute costs throughput, not a cliff: lower the cap and let
            # completions drain the pool. Re-trim only once the pool has drained
            # to the cap and a full window has been measured since.
            self.resize_down(
                int(self.clamp(inflight * RECOMPUTE_TRIM_FRACTION)),
                inflight,
                reason=f"waste {self.waste:.2f} (expired leases {self.expired_lease_rate:.3f})",
                cancel=False,
            )
            self.recompute_trim_wait = LOAD_WINDOW_POLLS

    def set_target(self, target: int | None) -> None:
        """Pin the cap at ``target``, outside the configured band too, or hand it back to the
        adaptive loop with None. A lower cap cancels nothing: admission waits for the pool
        to drain below it."""
        self.target = target
        self.cap = float(target) if target is not None else self.clamp(self.cap)
        self.apply_limit(int(self.cap), reason="manual target" if target is not None else "adaptive")

    def reconfigure(self, update: ConfigUpdate) -> None:
        """Move the band and clamp the cap into it. A manual target stays pinned, and a lower
        cap cancels nothing: admission waits for the pool to drain below it. ``initial_inflight``
        only seeds the startup cap, so a new value (e.g. to stay below a lowered
        ``max_inflight``) changes nothing."""
        self.config = update.config.concurrency
        self.floor = self.config.min_inflight
        if self.target is None:
            self.cap = self.clamp(self.cap)
            self.apply_limit(int(self.cap), reason="config update")

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
            rho = f" recompute={self.recompute_ratio:.2f}" if self.recompute_ratio is not None else ""
            get_logger().info(
                f"{verb} concurrency {self.max_inflight} -> {n_max} ({reason}) - "
                f"turnover={self.turnover:.1f} signal={self.signal} saturation={self.saturation:.2f}{rho}"
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
        gauges = {
            "concurrency/max_inflight": float(self.max_inflight),
            "concurrency/turnover": self.turnover,
            "concurrency/capacity": float(self.capacity or 0),
            "concurrency/signal": float(SEVERITY[self.signal]),
            "concurrency/growth_multiplier": self.growth_multiplier,
            "concurrency/paused": float(self.paused),
        }
        if self.load_polls:
            gauges["concurrency/saturation"] = self.saturation
            for role, saturation in self.saturation_by_role.items():
                gauges[f"concurrency/saturation/{role}"] = saturation
            gauges["concurrency/preemption_rate"] = self.preemption_rate
        if self.preemption_ceiling is not None:
            gauges["concurrency/preemption_ceiling"] = self.preemption_ceiling
        if self.queued_polls:
            gauges["concurrency/engines_busy"] = float(self.engines_busy)
        if self.recompute_ratio is not None:
            gauges["concurrency/recompute_ratio"] = self.recompute_ratio
            gauges["concurrency/waste"] = self.waste
        return gauges


def role_loads(samples: list[EngineLoadSample]) -> dict[str, RoleLoad]:
    """Pool one poll's engines per role. Queued counts every waiting request: one waiting on a KV load (offload
    tier, Mooncake, or a P/D transfer) waits just like one waiting for capacity."""
    by_role: dict[str, list[EngineLoadSample]] = {}
    for sample in samples:
        by_role.setdefault(sample.role or COLOCATED, []).append(sample)
    return {
        role: RoleLoad(
            running=sum(sample.running for sample in group),
            queued=sum(sample.waiting for sample in group),
            preemptions=sum(sample.preemptions_delta for sample in group),
            finished=sum(sample.finished_delta for sample in group),
            expired=sum(sample.expired_kv_delta for sample in group),
            kv_usage=quantile(
                [sample.kv_usage if sample.host_kv_usage is None else sample.host_kv_usage for sample in group],
                KV_USAGE_QUANTILE,
            ),
        )
        for role, group in by_role.items()
    }


def window_rate(polls: deque[dict[str, RoleLoad]], role: str, field: Literal["preemptions", "expired"]) -> float:
    """Events per finished request of one role over the window. With fewer than one finished request, any event
    counts in full."""
    events = sum(getattr(poll[role], field) for poll in polls if role in poll)
    finished = sum(poll[role].finished for poll in polls if role in poll)
    return events / max(finished, 1)


def quantile(values: list[float], q: float) -> float:
    """Linearly interpolated quantile (numpy's default)."""
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def recompute_ratio(polls: deque[tuple[int, int]]) -> float | None:
    """Recompute tokens over prefix tokens across the window; None until the window
    holds enough prefix tokens to be measured."""
    prefix = sum(p for p, _ in polls)
    if prefix < RECOMPUTE_MIN_PREFIX_TOKENS:
        return None
    return sum(r for _, r in polls) / prefix
