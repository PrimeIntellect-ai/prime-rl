"""ConcurrencyController: adaptive cap on in-flight units of inference work.

A unit is whatever the dispatcher admits against one permit — the controller
treats it as a black box. There is no cost model: the controller measures load
off the engines and reacts at the pipeline's own pace.

The main signal is the **queued share**: requests waiting in the engines' queues
(for capacity, or deferred on a KV load) over the units in flight, averaged over
polls. By Little's law it is the share of the units' time spent parked in an
engine queue, so it means the same in every topology: in P/D the prefill queue
and decode's pending KV pulls add up, a slow KV load counts by its duration, and
units idle between turns (tool calls) count in the denominator as the capacity
they are. Below the knee extra units run at once; past it they wait, adding
latency (staleness) and no throughput. Per poll the controller also reads:

- **KV usage**: the p90 over decode (or colocated) engines of the GPU-KV gauge
  (the HiSparse host pool gauge where exported). It counts running requests
  only, so it is used only until the recompute ratio — the share of each call's
  previously computed prefix the engines prefilled again — is measured: that
  needs cached-token counts, which single-call units never produce.
- **preemption rate**: preemptions per finished request over the window, on
  decode (or colocated) engines.
- **throughput**: generated tokens per poll and the engines' latency per call,
  over a few minutes, against the best window measured so far.

and acts:

- **Grow** the cap per pool turnover (each completed unit advances it by
  ``1/inflight``) while the cap binds admission, by up to ``GROWTH_MAX``: a
  unit's load shows in full only by the time it completes (its context grows
  call by call). The rate tapers linearly to zero as the queued share approaches
  ``QUEUE_TARGET`` (and, without a recompute ratio, as KV usage approaches
  ``KV_USAGE_SOFT_CAP``), so steps shrink near the knee.
- **Trim** without cancelling: when the queued share stays above
  ``QUEUE_TRIM``, drop the units queued beyond the target; on a preemption rate
  above ``PREEMPTION_RATE_CLIFF``; and, without a recompute ratio, on KV usage
  above ``KV_USAGE_SOFT_CAP``. Admission waits for completions to drain the
  pool, and the next trim waits for the pool to have drained and turned over.
- **Cut**, cancelling the youngest units, only at cliffs: throughput down by
  ``THROUGHPUT_CLIFF_DROP`` with as many units in flight and the engines'
  latency stretched by ``LATENCY_CLIFF_STRETCH`` (cache thrash as the pool
  matured, a collapsed frontend), a queue still above ``QUEUE_OVERLOAD`` once a
  queue trim has drained, or (without a recompute ratio) KV usage above
  ``KV_USAGE_HARD_CAP``. After a cliff or a preemption trim, growth slows near
  that cap.

While the engines are paused for a weight update the controller freezes, and
the queue that builds up behind the pause is ignored for a window after it.

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
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from prime_rl.configs.orchestrator import ConcurrencyConfig
from prime_rl.control.types import ConfigUpdate, Reconfigurable
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import format_num

GROWTH_MAX = 3.0
"""Cap growth per pool turnover with empty queues."""

TRIM_SPACING_TURNOVERS = 1.0
"""Pool turnovers between trims."""

BINDING_FRACTION = 0.9
"""The cap counts as binding when inflight reaches this fraction of it."""

QUEUE_TARGET = 0.1
"""Queued share at which growth stops: the knee."""

QUEUE_TRIM = 0.15
"""Queued share (over the window) above which the cap is trimmed back to the target."""

QUEUE_OVERLOAD = 0.5
"""Queued share that, still standing once a trim has drained, cancels the excess."""

GROWTH_WINDOW_POLLS = 4
"""Polls the queued share is averaged over for growth: short, so growth reacts within a turnover."""

LOAD_WINDOW_POLLS = 12
"""Polls the queued share for trims, the recompute ratio, and the preemption and expired-lease rates are measured
over."""

RESUME_SKIP_POLLS = 6
"""Polls after a weight update whose queue is the pause's backlog."""

KV_USAGE_SOFT_CAP = 0.8
"""Before the recompute ratio is measured: growth tapers to zero at this usage, and above it the cap is trimmed
without cancelling."""

KV_USAGE_HARD_CAP = 0.9
"""Before the recompute ratio is measured: above this usage in-place context growth outpaces completions, so the
trim also cancels the excess."""

KV_USAGE_TARGET = 0.7
"""A KV trim resizes to inflight * target / usage."""

KV_USAGE_QUANTILE = 0.9
"""KV usage is this quantile over a role's engines: one hot engine under sticky routing is a placement problem, not
pool pressure."""

PREEMPTION_RATE_CLIFF = 0.05
"""A role's preemptions over its finished requests in the window above which the running set no longer fits."""

PREEMPTION_CUT_FRACTION = 0.8
"""A preemption trim or cut targets this fraction of the in-flight pool."""

CEILING_PROBE = 0.1
"""Near the cap of the last preemption trim or cliff cut, growth slows to this fraction of its rate, rising back to full as the cap
moves away below it (CUBIC-style), and stays at it above it to keep probing."""

THROUGHPUT_WINDOW_POLLS = 36
"""Polls throughput and latency are measured over for the cliff check: long enough to average out a small pool.
A cliff must also hold for this many polls: units admitted together swing throughput in waves."""

THROUGHPUT_CLIFF_DROP = 0.4
"""A cliff: generated tokens per poll this far below the best window, with as many units in flight."""

LATENCY_CLIFF_STRETCH = 2.0
"""A cliff also stretches the engines' request latency by this much over the best window's: a slower environment
lowers throughput too, but not engine latency."""

CLIFF_CUT_MIN = 0.5
"""A cliff cut keeps at least this fraction of the units in flight (it cuts in proportion to the throughput lost)."""

THROUGHPUT_BEST_HALF_LIFE_POLLS = 360
"""The best throughput decays with this half-life, so slow workload drift (e.g. longer responses) does not read as
a cliff."""

RECOMPUTE_MIN_PREFIX_TOKENS = 100_000
"""Prefix tokens the window needs before its recompute ratio counts as measured."""

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
    generation_tokens_delta: int = 0
    # Summed end-to-end latency (arrival at the engine to finish) of the requests in ``finished_delta``
    latency_delta: float = 0.0


@dataclass(frozen=True)
class RoleLoad:
    """One poll's engines of one role, pooled."""

    running: int
    queued: int
    preemptions: int
    finished: int
    expired: int
    kv_usage: float
    generated: int = 0
    latency: float = 0.0


@dataclass(frozen=True)
class Poll:
    """One poll: the pooled load per role and the units in flight."""

    roles: dict[str, RoleLoad]
    inflight: int

    @property
    def queued(self) -> int:
        return sum(load.queued for load in self.roles.values())

    @property
    def generated(self) -> int:
        return sum(load.generated for role, load in self.roles.items() if role != "prefill")


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
        # Turnover since the last poll that grows the cap, and the turnover at the last trim
        self.pending_turnover = 0.0
        self.trim_turnover = -math.inf
        self.signal: Signal = "clear"
        self.growth_multiplier = 1.0
        # After a trim, no further trim until inflight has drained below the new cap and a window was measured
        self.draining = False
        self.trim_wait = 0
        # The last trim was a queue trim and no window since has measured the queue below the trim threshold
        self.queue_trimmed = False
        # Engines paused for a weight update, and the polls after resume whose queue is the pause's backlog
        self.paused = False
        self.resume_skip = 0
        self.polls: deque[Poll] = deque(maxlen=LOAD_WINDOW_POLLS)
        self.queued_share = 0.0
        self.queued_share_window = 0.0
        self.queued_share_by_role: dict[str, float] = {}
        self.preemption_rate = 0.0
        self.expired_lease_rate = 0.0
        self.kv_usage = 0.0
        # Per-poll (prefix, recompute) call tokens, and the ratio over the window
        self.recompute_polls: deque[tuple[int, int]] = deque(maxlen=LOAD_WINDOW_POLLS)
        self.pending_prefix = 0
        self.pending_recompute = 0
        self.recompute_ratio: float | None = None
        # The cap at the last preemption trim or cliff cut: growth slows near it
        self.ceiling: float | None = None
        # Polls of the throughput window (not cleared by trims), and the best window: generated tokens per poll,
        # mean units in flight, and engine latency per call
        self.rate_polls: deque[Poll] = deque(maxlen=THROUGHPUT_WINDOW_POLLS)
        self.throughput = 0.0
        self.latency = 0.0
        self.best_throughput = 0.0
        self.best_inflight = 0.0
        self.best_latency = 0.0
        self.cliff_turnover = 0.0
        self.cliff_polls = 0

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
        """One completed unit (from the dispatcher). Advances the turnover clock by ``1/inflight``, which paces
        growth: a unit's load has fully shown on the engines only once it completes (its context grows call by
        call), so the cap moves at most ``GROWTH_MAX`` per turnover of the pool."""
        # The dispatcher releases the permit before this hook fires, so the completing unit is already off the count
        inflight = (self.get_inflight() if self.get_inflight is not None else 0) + 1
        self.turnover += 1 / inflight
        # Error markers consumed no engine capacity: growing on them floods the engines once the environment recovers
        if tokens > 0:
            self.pending_turnover += 1 / inflight

    def record_calls(self, prefix: int, recompute: int) -> None:
        """Prefix and recompute tokens of a finished model call (from the dispatcher's
        live deltas), folded into the recompute ratio at the next poll."""
        self.pending_prefix += prefix
        self.pending_recompute += recompute

    def pause(self) -> None:
        """The engines are about to pause for a weight update: freeze growth, trims and cuts. Requests arriving
        meanwhile wait in the engines' queues, which say nothing about load."""
        self.paused = True

    def resume(self) -> None:
        """The engines resumed: skip the polls whose queue holds the backlog of the pause."""
        self.paused = False
        self.resume_skip = RESUME_SKIP_POLLS

    def ceiling_rate(self) -> float:
        """Fraction of the full growth rate near the ceiling: ``CEILING_PROBE`` at or above it,
        rising linearly to 1 at half of it and below."""
        if self.ceiling is None:
            return 1.0
        distance = 1.0 - self.cap / self.ceiling
        return min(1.0, max(CEILING_PROBE, distance / 0.5))

    def observe(self, samples: list[EngineLoadSample]) -> None:
        """Per-poll engine load push from the metrics collector: measure, then grow, trim or cut."""
        if self.paused:
            return
        self.recompute_polls.append((self.pending_prefix, self.pending_recompute))
        self.pending_prefix = self.pending_recompute = 0
        self.recompute_ratio = recompute_ratio(self.recompute_polls)
        if not samples:
            return
        for sample in samples:
            # The running set lives on decode (or colocated) engines; prefill engines only pass it through
            if sample.kv_capacity_tokens and sample.role != "prefill":
                self.capacity_by_engine[sample.engine_id] = sample.kv_capacity_tokens
            if sample.max_model_len:
                self.engine_max_len = max(self.engine_max_len or 0, sample.max_model_len)

        inflight = self.get_inflight() if self.get_inflight is not None else 0
        loads = role_loads(samples)
        # After a weight update the queue is the pause's backlog, not load
        backlog = self.resume_skip > 0
        self.resume_skip = max(0, self.resume_skip - 1)
        if not backlog:
            self.polls.append(Poll(loads, inflight))
        recent = list(self.polls)[-GROWTH_WINDOW_POLLS:]
        self.queued_share = queued_share(recent)
        self.queued_share_window = queued_share(self.polls)
        self.queued_share_by_role = {
            role: sum(poll.roles[role].queued for poll in self.polls if role in poll.roles)
            / max(sum(poll.inflight for poll in self.polls), 1)
            for role in loads
        }
        # Preemption and the KV gauge concern the running set, which lives on decode (or colocated) engines. A
        # preempted prefill re-queues, which the queued share sees; prefill KV holds transfers awaiting decode.
        running_set = [role for role in loads if role != "prefill"]
        self.preemption_rate = max((window_rate(self.polls, role, "preemptions") for role in running_set), default=0.0)
        self.expired_lease_rate = max((window_rate(self.polls, role, "expired") for role in loads), default=0.0)
        self.kv_usage = max((loads[role].kv_usage for role in running_set), default=0.0)
        measured = self.recompute_ratio is not None
        preempted = self.preemption_rate > PREEMPTION_RATE_CLIFF
        cliff = False
        if not backlog:
            self.rate_polls.append(self.polls[-1])
            self.best_throughput *= 0.5 ** (1 / THROUGHPUT_BEST_HALF_LIFE_POLLS)
        if len(self.rate_polls) == THROUGHPUT_WINDOW_POLLS:
            self.throughput = sum(poll.generated for poll in self.rate_polls) / THROUGHPUT_WINDOW_POLLS
            self.latency = call_latency(self.rate_polls)
            window_inflight = sum(poll.inflight for poll in self.rate_polls) / THROUGHPUT_WINDOW_POLLS
            # Only a pool that has turned over since the start or the last cliff carries its steady load: a younger
            # one runs shorter contexts, so its throughput overstates what the same units sustain
            if self.throughput > self.best_throughput and self.turnover - self.cliff_turnover >= 1.0:
                self.best_throughput, self.best_inflight = self.throughput, window_inflight
                self.best_latency = self.latency
            below = (
                self.throughput < (1 - THROUGHPUT_CLIFF_DROP) * self.best_throughput
                and window_inflight >= 0.9 * self.best_inflight
                and self.latency > LATENCY_CLIFF_STRETCH * self.best_latency
            )
            self.cliff_polls = self.cliff_polls + 1 if below else 0
            cliff = self.cliff_polls >= THROUGHPUT_WINDOW_POLLS

        if self.draining and inflight <= self.max_inflight:
            self.draining = False
            self.trim_wait = LOAD_WINDOW_POLLS
        if not self.draining:
            self.trim_wait = max(0, self.trim_wait - 1)

        headroom = 1.0 - self.queued_share / QUEUE_TARGET
        if not measured:
            headroom = min(headroom, 1.0 - self.kv_usage / KV_USAGE_SOFT_CAP)
        headroom = max(0.0, min(1.0, headroom))
        self.growth_multiplier = 1.0 + (GROWTH_MAX - 1.0) * headroom * self.ceiling_rate()
        if preempted or (not measured and self.kv_usage > KV_USAGE_HARD_CAP):
            self.signal = "hard"
        elif headroom <= 0.0 or self.queued_share_window > QUEUE_TRIM:
            self.signal = "soft"
        else:
            self.signal = "clear"
        growth, self.pending_turnover = self.pending_turnover, 0.0

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

        if inflight <= 0:
            return
        # A trim acts once the pool has drained to it and turned over: until then the units in flight are the ones
        # that overloaded the engines, matured further while admission waited
        settled = (
            not self.draining and self.trim_wait == 0 and self.turnover - self.trim_turnover >= TRIM_SPACING_TURNOVERS
        )

        if cliff and not self.draining:
            # As many units finish less work, each slower: past a cliff (cache thrash as the pool matured, a
            # collapsed frontend), which completions alone drain too slowly to escape
            target = int(self.clamp(inflight * max(CLIFF_CUT_MIN, self.throughput / self.best_throughput)))
            # The cliff lies somewhere below the cap, where the pool matured into it: approach it again slowly
            self.ceiling = float(target)
            self.trim(
                target,
                inflight,
                cancel=True,
                reason=f"throughput cliff ({self.throughput:.0f} < {self.best_throughput:.0f} tokens/poll, "
                f"latency {self.latency:.1f}s > {self.best_latency:.1f}s)",
            )
            self.rate_polls.clear()
            self.best_throughput = 0.0
            self.cliff_turnover = self.turnover
            self.cliff_polls = 0
            return

        if preempted and settled:
            self.ceiling = self.cap
            self.trim(
                int(self.clamp(inflight * PREEMPTION_CUT_FRACTION)),
                inflight,
                reason=f"preemption rate {self.preemption_rate:.3f}",
                cancel=False,
            )
            return

        if not measured and self.kv_usage > KV_USAGE_SOFT_CAP and settled:
            hard = self.kv_usage > KV_USAGE_HARD_CAP
            self.trim(
                int(self.clamp(inflight * KV_USAGE_TARGET / self.kv_usage)),
                inflight,
                reason=f"kv headroom (usage {self.kv_usage:.2f}, {'hard' if hard else 'soft'} trim)",
                cancel=hard,
            )
            return

        if settled and len(self.polls) == LOAD_WINDOW_POLLS:
            share = self.queued_share_window
            if share <= QUEUE_TRIM:
                self.queue_trimmed = False
            else:
                # The surplus units are the queued share above the target
                target = int(self.clamp(inflight * (1.0 - share + QUEUE_TARGET)))
                # A queue still overloaded once the last queue trim drained and turned over: the engines collapsed
                overload = self.queue_trimmed and share > QUEUE_OVERLOAD
                self.trim(
                    target, inflight, reason=f"{'queue overload' if overload else 'queue'} {share:.2f}", cancel=overload
                )
                self.queue_trimmed = True
                return

        if not self.draining and inflight >= BINDING_FRACTION * self.max_inflight and headroom > 0.0:
            self.cap = self.clamp(self.cap * self.growth_multiplier**growth)
            self.apply_limit(int(self.cap), reason=None)

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

    def trim(self, target: int, inflight: int, *, reason: str, cancel: bool) -> None:
        """Lower the cap (never raise) and, with ``cancel``, shed the in-flight
        excess; without it, admission stays blocked until the pool drains below
        the new cap on its own. Either way no further trim until it has."""
        target = min(target, self.max_inflight)
        self.cap = float(target)
        self.apply_limit(target, reason=reason)
        if cancel and self.on_overload is not None and inflight > target:
            self.on_overload(inflight - target)
        self.draining = True
        self.trim_turnover = self.turnover
        self.polls.clear()

    def apply_limit(self, n_max: int, *, reason: str | None) -> None:
        if n_max == self.max_inflight:
            return
        if reason is not None:
            verb = "Increased" if n_max > self.max_inflight else "Decreased"
            rho = f" recompute={self.recompute_ratio:.2f}" if self.recompute_ratio is not None else ""
            get_logger().info(
                f"{verb} concurrency {self.max_inflight} -> {n_max} ({reason}) - "
                f"turnover={self.turnover:.1f} signal={self.signal} queued={self.queued_share:.2f}{rho}"
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
        if self.polls:
            gauges["concurrency/queued_share"] = self.queued_share_window
            for role, share in self.queued_share_by_role.items():
                gauges[f"concurrency/queued_share/{role}"] = share
            gauges["concurrency/preemption_rate"] = self.preemption_rate
            gauges["concurrency/expired_lease_rate"] = self.expired_lease_rate
            gauges["concurrency/kv_usage"] = self.kv_usage
        if self.best_throughput > 0:
            gauges["concurrency/throughput"] = self.throughput
            gauges["concurrency/best_throughput"] = self.best_throughput
            gauges["concurrency/latency"] = self.latency
        if self.ceiling is not None:
            gauges["concurrency/ceiling"] = self.ceiling
        if self.recompute_ratio is not None:
            gauges["concurrency/recompute_ratio"] = self.recompute_ratio
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
            generated=sum(sample.generation_tokens_delta for sample in group),
            latency=sum(sample.latency_delta for sample in group),
            kv_usage=quantile(
                [sample.kv_usage if sample.host_kv_usage is None else sample.host_kv_usage for sample in group],
                KV_USAGE_QUANTILE,
            ),
        )
        for role, group in by_role.items()
    }


def queued_share(polls) -> float:
    """Waiting requests over units in flight, summed over the polls: by Little's law the share of the units' time
    spent in an engine queue."""
    return sum(poll.queued for poll in polls) / max(sum(poll.inflight for poll in polls), 1)


def call_latency(polls) -> float:
    """Engine latency of one call: the mean request latency of each role over the polls, summed over roles (in P/D
    a call is a prefill request then a decode request)."""
    roles = {role for poll in polls for role in poll.roles}
    total = 0.0
    for role in roles:
        finished = sum(poll.roles[role].finished for poll in polls if role in poll.roles)
        total += sum(poll.roles[role].latency for poll in polls if role in poll.roles) / max(finished, 1)
    return total


def window_rate(polls: deque[Poll], role: str, field: Literal["preemptions", "expired"]) -> float:
    """Events per finished request of one role over the window. With fewer than one finished request, any event
    counts in full."""
    events = sum(getattr(poll.roles[role], field) for poll in polls if role in poll.roles)
    finished = sum(poll.roles[role].finished for poll in polls if role in poll.roles)
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
