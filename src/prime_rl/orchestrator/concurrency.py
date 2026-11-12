"""ConcurrencyController: adaptive cap on in-flight units of inference work.

A unit is whatever the dispatcher admits against one permit; the controller treats it as a black box and models no
cost. It measures load off the engines and moves at the pipeline's own pace.

The cap is sized by **marginal gain**: a unit is worth adding only while it buys throughput. Past the knee extra
units add latency (staleness) and no throughput, wherever the bottleneck sits: in the engines, in their KV cache, or
outside them (an API server, the environment). The controller compares the engines' generated tokens per second
between windows long enough for the pool to carry its steady load by their **elasticity**, the relative change in
throughput per relative change in units in flight: a move up is kept when it reaches ``ELASTICITY``, a move down
when it does not.

The fast signal is the **queued share**: requests waiting in the engines' queues (for capacity, or deferred on a KV
load) over the units in flight, averaged over polls. By Little's law it is the share of the units' time spent parked
in an engine queue, so it means the same in every topology: in P/D the prefill queue and decode's pending KV pulls
add up, a slow KV load or a preempted request counts by how long it waits, and units idle between turns (tool calls)
count as the capacity they are.

Time is counted in **turnovers** of the pool: each completed unit advances it by ``1/inflight``. A unit's load shows
in full only once it completes (its context grows call by call), so the controller moves at that pace. Three rules:

- **Climb**: from the start the cap grows by up to ``GROWTH_MAX`` per turnover while it binds, tapered to zero as the
  queued share reaches ``QUEUE_TARGET``. Model calls pace it too: from the contexts of the units' calls the
  controller projects the pool's steady load, and grows the cap by up to ``CALL_GROWTH_MAX`` per call turnover while
  the pool carries ``MATURITY_MIN`` of it. The first window whose growth did not pay ends the fast climb at the two
  windows' geometric midpoint. From there the cap moves in steps from a measured level: a step is kept when it was
  worth its units and the next one grows by how clearly it paid; a rejected step returns to the level and shrinks,
  down to ``STEP_MIN``.
- **Queue**: a queued share above ``QUEUE_TRIM`` trims the units queued beyond the target while the cap binds,
  without cancelling. A queue still above ``QUEUE_CANCEL`` once a trim has drained cancels the excess.
- **Throughput drop**: two windows in a row ``THROUGHPUT_DROP`` below the best window's throughput with as many units
  in flight (a KV thrash, a collapsed frontend) trim the cap in proportion to the throughput lost, without
  cancelling. A third, with the engines the bottleneck (their latency stretched by ``LATENCY_STRETCH``, or a queue
  above ``QUEUE_CANCEL``), cancels the excess: past a cliff, completions drain too slowly.

The queue serves the elasticity: while the last measurement showed growth still pays, the queue neither tapers growth
nor trims (requests waiting on KV loads queue while extra units still buy throughput); only its cancelling tier
applies. While the engines are paused for a weight update the controller freezes, and it ignores the queue the pause
leaves behind for a few polls.

The cap starts at ``initial_inflight`` (else the pessimistic ``KV capacity / max_model_len``) and stays in
``[min_inflight, max_inflight]`` unless a manual target pins it.

The controller is a pure state machine: the metrics collector pushes ``observe(samples)`` every poll, the dispatcher
reports ``record_episode``/``record_call`` and ``pause``/``resume`` around weight updates, and consumes the cap
through the hooks bound in :meth:`bind`.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from prime_rl.configs.orchestrator import ConcurrencyConfig
from prime_rl.control.types import ConfigUpdate, Reconfigurable
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import format_num

ELASTICITY = 0.3
"""A move is worth its units when throughput moves by at least this share of the relative change in units in
flight (d ln throughput / d ln inflight): a move up is kept at or above it, a move down below it. It prices
staleness: the cap settles where one more unit buys 30 % of its proportional share of throughput."""

GROWTH_MAX = 3.0
"""Fast climb: cap growth per pool turnover with empty queues."""

CALL_GROWTH_MAX = 1.25
"""Fast climb: cap growth per call turnover (every unit in flight made one model call) with empty queues."""

MATURITY_MIN = 0.8
"""Fast climb: model calls grow the cap only while the pool's load is at least this share of its projected steady
load."""

WINDOW_CALLS = 4.0
"""Fast climb: a window of a mature pool lasts at least this call turnover (instead of ``DWELL_TURNOVERS``). Such
windows are short, so they are compared once the pool grew by ``STEP_MAX`` (``STEP_MIN`` otherwise)."""

PROJECTION_UNITS = 512
"""The load projection counts about this many recent units: its counts halve whenever more units started."""

STEP_MAX = 1.5
"""Largest step. After a kept step the next one scales with how far the measured elasticity cleared
``ELASTICITY``: the full step where throughput grew in proportion (or, stepping down, held), the smallest at the
threshold. A step down is at most ``STEP_MAX ** 0.7``: one below the knee costs throughput for a whole dwell."""

STEP_MIN = 1.15
"""Smallest step: a rejected step takes the square root of the next one, down to this. Smaller steps move throughput
less than a dwell's noise."""

DWELL_SECONDS = 180.0
"""Throughput is measured over windows of at least this long..."""

DWELL_TURNOVERS = 0.5
"""...and this much pool turnover, twice as much after a step up (its units show their load in full only as their
contexts grow call by call). A step up admits its units over this much turnover: a cohort admitted at once runs
short contexts together, then finishes together, and its waves bias the measurement."""

SETTLE_TURNOVERS = 0.1
"""Pool turnover after a lowered cap has drained before its level is measured."""

QUEUE_TARGET = 0.1
"""Queued share at which the cap stops growing: the extra units would wait in the engines."""

QUEUE_TRIM = 0.15
"""Queued share above which the cap is trimmed back to the target."""

QUEUE_CANCEL = 0.5
"""Queued share that, still standing once a trim has drained, cancels the excess."""

THROUGHPUT_DROP = 0.25
"""A window's throughput this far below the best window's, with as many units in flight, trims the cap."""

DROP_TRIM_MIN = 0.5
"""A throughput drop trims the pool in proportion to the throughput lost, keeping at least this share of it."""

LATENCY_STRETCH = 2.0
"""A throughput drop is the engines' (not a slower environment's) when it stretches their latency per call this much
over the best window's."""

BINDING_FRACTION = 0.9
"""The cap counts as binding when inflight reaches this fraction of it."""

LOAD_WINDOW_POLLS = 12
"""Polls the queued share is averaged over."""

RESUME_SKIP_POLLS = 6
"""Polls after a weight update whose queue is the pause's backlog."""

BEST_HALF_LIFE_POLLS = 360
"""The best window's throughput decays with this half-life, so slow workload drift does not read as a drop."""

COLOCATED = "colocated"
"""Role of engines that both prefill and decode (no P/D role)."""


@dataclass(frozen=True)
class EngineLoadSample:
    """Per-engine load facts for one ``/metrics`` poll. Raw values only: thresholds and verdicts live in the
    controller."""

    engine_id: str
    role: str | None
    kv_capacity_tokens: int | None
    max_model_len: int | None
    # All waiting requests: queued for capacity plus deferred (KV loads, grammar compilation)
    waiting: int
    finished_delta: int = 0
    generation_tokens_delta: int = 0
    # Summed end-to-end latency (arrival at the engine to finish) of the requests in ``finished_delta``
    latency_delta: float = 0.0
    # Seconds the engine spent executing model steps since the previous poll (None: not reported)
    busy_delta: float | None = None
    # What the engine's steps held since the previous poll (None: not reported): steps executed, and summed over its
    # scheduled steps, the sequences decoding, the context they read and the prefill tokens; and its max_num_seqs
    steps_delta: float | None = None
    decode_seqs_delta: float | None = None
    kv_read_delta: float | None = None
    prefill_delta: float | None = None
    max_num_seqs: int | None = None
    # The server whose data-parallel MoE engines step in lockstep with this one (None: it steps on its own)
    lockstep_group: str | None = None
    # Share of the engine's GPU KV cache in use at the poll (vLLM's ``kv_cache_usage_perc``)
    kv_usage: float | None = None


@dataclass(frozen=True)
class RoleLoad:
    """One poll's engines of one role, pooled."""

    queued: int
    finished: int = 0
    generated: int = 0
    latency: float = 0.0


@dataclass(frozen=True)
class Poll:
    """One poll: the pooled load per role, the units in flight, and the seconds since the previous poll."""

    roles: dict[str, RoleLoad]
    inflight: int
    seconds: float = 0.0

    @property
    def queued(self) -> int:
        return sum(load.queued for load in self.roles.values())

    @property
    def generated(self) -> int:
        return sum(load.generated for role, load in self.roles.items() if role != "prefill")


@dataclass
class Window:
    """Generated tokens per second, mean units in flight and engine latency per call over a span of polls. Poll
    intervals stretch under load, so throughput is per second of the poll loop's own clock."""

    start_turnover: float = 0.0
    # Fast-climb windows: the call clock at the start, and whether the load projection held growth back
    start_calls: float = 0.0
    held: bool = False
    tokens: float = 0.0
    seconds: float = 0.0
    unit_seconds: float = 0.0
    finished: dict[str, int] = field(default_factory=dict)
    latency_sum: dict[str, float] = field(default_factory=dict)

    def add(self, poll: Poll) -> None:
        # A poll without an interval (the first) cannot be timed
        if poll.seconds <= 0:
            return
        self.tokens += poll.generated
        self.seconds += poll.seconds
        self.unit_seconds += poll.inflight * poll.seconds
        for role, load in poll.roles.items():
            self.finished[role] = self.finished.get(role, 0) + load.finished
            self.latency_sum[role] = self.latency_sum.get(role, 0.0) + load.latency

    @property
    def timed(self) -> bool:
        """At least ``DWELL_SECONDS`` long, with units in flight."""
        return self.seconds >= DWELL_SECONDS and self.unit_seconds > 0

    def lasted(self, turnover: float, turnovers: float) -> bool:
        """Timed, and at least ``turnovers`` of pool turnover long."""
        return self.timed and turnover - self.start_turnover >= turnovers

    @property
    def throughput(self) -> float:
        return self.tokens / self.seconds

    @property
    def inflight(self) -> float:
        return self.unit_seconds / self.seconds

    @property
    def latency(self) -> float:
        """Engine latency of one call: each role's mean request latency, summed (in P/D a call is a prefill
        request then a decode request)."""
        return sum(self.latency_sum[role] / max(finished, 1) for role, finished in self.finished.items())


@dataclass
class Level:
    """A cap being measured. A step up first admits its units gradually from ``origin``, a lowered cap first
    drains; then ``window`` measures it."""

    cap: int
    origin: int | None = None
    origin_turnover: float = 0.0
    settle: float | None = None
    window: Window | None = None


@dataclass
class LoadProjection:
    """The pool's steady load, projected from its calls' contexts: a unit's context grows call by call, so a young
    pool carries less load than it will. In steady state the mean context per call weighs the mean context of each
    unit's n-th call by the share of units that make it, which a Kaplan-Meier estimate gives from the units that made
    call n and ended after it: units still running count to the calls they made, and the share still running at the
    last call number seen is assumed to grow at the mean rate to the engines' max context. Maturity is the load
    window's context per call over that projection."""

    # Per call number n of a unit: units that made call n, their summed context, and units that ended after n calls
    reached: dict[int, float] = field(default_factory=dict)
    context: dict[int, float] = field(default_factory=dict)
    ended: dict[int, float] = field(default_factory=dict)
    first_call: float = math.inf
    # The calls since the last poll, and per poll of the load window (context, calls)
    pending_context: int = 0
    pending_calls: int = 0
    polls: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=LOAD_WINDOW_POLLS))
    # The projected steady context per call, a first call's, and the load window's share of it
    steady_context: float = 1.0
    young_context: float = 0.0
    maturity: float | None = None

    def add_call(self, context: int, call: int) -> None:
        self.pending_context += context
        self.pending_calls += 1
        self.reached[call] = self.reached.get(call, 0.0) + 1
        self.context[call] = self.context.get(call, 0.0) + context
        # Keep about the last PROJECTION_UNITS units: workloads drift
        self.first_call = min(self.first_call, call)
        if call == self.first_call and self.reached[call] > 2 * PROJECTION_UNITS:
            for counts in (self.reached, self.context, self.ended):
                for n in counts:
                    counts[n] /= 2

    def add_end(self, calls: int) -> None:
        self.ended[calls] = self.ended.get(calls, 0.0) + 1

    def poll(self, max_len: int | None) -> int:
        """Fold the calls since the last poll into the load window and project again. Returns their number."""
        new_calls = self.pending_calls
        self.polls.append((self.pending_context, new_calls))
        self.pending_context = self.pending_calls = 0
        calls = sum(c for _, c in self.polls)
        # Call numbers no unit reported (zero-context calls: forks, compaction, sub-agent starts) leave gaps, which
        # the walk steps over rather than stopping at
        seen = sorted(n for n, reached in self.reached.items() if n >= self.first_call and reached >= 0.5)
        if calls == 0 or not seen:
            self.maturity = None
            return new_calls
        survival, weight, total = 1.0, 0.0, 0.0
        for n in seen:
            mean = self.context[n] / self.reached[n]
            weight += survival
            total += survival * mean
            survival *= max(0.0, 1.0 - self.ended.get(n, 0.0) / self.reached[n])
        first, last = seen[0], seen[-1]
        young = self.context[first] / self.reached[first]
        growth = (mean - young) / (last - first) if last > first else 0.0
        if survival > 0 and growth > 0 and max_len is not None and max_len > mean:
            tail = (max_len - mean) / growth
            weight += survival * tail
            total += survival * tail * (mean + max_len) / 2
        current = sum(c for c, _ in self.polls) / calls
        self.young_context = young
        self.steady_context = max(total / weight, current, 1.0)
        self.maturity = current / self.steady_context
        return new_calls


class ConcurrencyController(Reconfigurable):
    def __init__(self, config: ConcurrencyConfig, *, fallback_cost: int) -> None:
        self.config = config
        self.fallback_cost = fallback_cost
        """Pessimistic per-unit cost for the starting cap when the engine reports no max context."""

        self.cap = float(config.initial_inflight or config.min_inflight)
        self.max_inflight = int(self.clamp(self.cap))
        # The cap last logged, and the reason of the gradual change since (None: none)
        self.logged_cap = self.max_inflight
        self.growth_reason: str | None = None
        self.bootstrapped = config.initial_inflight is not None
        # A manual cap that overrides the band and the adaptive loop until cleared
        self.target: int | None = None
        self.engine_max_len: int | None = None
        self.capacity_by_engine: dict[str, int] = {}

        self.turnover = 0.0
        self.pending_turnover = 0.0
        # Throughput is measured once the pool has turned over since this turnover (the start, or a cancel)
        self.fresh_turnover = 0.0
        # Engines paused for a weight update, and the polls after resume whose queue is the pause's backlog
        self.paused = False
        self.resume_skip = 0
        self.last_poll_time: float | None = None
        self.polls: deque[Poll] = deque(maxlen=LOAD_WINDOW_POLLS)
        self.queued_share = 0.0
        self.queued_share_by_role: dict[str, float] = {}
        # No queue trim before a load window was measured: at the start, and after a trim once inflight drained below
        # the new cap (and the pool turned over)
        self.draining = False
        self.trim_wait = LOAD_WINDOW_POLLS
        self.trim_turnover = -math.inf
        # The last trim was a queue trim and no window since has measured the queue below the trim threshold
        self.queue_trimmed = False
        # The last measurement showed growth still pays: the queue neither holds nor trims the cap
        self.paying = False
        self.elasticity: float | None = None
        # Fast climb: the window being measured, the reference its growth is judged against, and the previous one
        self.climbing = True
        self.climb_window: Window | None = None
        self.climb_reference: Window | None = None
        self.climb_previous: Window | None = None
        # Call clock: each model call advances it by 1/inflight
        self.call_turnover = 0.0
        self.projection = LoadProjection()
        # Steps: the level being measured, the last kept level, the next step and its direction
        self.level = Level(cap=self.max_inflight)
        self.base: Level | None = None
        self.step = STEP_MAX
        self.direction = 1
        # Throughput drop: the window being measured, the best one (its throughput decays), and whether the last
        # windows in a row dropped below it
        self.window = Window()
        self.best: Window | None = None
        self.drops = 0
        # The cap a throughput drop trimmed to: the drop lies above it, so steps up past it stay smallest
        self.ceiling: float | None = None

        self.set_limit: Callable[[int], None] = lambda n_max: None
        self.get_inflight: Callable[[], int] = lambda: 0
        self.on_overload: Callable[[int], None] | None = None

    def bind(
        self,
        *,
        set_limit: Callable[[int], None],
        get_inflight: Callable[[], int],
        on_overload: Callable[[int], None] | None = None,
    ) -> None:
        """Attach the dispatcher's hooks (no-ops until then). The dispatcher is constructed with this controller's
        initial cap, so no ``set_limit`` fires here. ``on_overload`` receives the units to cancel on an overload;
        without it a trim only blocks admission."""
        self.set_limit = set_limit
        self.get_inflight = get_inflight
        self.on_overload = on_overload

    # ── inbound hooks ────────────────────────────────────────────────────────

    def record_episode(self, tokens: int, calls: int) -> None:
        """One completed unit (from the dispatcher) after ``calls`` model calls. Advances the turnover clock by
        ``1/inflight``."""
        # The dispatcher releases the permit before this hook fires, so the completing unit is already off the count
        inflight = self.get_inflight() + 1
        self.turnover += 1 / inflight
        # Error markers consumed no engine capacity: growing on them floods the engines once the environment recovers
        if tokens > 0:
            self.pending_turnover += 1 / inflight
            self.projection.add_end(calls)

    def record_call(self, context: int, call: int) -> None:
        """The context a model call carried on from its trace's history, the ``call``-th of its unit (from the
        dispatcher's live deltas), folded into the call clock and the load projection at the next poll."""
        self.projection.add_call(context, call)

    def pause(self) -> None:
        """The engines are about to pause for a weight update: freeze. Requests arriving meanwhile wait in the
        engines' queues, which say nothing about load."""
        self.paused = True

    def resume(self) -> None:
        """The engines resumed: skip the polls whose queue holds the backlog of the pause, and start a fresh drop
        window so no window straddles the update."""
        self.paused = False
        self.resume_skip = RESUME_SKIP_POLLS
        self.window = Window(start_turnover=self.turnover)
        self.drops = 0

    def observe(self, samples: list[EngineLoadSample]) -> None:
        """Per-poll engine load push from the metrics collector: measure, then trim or climb."""
        now = time.monotonic()
        seconds = now - self.last_poll_time if self.last_poll_time is not None else 0.0
        self.last_poll_time = now
        if self.paused or not samples:
            # Growth needs a measurement: turnovers that pass without one would land on the next poll in one jump
            self.pending_turnover = 0.0
            return
        for sample in samples:
            # The running set lives on decode (or colocated) engines; prefill engines only pass it through
            if sample.kv_capacity_tokens and sample.role != "prefill":
                self.capacity_by_engine[sample.engine_id] = sample.kv_capacity_tokens
            if sample.max_model_len:
                self.engine_max_len = max(self.engine_max_len or 0, sample.max_model_len)

        inflight = self.get_inflight()
        loads = role_loads(samples)
        poll = Poll(loads, inflight, seconds)
        growth, self.pending_turnover = self.pending_turnover, 0.0
        calls = self.projection.poll(self.engine_max_len)
        if inflight > 0:
            self.call_turnover += calls / inflight
        # After a weight update the queue is the pause's backlog, not load
        if self.resume_skip > 0:
            self.resume_skip -= 1
            return
        self.polls.append(poll)
        units = max(sum(p.inflight for p in self.polls), 1)
        self.queued_share = sum(p.queued for p in self.polls) / units
        self.queued_share_by_role = {
            role: sum(p.roles[role].queued for p in self.polls if role in p.roles) / units for role in loads
        }

        if self.draining and inflight <= self.max_inflight:
            self.draining = False
            self.trim_wait = LOAD_WINDOW_POLLS
        if not self.draining:
            self.trim_wait = max(0, self.trim_wait - 1)
        if self.best is not None:
            self.best.tokens *= 0.5 ** (1 / BEST_HALF_LIFE_POLLS)
        self.window.add(poll)

        if self.target is not None:
            return

        # Without a configured start, the first capacity observation derives the pessimistic starting cap
        if not self.bootstrapped and self.capacity is not None:
            self.bootstrapped = True
            cost = float(self.engine_max_len or self.fallback_cost)
            self.cap = self.clamp(self.capacity / cost)
            get_logger().info(
                f"Derived initial max inflight {int(self.cap)} - {format_num(self.capacity, precision=1)} "
                f"KV cache tokens / {format_num(cost, precision=1)} tokens per episode"
            )
            self.apply_limit(int(self.cap), reason=None)

        if inflight <= 0 or self.throughput_drop(inflight) or self.queue_trim(inflight):
            return
        if self.climbing:
            self.climb(poll, growth, calls)
        else:
            self.measure(poll)

    def set_target(self, target: int | None) -> None:
        """Pin the cap at ``target``, outside the configured band too, or hand it back to the adaptive loop with
        None. A lower cap cancels nothing: admission waits for the pool to drain below it."""
        self.target = target
        self.cap = float(target) if target is not None else self.clamp(self.cap)
        self.apply_limit(int(self.cap), reason="manual target" if target is not None else "adaptive")
        self.restart_steps()

    def reconfigure(self, update: ConfigUpdate) -> None:
        """Move the band and clamp the cap into it. A manual target stays pinned, and a lower cap cancels nothing.
        ``initial_inflight`` only seeds the startup cap, so a new value changes nothing."""
        self.config = update.config.concurrency
        if self.target is None:
            self.cap = self.clamp(self.cap)
            self.apply_limit(int(self.cap), reason="config update")
            self.restart_steps()

    # ── climb ────────────────────────────────────────────────────────────────

    @property
    def mature(self) -> bool:
        """The pool's calls carry ``MATURITY_MIN`` of its projected steady load."""
        maturity = self.projection.maturity
        return maturity is not None and maturity >= MATURITY_MIN

    @property
    def steady(self) -> bool:
        """The pool carries its steady load: it is mature, or turned over since the start or the last cancel. A
        younger pool runs shorter contexts, so its throughput overstates what the same units sustain."""
        return self.turnover - self.fresh_turnover >= 1.0 or self.mature

    @property
    def headroom(self) -> float:
        """Queue headroom for growth: 1 with empty queues, 0 at the target (and 1 while growth pays)."""
        return 1.0 if self.paying else max(0.0, 1.0 - self.queued_share / QUEUE_TARGET)

    def climb(self, poll: Poll, growth: float, calls: int) -> None:
        """Grow by up to ``GROWTH_MAX`` per turnover, or ``CALL_GROWTH_MAX`` per call turnover while the pool carries
        ``MATURITY_MIN`` of its projected load, while the cap binds, tapered by the queue, and judge every window."""
        if self.climb_window is None:
            self.climb_window = Window(start_turnover=self.turnover, start_calls=self.call_turnover)
        window = self.climb_window
        window.add(poll)
        turned = self.turnover - window.start_turnover >= DWELL_TURNOVERS
        called = self.mature and self.call_turnover - window.start_calls >= WINDOW_CALLS
        if window.timed and (turned or called):
            self.climb_window = None
            self.log_growth()
            if self.steady:
                self.judge_climb(window, turned)
                if not self.climbing:
                    return
        if poll.inflight < BINDING_FRACTION * self.max_inflight or self.headroom == 0.0:
            return
        multiplier = 1.0 + (GROWTH_MAX - 1.0) * self.headroom
        cap = self.cap * multiplier**growth
        projection = self.projection
        if projection.maturity is not None and calls > 0:
            # No faster than GROWTH_MAX per two windows: each window must see the units the last one added
            by_calls = self.cap * min(
                (1.0 + (CALL_GROWTH_MAX - 1.0) * self.headroom) ** (calls / poll.inflight),
                multiplier ** (poll.seconds / (2 * DWELL_SECONDS)),
            )
            # Young units carry little load yet: admit only as many as keep the pool's load at MATURITY_MIN of its
            # projection. The maturity was measured on the units in flight over the load window; later ones count as
            # young.
            young = projection.young_context / projection.steady_context
            if young < MATURITY_MIN:
                shown = min(p.inflight for p in self.polls)
                room = shown * (projection.maturity - young) / (MATURITY_MIN - young)
                if room < by_calls:
                    window.held = True
                    by_calls = room
            cap = max(cap, by_calls)
        self.cap = self.clamp(max(self.cap, cap))
        self.apply_limit(int(self.cap), reason="climb", gradual=True)

    def judge_climb(self, window: Window, turned: bool) -> None:
        """End the fast climb at the first window whose growth did not pay, or that did not grow the pool by a smallest
        step although the projection did not hold it (the queue holds it). A window that only lasted its calls is
        compared once the pool grew by a largest step."""
        reference, previous = self.climb_reference, self.climb_previous
        self.climb_previous = window
        if reference is None:
            self.climb_reference = window
        elif window.inflight >= (STEP_MIN if turned else STEP_MAX) * reference.inflight:
            self.climb_reference = window
            self.elasticity = elasticity(window, reference)
            if self.elasticity < ELASTICITY:
                # The knee lies between the two windows: step down from their geometric midpoint
                midpoint = math.sqrt(window.inflight * reference.inflight)
                self.end_climb(f"elasticity {self.elasticity:.2f}", int(midpoint))
        elif window.inflight < STEP_MIN * previous.inflight and not window.held:
            # The queue holds the pool where it is: the knee by the queue, approached from above
            self.end_climb("held by the queue", self.max_inflight)

    def projected(self, units: float) -> float:
        """During the fast climb, the units that carry as much load as ``units`` will once the pool matured."""
        if self.climbing and self.projection.maturity is not None:
            return units * self.projection.maturity
        return units

    def end_climb(self, reason: str, cap: int) -> None:
        """Hand over to the steps, which measure ``cap`` first and then step down. A lower cap cancels nothing."""
        cap = int(self.clamp(self.projected(cap)))
        self.climbing = False
        get_logger().info(f"Concurrency fast climb ended at {self.max_inflight} ({reason})")
        self.cap = float(cap)
        self.apply_limit(cap, reason="fast climb ended")
        self.restart_steps(direction=-1)

    # ── steps ────────────────────────────────────────────────────────────────

    def restart_steps(self, direction: int = 1) -> None:
        """Measure the current cap afresh as the reference level, then step in ``direction``."""
        self.level = Level(cap=self.max_inflight)
        self.base = None
        self.direction = direction

    def measure(self, poll: Poll) -> None:
        """Fold one poll into the level being measured; once its dwell is complete, keep or reject it."""
        level = self.level
        if level.origin is not None:
            progress = (self.turnover - level.origin_turnover) / DWELL_TURNOVERS
            if progress < 1:
                self.cap = level.origin + (level.cap - level.origin) * progress
                self.apply_limit(int(self.cap), reason="step in", gradual=True)
                return
            self.cap = float(level.cap)
            self.apply_limit(level.cap, reason="step in", gradual=True)
            self.log_growth()
            level.origin = None
        if level.window is None:
            # A lowered cap drains first
            if poll.inflight > level.cap:
                return
            if level.settle is None:
                level.settle = self.turnover
            if self.turnover - level.settle < SETTLE_TURNOVERS:
                return
            level.window = Window(start_turnover=self.turnover)
        level.window.add(poll)
        up = self.base is not None and level.cap > self.base.cap
        if not level.window.lasted(self.turnover, DWELL_TURNOVERS * (2 if up else 1)):
            return
        if self.base is None:
            self.base = level
        elif not self.judge(level, self.base):
            return
        self.next_step()

    def judge(self, level: Level, base: Level) -> bool:
        """Keep the measured level when it is worth its units, else return to the reference level."""
        units = math.log(level.window.inflight / base.window.inflight)
        up = level.cap > base.cap
        # A step that did not move the pool (the cap does not bind) says nothing: treat it as not worth it
        if abs(units) < 0.5 * math.log(STEP_MIN) or (units > 0) != up:
            self.elasticity = None
            keep = False
        else:
            self.elasticity = elasticity(level.window, base.window)
            keep = (self.elasticity >= ELASTICITY) == up
        # Whether the units at the level the steps settle on pay: a kept step up and a rejected step down measured
        # it; a rejected step up says only that the units above do not
        if keep or not up:
            self.paying = self.elasticity is not None and self.elasticity >= ELASTICITY
        shown = f"{self.elasticity:.2f}" if self.elasticity is not None else "n/a"
        get_logger().info(
            f"Step {base.cap} -> {level.cap} {'kept' if keep else 'rejected'}: elasticity {shown}, "
            f"{level.window.throughput:.0f} vs {base.window.throughput:.0f} tokens/s at "
            f"{level.window.inflight:.0f} vs {base.window.inflight:.0f} in flight"
        )
        if keep:
            self.base = level
            # How far the step cleared the threshold: in proportion going up, held throughput going down
            margin = (self.elasticity - ELASTICITY) / (1 - ELASTICITY) if up else 1 - self.elasticity / ELASTICITY
            self.step = max(STEP_MIN, STEP_MAX ** min(1.0, margin))
            return True
        self.step = max(STEP_MIN, math.sqrt(self.step))
        self.cap = float(base.cap)
        self.apply_limit(base.cap, reason="step rejected")
        # Re-measure the reference level: the pool and the workload moved during the step
        self.restart_steps(direction=-self.direction)
        return False

    def next_step(self) -> None:
        """Step the cap from the reference level in the current direction. No step up while the cap does not bind,
        or while the queue is at the target and growth did not pay last."""
        base = self.base
        if self.direction > 0 and (base.window.inflight < BINDING_FRACTION * base.cap or self.headroom == 0.0):
            self.direction = -1
        step = self.step if self.direction > 0 else min(self.step, STEP_MAX**0.7)
        if self.direction > 0 and self.ceiling is not None and base.cap * step > self.ceiling:
            step = STEP_MIN
        cap = int(self.clamp(base.cap * step**self.direction))
        if cap == base.cap:
            # At a bound of the band: hold there and step the other way next
            self.direction = -self.direction
            self.level = Level(cap=base.cap)
            return
        if cap > base.cap:
            get_logger().info(f"Stepping concurrency {base.cap} -> {cap}")
            self.level = Level(cap=cap, origin=base.cap, origin_turnover=self.turnover)
            return
        self.cap = float(cap)
        self.apply_limit(cap, reason="step down")
        self.level = Level(cap=cap)

    # ── trims ────────────────────────────────────────────────────────────────

    def throughput_drop(self, inflight: int) -> bool:
        """Close the drop window every ``DWELL_SECONDS``. Two windows in a row whose throughput fell well below the
        best window's with as many units in flight trim the cap; a third with the engines the bottleneck cancels.
        Windows are timed only: past a cliff units finish too slowly for turnovers to come."""
        window = self.window
        if not window.timed:
            return False
        self.window = Window(start_turnover=self.turnover)
        if not self.steady:
            return False
        best = self.best
        if best is None or window.throughput > best.throughput:
            self.best = window
        if (
            best is None
            or window.inflight < BINDING_FRACTION * best.inflight
            or window.throughput >= (1 - THROUGHPUT_DROP) * best.throughput
        ):
            self.drops = 0
            return False
        self.drops += 1
        if self.drops < 2:
            return False
        stretched = best.latency > 0 and window.latency > LATENCY_STRETCH * best.latency
        cancel = self.drops >= 3 and (stretched or self.queued_share > QUEUE_CANCEL)
        self.trim(
            inflight * max(DROP_TRIM_MIN, window.throughput / best.throughput),
            inflight,
            cancel=cancel,
            reason=f"throughput drop ({window.throughput:.0f} < {best.throughput:.0f} tokens/s, "
            f"latency {window.latency:.1f}s vs {best.latency:.1f}s)",
        )
        self.ceiling = self.cap
        if cancel:
            # A fresh pool: measure the best window again once it has turned over
            self.best = None
            self.drops = 0
            self.fresh_turnover = self.turnover
        return True

    def queue_trim(self, inflight: int) -> bool:
        """Trim the units queued beyond the target once the pool has drained from the last trim and turned over,
        unless growth was last measured to pay or the cap does not bind (admission is held back elsewhere, and a
        trim from the smaller pool would undercut the cap); cancel them when a drained trim did not clear the queue."""
        if self.draining or self.trim_wait > 0:
            return False
        share = self.queued_share
        if share <= QUEUE_TRIM:
            self.queue_trimmed = False
            return False
        cancel = self.queue_trimmed and share > QUEUE_CANCEL
        # Past a cliff units finish too slowly for a turnover to come: an overloaded queue does not wait for it
        binding = inflight >= BINDING_FRACTION * self.max_inflight
        if not cancel and (self.paying or not binding or self.turnover - self.trim_turnover < 1.0):
            return False
        self.trim(
            self.projected(inflight * (1.0 - share + QUEUE_TARGET)),
            inflight,
            cancel=cancel,
            reason=f"{'queue overload' if cancel else 'queue'} {share:.2f}",
            brake=not cancel,
        )
        self.queue_trimmed = True
        return True

    def trim(self, target: float, inflight: int, *, reason: str, cancel: bool, brake: bool = False) -> None:
        """Lower the cap to ``target`` (clamped, never raised) and, with ``cancel``, shed the in-flight excess;
        without it, admission stays blocked until the pool drains below the new cap on its own. The trim was the
        step down: the steps measure the trimmed level next and step up with the smallest step. A ``brake`` trim
        during the fast climb leaves it climbing."""
        target = min(int(self.clamp(target)), self.max_inflight)
        self.cap = float(target)
        self.apply_limit(target, reason=reason)
        if cancel and self.on_overload is not None and inflight > target:
            self.on_overload(inflight - target)
        self.draining = True
        self.trim_turnover = self.turnover
        self.polls.clear()
        self.paying = False
        if self.climbing and brake:
            return
        self.climbing = False
        self.step = STEP_MIN
        self.restart_steps()

    # ── internals ────────────────────────────────────────────────────────────

    def apply_limit(self, n_max: int, *, reason: str | None, gradual: bool = False) -> None:
        """Set the cap. Every change is logged: a ``gradual`` one (the climb, a step admitted gradually) coalesced
        into one line when it ends or before the next other change."""
        if n_max == self.max_inflight:
            return
        if not gradual:
            self.log_growth()
        self.max_inflight = n_max
        self.set_limit(n_max)
        if gradual:
            self.growth_reason = reason
        else:
            self.log_change(reason)

    def log_growth(self) -> None:
        """Log the gradual change since the last logged cap."""
        if self.growth_reason is not None:
            self.log_change(self.growth_reason)
            self.growth_reason = None

    def log_change(self, reason: str | None) -> None:
        """Log the change since the last logged cap."""
        old, new = self.logged_cap, self.max_inflight
        self.logged_cap = new
        if reason is None or old == new:
            return
        verb = "Increased" if new > old else "Decreased"
        get_logger().info(
            f"{verb} concurrency {old} -> {new} ({reason}) - turnover={self.turnover:.1f} "
            f"queued={self.queued_share:.2f}"
        )

    def clamp(self, n_max: float) -> float:
        return min(max(n_max, float(self.config.min_inflight)), float(self.config.max_inflight or math.inf))

    @property
    def capacity(self) -> int | None:
        """Total KV tokens across decode engines; None until the first poll."""
        return sum(self.capacity_by_engine.values()) or None

    @property
    def binding(self) -> bool:
        """The last poll's units in flight fill the cap."""
        return bool(self.polls) and self.polls[-1].inflight >= BINDING_FRACTION * self.max_inflight

    @property
    def env_bound(self) -> bool:
        """The configured ``max_inflight`` holds the cap while growth still pays: the episodes' external resources,
        not the engines, bound the pool."""
        ceiling = self.config.max_inflight
        growing = self.climbing or self.paying
        return ceiling is not None and self.max_inflight >= ceiling and self.binding and growing

    # ── observability ────────────────────────────────────────────────────────

    def gauges(self) -> dict[str, float]:
        gauges = {
            "concurrency/max_inflight": float(self.max_inflight),
            "concurrency/turnover": self.turnover,
            "concurrency/capacity": float(self.capacity or 0),
            "concurrency/climbing": float(self.climbing),
            "concurrency/step": self.step**self.direction,
            "concurrency/paying": float(self.paying),
            "concurrency/paused": float(self.paused),
            "concurrency/call_turnover": self.call_turnover,
        }
        if self.projection.maturity is not None:
            gauges["concurrency/maturity"] = self.projection.maturity
            gauges["concurrency/steady_context"] = self.projection.steady_context
        if self.polls:
            gauges["concurrency/queued_share"] = self.queued_share
            for role, share in self.queued_share_by_role.items():
                gauges[f"concurrency/queued_share/{role}"] = share
        if self.base is not None:
            gauges["concurrency/base_throughput"] = self.base.window.throughput
            gauges["concurrency/base_inflight"] = self.base.window.inflight
        if self.best is not None:
            gauges["concurrency/best_throughput"] = self.best.throughput
            gauges["concurrency/best_latency"] = self.best.latency
        if self.elasticity is not None:
            gauges["concurrency/elasticity"] = self.elasticity
        return gauges


def elasticity(level: Window, reference: Window) -> float:
    """Relative change in throughput per relative change in units in flight between two windows."""
    units = math.log(level.inflight / reference.inflight)
    return math.log(max(level.throughput, 1e-9) / max(reference.throughput, 1e-9)) / units


def role_loads(samples: list[EngineLoadSample]) -> dict[str, RoleLoad]:
    """Pool one poll's engines per role. Queued counts every waiting request: one waiting on a KV load (offload
    tier, Mooncake, or a P/D transfer) waits just like one waiting for capacity."""
    by_role: dict[str, list[EngineLoadSample]] = {}
    for sample in samples:
        by_role.setdefault(sample.role or COLOCATED, []).append(sample)
    return {
        role: RoleLoad(
            queued=sum(sample.waiting for sample in group),
            finished=sum(sample.finished_delta for sample in group),
            generated=sum(sample.generation_tokens_delta for sample in group),
            latency=sum(sample.latency_delta for sample in group),
        )
        for role, group in by_role.items()
    }
