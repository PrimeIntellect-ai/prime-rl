"""ConcurrencyController: adaptive cap on in-flight units of inference work.

A unit is whatever the dispatcher admits against one permit — the controller
treats it as a black box with a token cost and a duration. Sets
``n_max = clamp(kappa * C / G, floor, max_inflight)``:

- ``C`` — GPU KV capacity in tokens, summed over decode engines (from
  ``vllm:cache_config_info`` labels, pushed by ``InferenceMetricsCollector``).
- ``G`` — expected cost per unit in tokens: the max of the per-env completion
  EWMAs (weighted by the dispatcher's live in-flight mix) and the size-biased
  live request cost measured off the engines.
- ``kappa`` — learned over-commit factor. Grows slowly while the engines are
  clear and the cap binds; backs off multiplicatively on overload.

The controller is a pure state machine — it owns no tasks or clients. Three
drivers call into it:

- the metrics collector pushes ``observe(samples)`` every poll (where all
  cap control happens, clocked by pipeline turnovers),
- the dispatcher reports ``record_episode(...)`` per completed unit,
- the orchestrator's step loop calls ``on_step(...)`` once per shipped step.

Outbound it calls the hooks bound via :meth:`bind`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

from prime_rl.configs.orchestrator import ConcurrencyConfig
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.utils import format_num

ESTIMATE_ALPHA = 1 / 1024
"""EWMA smoothing for the per-env cost/duration estimates: an effective window of ~1/alpha completed units per env."""

BACKOFF_FACTOR = 0.8
"""Multiplicative backoff of the first overload cut."""

ESCALATED_BACKOFF_FACTOR = 0.5
"""Backoff of a follow-up cut when overload survives a full drain."""

GROWTH_FACTOR = 1.003
"""Per-poll kappa growth while the engines are clear and the cap binds."""

BINDING_FRACTION = 0.9
"""The cap counts as binding when inflight reaches this fraction of it."""

KAPPA_MIN = 0.25
KAPPA_MAX = 8.0
"""Bounds on the learned over-commit factor."""

KAPPA_CEILING_FRACTION = 0.9
"""An overload cut pins future kappa growth at this fraction of the kappa that overloaded."""

KAPPA_CEILING_RELIEF = 1.00002
"""Per-poll relief of the kappa ceiling (~+2% per 10 minutes), so a lightened workload can re-probe."""

KV_USAGE_SOFT = 0.7
"""SOFT (growth veto) once any decode engine's KV usage crosses this."""

KV_USAGE_TRIGGER = 0.85
"""Above this usage, trim the cap and the in-flight pool."""

KV_USAGE_TARGET = 0.75
"""A trim resizes to inflight * target / usage — below the trigger, so pool growth has headroom before the next trim."""

KV_TRIM_COOLDOWN_POLLS = 6
"""Polls between kv-headroom trims, letting each trim propagate before the next is sized."""

QUEUE_RATIO = 0.5
"""HARD once capacity-queued requests exceed this fraction of running requests for the persistence window."""

QUEUE_PERSISTENCE_POLLS = 6
"""Consecutive polls of queue overload before the HARD cut; filters natural turn-completion bursts."""

QUEUE_CUT_FRACTION = 0.9
"""A queue cut targets this fraction of what the engines are serving."""

MAX_TURNOVER_GROWTH = 1.25
"""Maximum cap raise per pipeline turnover (each completion advances the turnover by 1/inflight)."""

REEVAL_DEADBAND = 0.02
"""Minimum relative cap move a per-poll re-evaluation applies."""


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
    # Mean prompt+generation tokens per request this poll interval and the
    # request count behind it (None/0 when the interval saw no completions)
    mean_request_cost: float | None
    interval_requests: int


@dataclass
class EnvEstimate:
    """Bias-corrected per-episode EWMA of final context tokens and wall-clock
    duration for one ``(kind, env)`` stream. Decayed-sum-over-decayed-weight
    behaves as the plain mean while samples are few (no single-episode
    cold-start bias) and converges to a ~1/ESTIMATE_ALPHA-episode window."""

    weight: float = 0.0
    tokens_sum: float = 0.0
    duration_sum: float = 0.0

    def update(self, tokens: int, duration: float) -> None:
        decay = 1 - ESTIMATE_ALPHA
        self.weight = decay * self.weight + 1
        self.tokens_sum = decay * self.tokens_sum + tokens
        self.duration_sum = decay * self.duration_sum + duration

    @property
    def tokens(self) -> float | None:
        return self.tokens_sum / self.weight if self.weight > 0 else None

    @property
    def duration(self) -> float | None:
        return self.duration_sum / self.weight if self.weight > 0 else None


class ConcurrencyController:
    def __init__(
        self,
        config: ConcurrencyConfig,
        *,
        train_env_ratios: dict[str, float],
        floor: int,
        fallback_cost: int,
    ) -> None:
        self.config = config
        self.train_env_ratios = train_env_ratios
        self.floor = floor
        self.fallback_cost = fallback_cost
        # Engine-reported max context; the pessimistic per-episode cost bound
        self.engine_max_len: int | None = None

        self.max_inflight = config.initial_inflight or floor
        # None until the first poll with real cost data (or a cut initializes it early)
        self.kappa: float | None = None
        self.bootstrapped = False
        self.completed_steps = 0
        # Pipeline turnovers completed; the controller's clock (see MAX_TURNOVER_GROWTH)
        self.turnover = 0.0
        # Highest cap the slew currently permits; compounds with the turnover
        self.slew_allowance = float(self.max_inflight)

        self.estimates: dict[tuple[str, str], EnvEstimate] = {}
        # Request-weighted decayed mean of measured per-request KV cost.
        # Completion-based estimates only see episodes that finished — after a
        # cap raise the fast short episodes finish first and drag them down,
        # inviting a bigger raise. The request stream has no such bias: every
        # turn resends its episode's full context, so in-flight episodes are
        # counted at their current size the whole time they run.
        self.observed_cost_sum = 0.0
        self.observed_cost_weight = 0.0
        self.capacity_by_engine: dict[str, int] = {}
        self.capacity_reported = False

        self.signal = Signal.CLEAR
        self.all_clear_since_step = True
        self.prev_waiting: dict[str, int] = {}
        # Consecutive polls with the capacity queue above QUEUE_RATIO of running
        self.queue_overload_polls = 0
        self.kappa_ceiling = KAPPA_MAX
        # Polls until the next kv-headroom trim may fire
        self.trim_cooldown = 0
        # After a cut, ignore further HARD signals until inflight has drained
        # below the new cap — the overload during drain is stale
        self.draining = False
        self.backoff_factor = BACKOFF_FACTOR

        self._set_limit: Callable[[int], None] | None = None
        self._get_inflight: Callable[[], int] | None = None
        self._get_inflight_mix: Callable[[], dict[tuple[str, str], int]] | None = None
        self._on_overload: Callable[[int], None] | None = None

    def bind(
        self,
        *,
        set_limit: Callable[[int], None],
        get_inflight: Callable[[], int],
        get_inflight_mix: Callable[[], dict[tuple[str, str], int]] | None = None,
        on_overload: Callable[[int], None] | None = None,
    ) -> None:
        """Attach the outbound hooks. The dispatcher is constructed with this
        controller's initial cap, so no ``set_limit`` fires here.
        ``get_inflight_mix`` supplies the live per-``(kind, env)`` in-flight
        counts that weight the cost mix; ``on_overload`` receives the unit
        excess on an overload cut so the dispatcher can cancel in-flight work
        instead of just blocking admission."""
        self._set_limit = set_limit
        self._get_inflight = get_inflight
        self._get_inflight_mix = get_inflight_mix
        self._on_overload = on_overload

    # ── inbound hooks ────────────────────────────────────────────────────────

    def record_episode(self, env_name: str, kind: str, tokens: int, duration: float) -> None:
        """One completed episode (from the dispatcher). Every completion
        advances the turnover clock and the slew allowance (errored episodes
        free a slot too); only episodes with tokens carry cost information."""
        inflight = self._get_inflight() if self._get_inflight is not None else 0
        fraction = 1 / max(inflight, self.floor, 1)
        self.turnover += fraction
        self.slew_allowance = max(self.slew_allowance, float(self.max_inflight)) * MAX_TURNOVER_GROWTH**fraction
        if tokens <= 0 or duration <= 0:
            return
        self.estimates.setdefault((kind, env_name), EnvEstimate()).update(tokens, duration)

    def observe(self, samples: list[EngineLoadSample]) -> None:
        """Per-poll engine load push from the metrics collector. Classifies
        the worst engine and applies the (rare) immediate cut on HARD."""
        if not samples:
            return
        for sample in samples:
            if sample.kv_capacity_tokens and sample.role != "prefill":
                self.capacity_by_engine[sample.engine_id] = sample.kv_capacity_tokens
            if sample.max_model_len:
                self.engine_max_len = max(self.engine_max_len or 0, sample.max_model_len)
        for sample in samples:
            if sample.mean_request_cost is not None and sample.interval_requests > 0:
                decay = 1 - ESTIMATE_ALPHA
                self.observed_cost_sum = decay * self.observed_cost_sum + sample.interval_requests * sample.mean_request_cost
                self.observed_cost_weight = decay * self.observed_cost_weight + sample.interval_requests
        if not self.capacity_reported and self.capacity is not None:
            self.capacity_reported = True
            max_len = format_num(self.engine_max_len, precision=1) if self.engine_max_len else "unknown"
            get_logger().info(
                f"Inference reports {format_num(self.capacity, precision=1)} tokens of KV cache capacity - "
                f"max model len {max_len}"
            )

        worst = Signal.CLEAR
        for sample in samples:
            if sample.preemptions_delta > 0:
                worst = Signal.HARD
                break
            if sample.waiting > 0 and self.prev_waiting.get(sample.engine_id, 0) > 0:
                worst = Signal.SOFT
            if sample.role != "prefill" and sample.kv_usage > KV_USAGE_SOFT:
                worst = Signal.SOFT
        self.prev_waiting = {sample.engine_id: sample.waiting for sample in samples}

        total_running = sum(sample.running for sample in samples)
        total_queued = sum(
            sample.waiting_capacity if sample.waiting_capacity is not None else sample.waiting for sample in samples
        )
        if total_running > 0 and total_queued > QUEUE_RATIO * total_running:
            self.queue_overload_polls += 1
        else:
            self.queue_overload_polls = 0
        queue_overload = self.queue_overload_polls >= QUEUE_PERSISTENCE_POLLS
        if queue_overload:
            worst = Signal.HARD

        self.signal = worst
        if worst != Signal.CLEAR:
            self.all_clear_since_step = False

        inflight = self._get_inflight() if self._get_inflight is not None else 0
        if self.draining and inflight <= self.max_inflight:
            self.draining = False

        self.kappa_ceiling = min(KAPPA_MAX, self.kappa_ceiling * KAPPA_CEILING_RELIEF)
        if (
            worst == Signal.CLEAR
            and total_queued == 0
            and not self.draining
            and self.kappa is not None
            and inflight >= BINDING_FRACTION * self.max_inflight
        ):
            # total_queued == 0: any capacity-queuing means the KV blocks are
            # full right now — unlike generic waiting it is never a benign
            # burst, and thrash onset is a cliff, not a slope
            self.kappa = min(self.kappa_ceiling, self.kappa * GROWTH_FACTOR)

        if queue_overload and not self.draining:
            self.queue_overload_polls = 0
            self.cut(inflight, target=self.clamp(QUEUE_CUT_FRACTION * total_running), reason="queue overload")
            return

        if worst == Signal.HARD and not self.draining:
            self.cut(inflight)
            return

        # First capacity observation without a user-set start: raise the
        # pre-capacity floor to the feedforward bootstrap, once
        if (
            not self.bootstrapped
            and self.kappa is None
            and self.config.initial_inflight is None
            and self.capacity is not None
        ):
            self.bootstrapped = True
            derived = self.clamp(self.capacity / self.cost_estimate())
            get_logger().info(
                f"Derived initial max inflight {derived} - {format_num(self.capacity, precision=1)} KV cache tokens "
                f"/ {format_num(self.bootstrap_cost, precision=1)} tokens per episode"
            )
            self.apply_limit(derived, reason=None)

        capacity = self.capacity
        if capacity is None:
            return
        if self.kappa is None:
            # Wait for real cost data — the pre-traffic max-context fallback
            # would read a user-set start as intentional over-commit
            if self.observed_cost <= 0 and not self.estimates:
                return
            # Continuity: respect a user-set start that implies over-commit;
            # otherwise jump to the safe full budget (kappa = 1)
            self.kappa = max(1.0, self.max_inflight * self.cost_estimate() / capacity)

        max_usage = max((s.kv_usage for s in samples if s.role != "prefill"), default=0.0)
        self.trim_cooldown = max(0, self.trim_cooldown - 1)
        if max_usage > KV_USAGE_TRIGGER and inflight > 0 and not self.draining and self.trim_cooldown == 0:
            sustainable = self.clamp(inflight * KV_USAGE_TARGET / max_usage)
            if sustainable < self.max_inflight:
                # Re-derive kappa like a cut does — otherwise the feedforward
                # pushes the cap straight back into the trim zone and the
                # controller sheds young episodes every cooldown. Growth
                # resumes once usage falls below KV_USAGE_SOFT, giving a
                # grow / hold / trim band instead of a shed loop.
                if self.kappa is not None:
                    self.kappa = max(KAPPA_MIN, sustainable * self.cost_estimate() / capacity)
                self.apply_limit(sustainable, reason=f"kv headroom (usage {max_usage:.2f})")
                self.slew_allowance = float(self.max_inflight)
            if self._on_overload is not None and inflight > sustainable:
                self._on_overload(inflight - sustainable)
            self.trim_cooldown = KV_TRIM_COOLDOWN_POLLS
            return

        target = self.clamp(min(self.kappa * capacity / self.cost_estimate(), self.slew_allowance))
        if self.draining:
            target = min(target, self.max_inflight)
        if abs(target - self.max_inflight) >= REEVAL_DEADBAND * self.max_inflight:
            self.apply_limit(target, reason="re-evaluation")
            self.slew_allowance = float(self.max_inflight)

    def on_step(self, *, inflight: int) -> None:
        """Once per shipped train step: backoff reset. All cap control is
        poll-clocked in ``observe``."""
        self.completed_steps += 1
        if self.all_clear_since_step:
            self.backoff_factor = BACKOFF_FACTOR
        self.all_clear_since_step = True

    # ── control law internals ────────────────────────────────────────────────

    def cut(self, inflight: int, target: int | None = None, reason: str = "overload") -> None:
        """Cut the cap and freeze further cuts until the drain completes.
        Preemption cuts back off multiplicatively from what is in flight; a
        queue cut passes an explicit target — what the engines actually serve."""
        if target is None:
            target = self.clamp(self.backoff_factor * max(inflight, self.floor))
        # A cut never raises: a queue-derived target can exceed the cap when
        # ``running`` is inflated by work the dispatcher no longer tracks
        target = min(target, self.max_inflight)
        if self.kappa is not None:
            self.kappa_ceiling = max(KAPPA_MIN, self.kappa * KAPPA_CEILING_FRACTION)
        capacity = self.capacity
        if capacity is not None:
            self.kappa = max(KAPPA_MIN, target * self.cost_estimate() / capacity)
        self.draining = True
        # Escalate if overload survives the drain; reset on the next clear step
        self.backoff_factor = ESCALATED_BACKOFF_FACTOR
        self.apply_limit(target, reason=reason)
        self.slew_allowance = float(self.max_inflight)
        if self._on_overload is not None and inflight > target:
            self._on_overload(inflight - target)

    def apply_limit(self, n_max: int, *, reason: str | None) -> None:
        if n_max == self.max_inflight:
            return
        if reason is not None:
            verb = "Increased" if n_max > self.max_inflight else "Decreased"
            kappa = f"{self.kappa:.2f}" if self.kappa is not None else "unset"
            get_logger().info(
                f"{verb} concurrency {self.max_inflight} -> {n_max} at step {self.completed_steps} ({reason}) - "
                f"kappa={kappa} cost={format_num(self.cost_estimate(), precision=1)} "
                f"capacity={format_num(self.capacity or 0, precision=1)} signal={self.signal.name.lower()}"
            )
            if self.estimates:
                snapshot = " | ".join(
                    f"{kind}/{env} tokens={format_num(estimate.tokens or 0, precision=1)} "
                    f"duration={format_time(estimate.duration or 0)}"
                    for (kind, env), estimate in sorted(self.estimates.items())
                )
                get_logger().debug(f"Concurrency estimates - {snapshot}")
        self.max_inflight = n_max
        if self._set_limit is not None:
            self._set_limit(n_max)

    def clamp(self, n_max: float) -> int:
        ceiling = self.config.max_inflight or math.inf
        return int(min(max(n_max, self.floor), ceiling))

    @property
    def capacity(self) -> int | None:
        """Total KV tokens across decode engines; None until the first poll."""
        return sum(self.capacity_by_engine.values()) or None

    @property
    def bootstrap_cost(self) -> int:
        """Pessimistic per-episode cost before completions exist: the
        engine-reported max context length (an episode cannot exceed it),
        falling back to the configured training sequence length."""
        return self.engine_max_len or self.fallback_cost

    def cost_estimate(self) -> float:
        """``G``: per-env cost mix weighted by the dispatcher's live in-flight
        counts — a measurement of the standing pool, so train and eval need no
        separate treatment (eval episodes enter the mix as they are admitted).
        Before anything is in flight, the configured train ratios stand in."""
        mix: dict[tuple[str, str], float] = {}
        if self._get_inflight_mix is not None:
            mix = {key: float(n) for key, n in self._get_inflight_mix().items()}
        if not mix:
            mix = {("train", env): ratio for env, ratio in self.train_env_ratios.items()}
        return max(self.mix_cost(mix), self.observed_cost)

    @property
    def observed_cost(self) -> float:
        """Measured mean per-request KV cost off the live request stream;
        floors the completion-based estimate, which under-prices the pool
        whenever completions lag admissions (cap raises, cold start)."""
        return self.observed_cost_sum / self.observed_cost_weight if self.observed_cost_weight > 0 else 0.0

    def mix_cost(self, weights: dict[tuple[str, str], float]) -> float:
        """Mean per-unit cost over one ``(kind, env)`` weight mix. Envs
        without completions fall back to the observed request cost (the best
        available measurement), and to the pessimistic bootstrap cost only
        before any traffic exists."""
        fallback = self.observed_cost or float(self.bootstrap_cost)
        weighted_cost = 0.0
        total_weight = 0.0
        for key, weight in weights.items():
            estimate = self.estimates.get(key)
            tokens = estimate.tokens if estimate is not None and estimate.tokens is not None else fallback
            weighted_cost += weight * tokens
            total_weight += weight
        return weighted_cost / total_weight if total_weight > 0 else fallback

    # ── observability ────────────────────────────────────────────────────────

    def gauges(self) -> dict[str, float]:
        return {
            "concurrency/max_inflight": float(self.max_inflight),
            "concurrency/kappa": self.kappa if self.kappa is not None else 0.0,
            "concurrency/cost_estimate": self.cost_estimate(),
            "concurrency/capacity_tokens": float(self.capacity or 0),
            "concurrency/signal": float(
                {Signal.CLEAR: 0, Signal.SOFT: 1, Signal.HARD: 2}[self.signal],
            ),
            "concurrency/queue_overload_polls": float(self.queue_overload_polls),
            "concurrency/observed_cost": self.observed_cost,
            "concurrency/turnover": self.turnover,
            "concurrency/slew_allowance": self.slew_allowance,
        }
