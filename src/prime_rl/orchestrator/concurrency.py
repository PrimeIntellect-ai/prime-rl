"""ConcurrencyController: adaptive in-flight episode cap for the dispatcher.

Sets ``n_max = clamp(kappa * C / G, floor, max_inflight)``:

- ``C`` — GPU KV capacity in tokens, summed over decode engines (from
  ``vllm:cache_config_info`` labels, pushed by ``InferenceMetricsCollector``).
- ``G`` — expected episode cost in tokens: per-env EWMAs of final context
  size and duration over completed episodes, aggregated with
  duration-corrected weights (train envs by sampling ratio, eval envs by
  scheduled episodes while an eval epoch is in flight).
- ``kappa`` — learned over-commit factor. Grows slowly while the engines are
  clear and the cap binds; backs off multiplicatively on overload.

The controller is a pure state machine — it owns no tasks or clients. Three
drivers call into it:

- the metrics collector pushes ``observe(samples)`` every poll (the only
  path that may cut the cap outside a step boundary),
- the dispatcher reports ``record_episode(...)`` per completion,
- the orchestrator's step loop calls ``on_step(...)`` once per shipped step
  and ``on_eval_epoch(census)`` when evals fire.

Outbound it only calls the ``set_limit`` hook bound via :meth:`bind`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

from prime_rl.configs.orchestrator import ConcurrencyConfig
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.utils import format_num

# Per-episode EWMA smoothing for the cost/duration estimates: an effective
# window of ~1/alpha episodes per env, independent of batch size and step
# cadence, wide enough to average out heavy-tailed episode lengths.
ESTIMATE_ALPHA = 1 / 512
# Multiplicative backoff: the first cut, and the harsher follow-up when
# overload survives a full drain
BACKOFF_FACTOR = 0.75
ESCALATED_BACKOFF_FACTOR = 0.5
# Per-poll growth while the engines are clear and the cap binds. Clocked on
# polls, not steps: step duration varies from seconds to tens of minutes
# across workloads, so step-clocked growth adapts at wildly different speeds
# — and a binding check sampled only at the boundary instant misses caps
# that bind all step long (episodes complete in bursts at boundaries).
GROWTH_FACTOR = 1.003
# Grow only when the cap actually constrains admission
BINDING_FRACTION = 0.9
KAPPA_MIN = 0.25
KAPPA_MAX = 16.0
# Queue-overload HARD: capacity-queued requests exceed this fraction of
# running requests for QUEUE_PERSISTENCE_POLLS consecutive polls. Agentic
# rollouts overload by queueing, not preempting — admission control parks
# excess load in the waiting queue, so preemptions alone miss it. The
# persistence window filters natural turn-completion bursts.
QUEUE_RATIO = 0.5
QUEUE_PERSISTENCE_POLLS = 12
# Cut to just under what the engines are actually serving
QUEUE_CUT_FRACTION = 0.9
# Cap raises at most this factor per re-evaluation. A raise floods in fresh
# episodes whose fast finishers drag the cost EWMA down, inviting a bigger
# raise — the slew limit turns that spiral into a ramp the overload cut can
# interrupt early. Cuts are never limited.
MAX_STEP_GROWTH = 1.25


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
        # None until the first unfrozen on_step (or a cut initializes it early)
        self.kappa: float | None = None
        self.bootstrapped = False
        self.completed_steps = 0

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
        # Eval episodes still in flight or owed, per env; set on trigger,
        # decremented per completion, force-cleared once the dispatcher
        # reports no eval work at a step boundary
        self.eval_remaining: dict[str, int] = {}

        self.signal = Signal.CLEAR
        self.all_clear_since_step = True
        self.prev_waiting: dict[str, int] = {}
        # Consecutive polls with the capacity queue above QUEUE_RATIO of running
        self.queue_overload_polls = 0
        # After a cut, ignore further HARD signals until inflight has drained
        # below the new cap — the overload during drain is stale
        self.draining = False
        self.backoff_factor = BACKOFF_FACTOR

        self._set_limit: Callable[[int], None] | None = None
        self._get_inflight: Callable[[], int] | None = None
        self._on_overload: Callable[[int], None] | None = None

    def bind(
        self,
        *,
        set_limit: Callable[[int], None],
        get_inflight: Callable[[], int],
        on_overload: Callable[[int], None] | None = None,
    ) -> None:
        """Attach the outbound hooks. The dispatcher is constructed with this
        controller's initial cap, so no ``set_limit`` fires here.
        ``on_overload`` receives the episode excess on an overload cut so the
        dispatcher can shed in-flight work instead of just blocking admission."""
        self._set_limit = set_limit
        self._get_inflight = get_inflight
        self._on_overload = on_overload

    # ── inbound hooks ────────────────────────────────────────────────────────

    def record_episode(self, env_name: str, kind: str, tokens: int, duration: float) -> None:
        """One completed episode (from the dispatcher). Errored episodes with
        no tokens carry no cost information and are skipped."""
        if kind == "eval" and env_name in self.eval_remaining:
            self.eval_remaining[env_name] = max(0, self.eval_remaining[env_name] - 1)
        if tokens <= 0 or duration <= 0:
            return
        self.estimates.setdefault((kind, env_name), EnvEstimate()).update(tokens, duration)

    def on_eval_epoch(self, census: dict[str, int]) -> None:
        """Eval fired: ``census`` maps env name to total scheduled episodes.
        Decremented per completion, so the eval share of the cost mix decays
        continuously as the epoch drains."""
        for env_name, count in census.items():
            self.eval_remaining[env_name] = self.eval_remaining.get(env_name, 0) + count

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

        if (
            worst == Signal.CLEAR
            and not self.draining
            and not self.frozen
            and self.kappa is not None
            and inflight >= BINDING_FRACTION * self.max_inflight
        ):
            self.kappa = min(KAPPA_MAX, self.kappa * GROWTH_FACTOR)

        if queue_overload and not self.draining:
            self.queue_overload_polls = 0
            self.cut(inflight, target=self.clamp(QUEUE_CUT_FRACTION * total_running), reason="queue overload")
            return

        if worst == Signal.HARD and not self.draining:
            self.cut(inflight)
            return

        # First capacity observation before any step completed, without a
        # user-set start: raise the pre-capacity floor to the feedforward
        # bootstrap. Fires once — every later change goes through on_step
        # (or a HARD cut), so the cap is stable between step boundaries.
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

    def on_step(self, *, inflight: int, eval_in_flight: bool) -> None:
        """Once per shipped train step: recompute the cap from the current
        cost estimate (kappa grows per poll in ``observe``). Inert while
        still inside the configured freeze window (``frozen_steps``)."""
        frozen = self.frozen
        self.completed_steps += 1
        # Safety net for eval episodes that never report a completion
        # (task exceptions surface as synthetic markers)
        if not eval_in_flight and self.eval_remaining:
            self.eval_remaining.clear()

        capacity = self.capacity
        if frozen or capacity is None:
            self.all_clear_since_step = True
            return

        cost = self.cost_estimate()
        if self.kappa is None:
            # Continuity: respect a user-set start that implies over-commit;
            # otherwise jump to the safe full budget (kappa = 1)
            self.kappa = max(1.0, self.max_inflight * cost / capacity)
        elif self.all_clear_since_step:
            self.backoff_factor = BACKOFF_FACTOR

        target = min(self.kappa * capacity / cost, MAX_STEP_GROWTH * self.max_inflight)
        self.apply_limit(self.clamp(target), reason="re-evaluation")
        self.all_clear_since_step = True

    # ── control law internals ────────────────────────────────────────────────

    @property
    def frozen(self) -> bool:
        """Inside the configured freeze window: the first ``frozen_steps``
        step boundaries do not recompute the cap or grow ``kappa``. The
        initial bootstrap derivation and HARD cuts stay live — the starting
        value and the emergency brake are never frozen."""
        return self.completed_steps < self.config.frozen_steps

    def cut(self, inflight: int, target: int | None = None, reason: str = "overload") -> None:
        """Cut the cap and freeze further cuts until the drain completes.
        Preemption cuts back off multiplicatively from what is in flight; a
        queue cut passes an explicit target — what the engines actually serve."""
        if target is None:
            target = self.clamp(self.backoff_factor * max(inflight, self.floor))
        # A cut never raises: a queue-derived target can exceed the cap when
        # ``running`` is inflated by work the dispatcher no longer tracks
        target = min(target, self.max_inflight)
        capacity = self.capacity
        if capacity is not None:
            self.kappa = max(KAPPA_MIN, target * self.cost_estimate() / capacity)
        self.draining = True
        # Escalate if overload survives the drain; reset on the next clear step
        self.backoff_factor = ESCALATED_BACKOFF_FACTOR
        self.apply_limit(target, reason=reason)
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
        """``G``: duration-weighted per-env cost mix of the in-flight pool.
        The train mix (by sampling ratio) and the eval mix (by remaining
        census episodes) interpolate on the eval share of the cap — the share
        starts at ``min(1, census / n_max)`` when an epoch fires and decays
        continuously to zero as it drains, so the cap glides back to the
        train price instead of snapping."""
        train_cost = self.mix_cost({("train", env): ratio for env, ratio in self.train_env_ratios.items()})
        remaining = sum(self.eval_remaining.values())
        if remaining <= 0:
            return max(train_cost, self.observed_cost)
        eval_cost = self.mix_cost({("eval", env): float(n) for env, n in self.eval_remaining.items() if n > 0})
        eval_share = min(1.0, remaining / max(self.max_inflight, 1))
        return max(eval_share * eval_cost + (1 - eval_share) * train_cost, self.observed_cost)

    @property
    def observed_cost(self) -> float:
        """Measured mean per-request KV cost off the live request stream;
        floors the completion-based estimate, which under-prices the pool
        whenever completions lag admissions (cap raises, cold start)."""
        return self.observed_cost_sum / self.observed_cost_weight if self.observed_cost_weight > 0 else 0.0

    def mix_cost(self, weights: dict[tuple[str, str], float]) -> float:
        """Duration-weighted mean episode cost over one ``(kind, env)`` weight
        mix. Envs without completions fall back to the observed request cost
        (the best available measurement), and to the pessimistic bootstrap
        cost only before any traffic exists."""
        fallback = self.observed_cost or float(self.bootstrap_cost)
        weighted_cost = 0.0
        total_weight = 0.0
        for key, weight in weights.items():
            estimate = self.estimates.get(key)
            tokens = estimate.tokens if estimate is not None and estimate.tokens is not None else fallback
            duration = estimate.duration if estimate is not None and estimate.duration is not None else 1.0
            weighted_cost += weight * duration * tokens
            total_weight += weight * duration
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
        }
