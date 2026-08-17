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

# EWMA smoothing for per-env cost/duration estimates
ESTIMATE_ALPHA = 0.1
# Multiplicative backoff: the first cut, and the harsher follow-up when
# overload survives a full drain
BACKOFF_FACTOR = 0.75
ESCALATED_BACKOFF_FACTOR = 0.5
# Per-step growth once the engines are clear and the cap binds
GROWTH_FACTOR = 1.02
# Grow only when the cap actually constrains admission
BINDING_FRACTION = 0.9
KAPPA_MIN = 0.25
KAPPA_MAX = 16.0


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
    waiting: int
    preemptions_delta: int


@dataclass
class EnvEstimate:
    """EWMA of final context tokens and wall-clock duration for one
    ``(kind, env)`` stream of completed episodes."""

    tokens: float | None = None
    duration: float | None = None

    def update(self, tokens: int, duration: float) -> None:
        self.tokens = tokens if self.tokens is None else (1 - ESTIMATE_ALPHA) * self.tokens + ESTIMATE_ALPHA * tokens
        self.duration = (
            duration if self.duration is None else (1 - ESTIMATE_ALPHA) * self.duration + ESTIMATE_ALPHA * duration
        )


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
        self.completed_steps = 0

        self.estimates: dict[tuple[str, str], EnvEstimate] = {}
        self.capacity_by_engine: dict[str, int] = {}
        self.capacity_reported = False
        # Eval episodes still owed per env; set on trigger, cleared once the
        # dispatcher reports no eval work at a step boundary
        self.eval_census: dict[str, int] | None = None
        self.eval_active = False

        self.signal = Signal.CLEAR
        self.all_clear_since_step = True
        self.prev_waiting: dict[str, int] = {}
        # After a cut, ignore further HARD signals until inflight has drained
        # below the new cap — the overload during drain is stale
        self.draining = False
        self.backoff_factor = BACKOFF_FACTOR

        self._set_limit: Callable[[int], None] | None = None
        self._get_inflight: Callable[[], int] | None = None

    def bind(self, *, set_limit: Callable[[int], None], get_inflight: Callable[[], int]) -> None:
        """Attach the outbound hooks. The dispatcher is constructed with this
        controller's initial cap, so no ``set_limit`` fires here."""
        self._set_limit = set_limit
        self._get_inflight = get_inflight

    # ── inbound hooks ────────────────────────────────────────────────────────

    def record_episode(self, env_name: str, kind: str, tokens: int, duration: float) -> None:
        """One completed episode (from the dispatcher). Errored episodes with
        no tokens carry no cost information and are skipped."""
        if tokens <= 0 or duration <= 0:
            return
        self.estimates.setdefault((kind, env_name), EnvEstimate()).update(tokens, duration)

    def on_eval_epoch(self, census: dict[str, int]) -> None:
        """Eval fired: ``census`` maps env name to total scheduled episodes."""
        if self.eval_census is None:
            self.eval_census = dict(census)
        else:
            for env_name, count in census.items():
                self.eval_census[env_name] = self.eval_census.get(env_name, 0) + count
        self.eval_active = True

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
        self.signal = worst
        if worst != Signal.CLEAR:
            self.all_clear_since_step = False

        inflight = self._get_inflight() if self._get_inflight is not None else 0
        if self.draining and inflight <= self.max_inflight:
            self.draining = False

        if worst == Signal.HARD and not self.draining:
            self.cut(inflight)
            return

        # First capacity observation before any step completed, without a
        # user-set start: raise the pre-capacity floor to the feedforward
        # bootstrap
        if (
            not self.frozen
            and self.kappa is None
            and self.config.initial_inflight is None
            and self.capacity is not None
        ):
            derived = self.clamp(self.capacity / self.cost_estimate())
            get_logger().info(
                f"Derived initial max inflight {derived} - {format_num(self.capacity, precision=1)} KV cache tokens "
                f"/ {format_num(self.bootstrap_cost, precision=1)} tokens per episode"
            )
            self.apply_limit(derived, reason=None)

    def on_step(self, *, inflight: int, eval_in_flight: bool) -> None:
        """Once per shipped train step: reweigh ``G``, grow ``kappa`` at most
        once, recompute the cap. Inert while still inside the configured
        freeze window (``frozen_steps``)."""
        self.completed_steps += 1
        if not eval_in_flight and self.eval_census is not None:
            self.eval_census = None
        self.eval_active = eval_in_flight and self.eval_census is not None

        capacity = self.capacity
        if self.frozen or capacity is None:
            self.all_clear_since_step = True
            return

        cost = self.cost_estimate()
        if self.kappa is None:
            # Continuity: respect a user-set start that implies over-commit;
            # otherwise jump to the safe full budget (kappa = 1)
            self.kappa = max(1.0, self.max_inflight * cost / capacity)
        elif self.all_clear_since_step:
            self.backoff_factor = BACKOFF_FACTOR
            if inflight >= BINDING_FRACTION * self.max_inflight:
                self.kappa = min(KAPPA_MAX, self.kappa * GROWTH_FACTOR)

        self.apply_limit(self.clamp(self.kappa * capacity / cost), reason="re-evaluation")
        self.all_clear_since_step = True

    # ── control law internals ────────────────────────────────────────────────

    @property
    def frozen(self) -> bool:
        """Inside the configured freeze window: no feedforward recompute, no
        growth, no bootstrap. HARD cuts stay live — an emergency brake is
        never frozen."""
        return self.completed_steps < self.config.frozen_steps

    def cut(self, inflight: int) -> None:
        """Multiplicative cut relative to what is actually running; freeze
        further cuts until the drain completes."""
        target = self.clamp(self.backoff_factor * max(inflight, self.floor))
        capacity = self.capacity
        if capacity is not None:
            self.kappa = max(KAPPA_MIN, target * self.cost_estimate() / capacity)
        self.draining = True
        # Escalate if overload survives the drain; reset on the next clear step
        self.backoff_factor = ESCALATED_BACKOFF_FACTOR
        self.apply_limit(target, reason="overload")

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
        """``G``: duration-weighted per-env cost mix over the work being
        admitted — eval census while an eval epoch is in flight, train ratios
        otherwise. Envs without completions fall back to the bootstrap cost."""
        if self.eval_active and self.eval_census:
            weights = {("eval", env): float(count) for env, count in self.eval_census.items()}
        else:
            weights = {("train", env): ratio for env, ratio in self.train_env_ratios.items()}

        weighted_cost = 0.0
        total_weight = 0.0
        for key, weight in weights.items():
            estimate = self.estimates.get(key)
            tokens = estimate.tokens if estimate is not None and estimate.tokens is not None else self.bootstrap_cost
            duration = estimate.duration if estimate is not None and estimate.duration is not None else 1.0
            weighted_cost += weight * duration * tokens
            total_weight += weight * duration
        return weighted_cost / total_weight if total_weight > 0 else float(self.bootstrap_cost)

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
        }
