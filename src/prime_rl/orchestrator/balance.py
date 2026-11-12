"""Trainer / inference GPU split advice.

An RL run is a pipeline of GPU pools: inference engines make rollouts (in P/D, a prefill pool feeds a decode pool)
and the trainer consumes them. The slowest pool sets the run's speed while the others idle part of the time. Each
pool's capacity is its throughput over its busy share, what it would make busy all the time, assumed to scale with
its units; the advisor searches every valid split of the same units for the fastest:

    speed(split) = min over pools p of  capacity_p * units'_p / units_p

- **Trainer:** busy from receiving a batch to requesting the next (training, broadcasting, saving).
- **Inference:** the seconds the engines spent executing model steps (``/prime_rl/engine_busy``), outside weight-update
  pauses, per role under P/D. Engines that step whenever they hold a sequence (decode) read as busy below their knee
  too, so from ``SATURATED_BUSY`` on, and when busy time is not reported, a role's capacity is only a lower bound.
- **Environment:** a cap held at ``concurrency.max_inflight`` while growth still pays means the episodes' external
  resources bound the run, not GPUs; no split is advised.

A lower bound only understates a pool, so a speed computed with one is itself a lower bound.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from prime_rl.configs.orchestrator import BalanceConfig, BalancePoolConfig
from prime_rl.orchestrator.concurrency import EngineLoadSample

MIN_STEPS = 3
"""Trainer steps the window must hold before the pools are measured."""

MIN_COVERAGE = 0.5
"""Share of the window the measured polls must cover."""

ENV_SHARE = 0.5
"""Share of the window's polls held at the configured cap while growth paid that makes the run environment-bound."""

SATURATED_BUSY = 0.75
"""Busy share from which an inference role's capacity is only a lower bound."""


@dataclass(frozen=True)
class InferencePoll:
    seconds: float
    finished: int
    """Train rollouts finished since the previous poll."""
    env_bound: bool
    busy: dict[str, float | None]
    """Per role (``inference``, or ``prefill`` and ``decode``), its engines' busy share; None when not all report it."""
    paused: bool = False
    """The engines were paused for a weight update."""
    generated: int = 0
    """Tokens the decode (or colocated) engines generated since the previous poll."""


@dataclass(frozen=True)
class TrainerStep:
    rollouts: int
    tokens: int
    """Tokens the trainer processes for the batch, padding included."""
    period: float
    """Seconds since the previous batch shipped."""
    busy: float | None
    """Seconds of ``period`` before the trainer requested this batch; None when the transport did not report it."""


@dataclass(frozen=True)
class Capacity:
    rate: float
    """Train rollouts per second the pool sustains busy all the time at its current size."""
    busy: float | None
    lower_bound: bool


@dataclass(frozen=True)
class Advice:
    capacities: dict[str, Capacity]
    current: dict[str, int]
    best: dict[str, int]
    speed: float
    best_speed: float
    bottleneck: str
    """The pool with the least capacity, or ``environment``."""
    trained_per_rollout: float
    demand: float | None
    """Tokens per second the engines must generate to keep the trainer busy all the time; None when its busy time
    is not reported."""

    @property
    def gain(self) -> float:
        return self.best_speed / self.speed if self.speed > 0 else 1.0


def inference_poll(
    samples: list[EngineLoadSample], seconds: float, *, finished: int, env_bound: bool, paused: bool = False
) -> InferencePoll:
    """Pool one metrics poll's engines per role."""
    by_role: dict[str, list[EngineLoadSample]] = {}
    for sample in samples:
        by_role.setdefault(sample.role or "inference", []).append(sample)
    busy: dict[str, float | None] = {}
    for role, group in by_role.items():
        reported = [s.busy_delta for s in group if s.busy_delta is not None]
        complete = len(reported) == len(group) and seconds > 0
        busy[role] = min(sum(reported) / (len(group) * seconds), 1.0) if complete else None
    # The orchestrator's pause spans a whole weight update, transfer included, while the engines stop only for the
    # swap itself: a poll in which an engine worked saw the engines running
    worked = any(s.busy_delta for s in samples)
    generated = sum(s.generation_tokens_delta for s in samples if s.role != "prefill")
    return InferencePoll(seconds, finished, env_bound, busy, paused=paused and not worked, generated=generated)


def best_split(pools: list[BalancePoolConfig], capacities: dict[str, Capacity]) -> tuple[float, dict[str, int], float]:
    """The current speed, and the fastest split of the same units into each pool's valid sizes with its speed. Ties
    keep the split closest to the current one."""
    current = {pool.name: pool.units for pool in pools}

    def speed(units: dict[str, int]) -> float:
        return min(capacities[p.name].rate * units[p.name] / p.units for p in pools)

    def splits(index: int, room: int) -> Iterator[dict[str, int]]:
        pool = pools[index]
        if index == len(pools) - 1:
            if room in pool.sizes:
                yield {pool.name: room}
            return
        for size in (s for s in pool.sizes if s < room):
            for rest in splits(index + 1, room - size):
                yield {pool.name: size, **rest}

    def score(units: dict[str, int]) -> tuple[float, int]:
        return round(speed(units), 9), -sum(abs(units[name] - current[name]) for name in units)

    best = max([current, *splits(0, sum(current.values()))], key=score)
    return speed(current), best, speed(best)


class BalanceAdvisor:
    def __init__(self, config: BalanceConfig, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.config = config
        self.clock = clock
        self.window_s = config.window_minutes * 60
        self.polls: deque[tuple[float, InferencePoll]] = deque()
        self.steps: deque[tuple[float, TrainerStep]] = deque()
        self.candidate: dict[str, int] | None = None
        self.candidate_since = 0.0
        self.advice: Advice | None = None

    def observe(self, poll: InferencePoll) -> None:
        self.polls.append((self.clock(), poll))

    def observe_step(self, step: TrainerStep) -> None:
        self.steps.append((self.clock(), step))

    def utilization(self, role: str) -> float | None:
        """The role's busy share over the window's polls that report it, outside weight-update pauses."""
        timed = [p for _, p in self.polls if p.seconds > 0 and not p.paused and p.busy.get(role) is not None]
        seconds = sum(p.seconds for p in timed)
        if seconds < MIN_COVERAGE * self.window_s:
            return None
        busy = sum(p.busy[role] * p.seconds for p in timed) / seconds
        return busy if busy > 0 else None

    def evaluate(self) -> Advice | None:
        """Measure every pool over the window and search the splits. None until the window holds enough."""
        now = self.clock()
        for entries in (self.polls, self.steps):
            while entries and now - entries[0][0] > self.window_s:
                entries.popleft()
        pools = self.config.pools or []
        steps = [s for _, s in self.steps]
        poll_seconds = sum(p.seconds for _, p in self.polls)
        if not pools or len(steps) < MIN_STEPS or poll_seconds < MIN_COVERAGE * self.window_s:
            return None
        rollouts, period = sum(s.rollouts for s in steps), sum(s.period for s in steps)
        # Rollouts the engines finished per second: the trainer's pace decouples from it while a backlog builds
        finished = sum(p.finished for _, p in self.polls) / poll_seconds
        if rollouts <= 0 or period <= 0 or finished <= 0:
            return None
        rate = rollouts / period
        busy = [s.busy for s in steps]
        trainer_busy = sum(busy) / period if None not in busy and sum(busy) > 0 else None
        capacities = {"trainer": Capacity(rate / (trainer_busy or 1.0), trainer_busy, trainer_busy is None)}
        for pool in pools:
            if pool.name != "trainer":
                share = self.utilization(pool.name)
                capacity = max(finished / share, rate) if share else rate
                capacities[pool.name] = Capacity(capacity, share, share is None or share >= SATURATED_BUSY)

        current = {pool.name: pool.units for pool in pools}
        speed, best, best_speed = best_split(pools, capacities)
        if sum(p.seconds for _, p in self.polls if p.env_bound) >= ENV_SHARE * poll_seconds:
            bottleneck, best, best_speed = "environment", current, speed
        else:
            # The slowest pool, preferring a measured one among those within 10% of it: a lower bound only says the
            # pool makes at least what it makes now
            slowest = min(c.rate for c in capacities.values())
            close = {name: c for name, c in capacities.items() if c.rate <= 1.1 * slowest}
            bottleneck = min(close, key=lambda name: (close[name].lower_bound, close[name].rate))
        trained = sum(s.tokens for s in steps) / rollouts
        trainer = capacities["trainer"]
        generated = sum(p.generated for _, p in self.polls) / (finished * poll_seconds)
        demand = None if trainer.lower_bound else trainer.rate * generated
        self.advice = Advice(capacities, current, best, speed, best_speed, bottleneck, trained, demand)
        return self.advice

    def notice(self, advice: Advice) -> str | None:
        """The notice to log each time a split has promised ``min_gain`` for another ``sustain_minutes``."""
        now = self.clock()
        if advice.best == advice.current or advice.gain < 1 + self.config.min_gain:
            self.candidate = None
            return None
        if advice.best != self.candidate:
            self.candidate, self.candidate_since = advice.best, now
        if now - self.candidate_since < self.config.sustain_minutes * 60:
            return None
        self.candidate_since = now
        moves = ", ".join(
            f"{name} {advice.current[name]} -> {units}"
            for name, units in advice.best.items()
            if units != advice.current[name]
        )
        bound = "at least " if any(c.lower_bound for c in advice.capacities.values()) else ""
        return (
            f"[balance] {advice.bottleneck.capitalize()}-bound: {moves} {self.config.unit}s would be {bound}"
            f"{advice.gain - 1:.0%} faster"
        )

    def gauges(self) -> dict[str, float]:
        advice = self.advice
        if advice is None:
            return {}
        gauges = {
            "balance/gain": advice.gain,
            "balance/environment/bottleneck": float(advice.bottleneck == "environment"),
        }
        for name, capacity in advice.capacities.items():
            gauges[f"balance/{name}/capacity"] = capacity.rate * advice.trained_per_rollout  # trained tok/s
            if capacity.busy is not None:
                gauges[f"balance/{name}/busy"] = capacity.busy
            gauges[f"balance/{name}/best_units"] = float(advice.best[name])
        return gauges
