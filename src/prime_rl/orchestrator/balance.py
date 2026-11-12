"""Trainer / inference GPU split advice.

An RL run is a pipeline of GPU pools: inference engines make rollouts (in P/D, a prefill pool feeds a decode pool)
and the trainer consumes them. The slowest pool sets the run's speed while the others idle part of the time. Each
pool's capacity is its throughput over its busy share, what it would make busy all the time, assumed to scale with
its units; the advisor searches every valid split of the same units for the fastest:

    speed(split) = min over pools p of  capacity_p * units'_p / units_p

- **Trainer:** ``TrainerCapacity``, the rollouts it consumes per second of busy time.
- **Inference:** the seconds the engines spent executing model steps (``/prime_rl/engine_busy``), outside weight-update
  pauses, per role under P/D. Engines that step whenever they hold a sequence (decode) read as busy below their knee
  too: their spare capacity is in batch size, not time. For a role at least ``SATURATED_BUSY`` busy, the advisor fits
  how an engine's step time grows with what a step holds (sequences decoding, the context they read, prefill tokens)
  and extrapolates to the batch the engines can hold (``KV_TARGET`` of their KV cache, or ``max_num_seqs``). Without a
  reliable fit, and when busy time is not reported, a role's capacity is only a lower bound.
- **Environment:** a cap held at ``concurrency.max_inflight`` while growth still pays means the episodes' external
  resources bound the run, not GPUs; no split is advised.

A lower bound only understates a pool, so a speed computed with one is itself a lower bound.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np

from prime_rl.configs.orchestrator import BalanceConfig, BalancePoolConfig
from prime_rl.orchestrator.concurrency import EngineLoadSample
from prime_rl.orchestrator.trainer_capacity import TrainerCapacity

MIN_COVERAGE = 0.5
"""Share of the window the measured polls must cover."""

ENV_SHARE = 0.5
"""Share of the window's polls held at the configured cap while growth paid that makes the run environment-bound."""

SATURATED_BUSY = 0.75
"""Busy share from which an inference role's spare capacity is read from its batch, not its busy time."""

MIN_FIT_POINTS = 30
"""Engine polls the step-time fit needs."""

MIN_FIT_R2 = 0.5
"""Variance in per-step time the fit must explain to be trusted."""

MIN_BATCH_SPREAD = 0.1
"""Coefficient of variation the per-step batch must show across polls for the fit to see how step time grows."""

KV_TARGET = 0.6
"""Share of the GPU KV cache a full batch may fill. Calibrated on Qwen3-30B-A3B DP4xEP4 kv-stress (job 8435): pinned at
512 in flight, ~500 running at 61% KV made the most tokens per second (15.0K); at 1024, ~600 running filled the KV and
thrashed down to 8.6K. Growing sequences and prefix-cache retention leave less headroom than the KV size suggests."""

MAX_BATCH_GAIN = 4.0
"""Most throughput gain the batch extrapolation claims."""


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
    engines: dict[str, list[EngineSteps]] = field(default_factory=dict)
    """Per role, each engine's (or lockstep group's) steps over the poll."""


@dataclass(frozen=True)
class EngineSteps:
    """One engine's (or lockstep group's) steps over one poll: their count and seconds, summed over them the sequences
    decoding, the context they read and the prefill tokens, its ``max_num_seqs`` and its KV cache usage."""

    steps: float
    busy: float
    decode_seqs: float
    kv_read: float
    prefill: float
    max_num_seqs: int | None
    kv_usage: float | None


@dataclass(frozen=True)
class BatchFit:
    gain: float
    """Throughput at the full batch over throughput at the current one."""
    batch: float
    full_batch: float


@dataclass(frozen=True)
class Capacity:
    rate: float
    """Train rollouts per second the pool sustains busy all the time at its current size."""
    busy: float | None
    lower_bound: bool
    batch: BatchFit | None = None


@dataclass(frozen=True)
class Advice:
    capacities: dict[str, Capacity]
    current: dict[str, int]
    best: dict[str, int]
    speed: float
    best_speed: float
    bottleneck: str
    """The pool with the least capacity, or ``environment``."""

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
    engines = {
        role: _engine_steps([s for s in group if s.steps_delta and s.busy_delta]) for role, group in by_role.items()
    }
    return InferencePoll(seconds, finished, env_bound, busy, paused=paused and not worked, engines=engines)


def _engine_steps(samples: list[EngineLoadSample]) -> list[EngineSteps]:
    """One point per engine for the batch fit, and one per lockstep group: data-parallel MoE ranks step together, so
    the group's step time follows the batch of the whole group, and it holds the KV and sequences of all its ranks."""
    groups: dict[str | None, list[EngineLoadSample]] = {}
    for sample in samples:
        groups.setdefault(sample.lockstep_group, []).append(sample)
    points = []
    for engines in [[sample] for sample in groups.pop(None, [])] + list(groups.values()):
        seqs = [s.max_num_seqs for s in engines]
        usage = [s.kv_usage for s in engines]
        points.append(
            EngineSteps(
                steps=sum(s.steps_delta for s in engines) / len(engines),
                busy=sum(s.busy_delta for s in engines) / len(engines),
                decode_seqs=sum(s.decode_seqs_delta or 0.0 for s in engines),
                kv_read=sum(s.kv_read_delta or 0.0 for s in engines),
                prefill=sum(s.prefill_delta or 0.0 for s in engines),
                max_num_seqs=sum(seqs) if None not in seqs else None,
                kv_usage=sum(usage) / len(usage) if None not in usage else None,
            )
        )
    return points


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
    def __init__(
        self, config: BalanceConfig, trainer: TrainerCapacity, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.config = config
        self.trainer = trainer
        self.clock = clock
        self.window_s = config.window_minutes * 60
        self.polls: deque[tuple[float, InferencePoll]] = deque()
        self.candidate: dict[str, int] | None = None
        self.candidate_since = 0.0
        self.advice: Advice | None = None

    def observe(self, poll: InferencePoll) -> None:
        self.polls.append((self.clock(), poll))

    def utilization(self, role: str) -> float | None:
        """The role's busy share over the window's polls that report it, outside weight-update pauses."""
        timed = [p for _, p in self.polls if p.seconds > 0 and not p.paused and p.busy.get(role) is not None]
        seconds = sum(p.seconds for p in timed)
        if seconds < MIN_COVERAGE * self.window_s:
            return None
        busy = sum(p.busy[role] * p.seconds for p in timed) / seconds
        return busy if busy > 0 else None

    def batch_fit(self, role: str) -> BatchFit | None:
        """Fit per-step time over the window's engine polls as a fixed cost plus a cost per sequence decoding, per
        context token read and per prefill token, and extrapolate to the batch the engines could hold: ``KV_TARGET`` of
        their KV cache at the current KV use per sequence, or ``max_num_seqs``. None without a reliable fit."""
        points = [e for _, p in self.polls if not p.paused for e in p.engines.get(role, []) if e.decode_seqs > 0]
        if len(points) < MIN_FIT_POINTS:
            return None
        features = np.array([[1.0, e.decode_seqs / e.steps, e.kv_read / e.steps, e.prefill / e.steps] for e in points])
        step_time = np.array([e.busy / e.steps for e in points])
        batch = features[:, 1]
        if batch.std() < MIN_BATCH_SPREAD * batch.mean():
            return None
        # Costs are not negative: drop a term that fits negative and refit without it
        columns = list(range(4))
        while True:
            coef, *_ = np.linalg.lstsq(features[:, columns], step_time, rcond=None)
            negative = [column for column, value in zip(columns, coef) if value < 0 and column != 0]
            if not negative:
                break
            columns.remove(negative[0])
        residual = step_time - features[:, columns] @ coef
        if 1 - residual.var() / step_time.var() < MIN_FIT_R2 or coef[0] <= 0:
            return None
        cost = np.zeros(4)
        cost[columns] = coef
        seqs = sum(e.decode_seqs for e in points)
        batch_now = seqs / sum(e.steps for e in points)
        kv_per_seq = sum(e.kv_read for e in points) / seqs
        prefill_per_seq = sum(e.prefill for e in points) / seqs
        limits = [e.max_num_seqs for e in points if e.max_num_seqs]
        usage = [e.kv_usage for e in points if e.kv_usage]
        if usage:
            limits.append(batch_now * KV_TARGET / (sum(usage) / len(usage)))
        if not limits:
            return None
        full = max(min(limits), batch_now)

        def throughput(seqs_per_step: float) -> float:
            step = cost @ np.array([1.0, seqs_per_step, kv_per_seq * seqs_per_step, prefill_per_seq * seqs_per_step])
            return seqs_per_step / step

        gain = min(max(throughput(full) / throughput(batch_now), 1.0), MAX_BATCH_GAIN)
        return BatchFit(gain=float(gain), batch=float(batch_now), full_batch=float(full))

    def evaluate(self) -> Advice | None:
        """Measure every pool over the window and search the splits. None until the window holds enough."""
        now = self.clock()
        while self.polls and now - self.polls[0][0] > self.window_s:
            self.polls.popleft()
        pools, trainer = self.config.pools or [], self.trainer
        poll_seconds = sum(p.seconds for _, p in self.polls)
        if not pools or trainer.rate is None or poll_seconds < MIN_COVERAGE * self.window_s:
            return None
        # What the run ships per second, and the rollouts the engines finished per second: the trainer's pace
        # decouples from it while a backlog builds
        rate = sum(s.rollouts for s in trainer.steps) / sum(s.period for s in trainer.steps)
        finished = sum(p.finished for _, p in self.polls) / poll_seconds
        if finished <= 0:
            return None
        capacities = {"trainer": Capacity(trainer.rate, trainer.busy, False)}
        for pool in pools:
            if pool.name != "trainer":
                share = self.utilization(pool.name)
                fit = self.batch_fit(pool.name) if share and share >= SATURATED_BUSY else None
                capacity = max(finished / share * (fit.gain if fit else 1.0), rate) if share else rate
                lower = share is None or (share >= SATURATED_BUSY and fit is None)
                capacities[pool.name] = Capacity(capacity, share, lower, fit)

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
        self.advice = Advice(capacities, current, best, speed, best_speed, bottleneck)
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
            gauges[f"balance/{name}/capacity"] = capacity.rate  # train rollouts per second
            if capacity.busy is not None:
                gauges[f"balance/{name}/busy"] = capacity.busy
            if capacity.batch is not None:
                gauges[f"balance/{name}/batch_gain"] = capacity.batch.gain
            gauges[f"balance/{name}/best_units"] = float(advice.best[name])
        return gauges
