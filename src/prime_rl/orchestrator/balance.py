"""Trainer / inference GPU split advice.

An RL run is a pipeline of GPU pools: inference engines make rollouts (in P/D, a prefill pool feeds a decode pool)
and the trainer consumes them. The slowest pool sets the run's speed; the others idle part of the time. The advisor
measures what each pool sustains at its current size and searches every split of the same units into valid pool sizes
for the fastest, assuming each pool's capacity scales with its units:

    speed(split) = min over pools p of  capacity_p * units'_p / units_p

Every pool's capacity is its throughput over its busy share: what it would make busy all the time. Context lengths,
KV offload, Mooncake and P/D transfers cancel, since each pool is measured end to end.

- **Trainer:** busy from receiving a batch to requesting the next (training, broadcasting, saving).
- **Inference:** the seconds the engines spent executing model steps (``/prime_rl/engine_busy``), outside weight-update
  pauses. Under P/D, prefill and decode engines are measured apart. Engines that step whenever they hold a sequence
  (decode) read as fully busy below their knee too: their spare capacity is in batch size, not time. For a role at
  least ``SATURATED_BUSY`` busy, the advisor fits how an engine's step time grows with what the step holds (sequences
  decoding, the context they read, prefill tokens) and extrapolates to the batch the engine could hold (its KV cache
  or ``max_num_seqs``); without a reliable fit the busy-time estimate stands, which only underestimates.
- **Environment:** a cap held at ``concurrency.max_inflight`` while growth still pays means the episodes' external
  resources (e.g. sandboxes) bound the run, not GPUs; the advisor says so instead of advising a split.

Capacities are matched per train rollout, the unit both sides share, and shown as trained tokens per second. A pool
whose busy time is not reported counts only what it makes now, a lower bound: a split that takes units from it never
looks better than it is.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from prime_rl.configs.orchestrator import BalanceConfig, BalancePoolConfig
from prime_rl.orchestrator.concurrency import EngineLoadSample
from prime_rl.utils.utils import format_num

MIN_STEPS = 3
"""Trainer steps the window must hold before a pool is measured."""

MIN_COVERAGE = 0.5
"""Share of the window the polls must cover before a pool is measured."""

ENV_SHARE = 0.5
"""Share of the window's polls held at the configured cap while growth paid that makes the run environment-bound."""

BOTTLENECK_TOLERANCE = 0.1
"""Pools within this share of the slowest capacity are candidates for the bottleneck."""

REMIND_SECONDS = 7200.0
"""Seconds before an unchanged notice is repeated."""

SATURATED_BUSY = 0.75
"""Busy share from which an engine's spare capacity is read from its batch, not its busy time."""

MIN_FIT_POINTS = 30
"""Engine polls the step-time fit needs."""

MIN_FIT_R2 = 0.5
"""Variance in per-step time the fit must explain to be trusted."""

MIN_BATCH_SPREAD = 0.1
"""Coefficient of variation the per-step batch must show across polls for the fit to see how step time grows."""

KV_HEADROOM = 0.95
"""Share of GPU KV cache a full batch may fill."""

MAX_BATCH_GAIN = 4.0
"""Most throughput gain the batch extrapolation claims."""


@dataclass(frozen=True)
class EngineSteps:
    """One engine's steps over one poll: their count and seconds, and summed over them, the sequences decoding, the
    context they read and the prefill tokens."""

    steps: float
    busy: float
    decode_seqs: float
    kv_read: float
    prefill: float
    max_num_seqs: int | None
    kv_capacity: int | None


@dataclass(frozen=True)
class BatchFit:
    """How a role's engines would grow their throughput with a full batch: the fitted step time per sequence decoding,
    per context token read and per prefill token, over a fixed cost per step."""

    gain: float
    batch: float
    full_batch: float


@dataclass(frozen=True)
class InferencePoll:
    """One metrics poll of the inference engines."""

    seconds: float
    generated: int
    """Tokens the decode (or colocated) engines generated since the previous poll."""
    finished: int
    """Train rollouts finished since the previous poll."""
    env_bound: bool
    """The cap sits at the configured ``max_inflight`` and binds while growth still pays."""
    busy: dict[str, float | None]
    """Per inference role (``inference``, or ``prefill`` and ``decode``), the share of the poll its engines spent
    executing model steps; None when not every engine of the role reported it."""
    engines: dict[str, list[EngineSteps]] = field(default_factory=dict)
    """Per inference role, each engine's steps over the poll."""
    paused: bool = False
    """The engines were paused for a weight update."""


def inference_poll(
    samples: list[EngineLoadSample], seconds: float, *, finished: int, env_bound: bool, paused: bool = False
) -> InferencePoll:
    """Pool one metrics poll's engines into an :class:`InferencePoll`."""
    by_role: dict[str, list[EngineLoadSample]] = {}
    for sample in samples:
        by_role.setdefault(sample.role or "inference", []).append(sample)
    busy: dict[str, float | None] = {}
    engines: dict[str, list[EngineSteps]] = {}
    for role, group in by_role.items():
        reported = [sample.busy_delta for sample in group if sample.busy_delta is not None]
        complete = len(reported) == len(group) and seconds > 0
        busy[role] = min(sum(reported) / (len(group) * seconds), 1.0) if complete else None
        engines[role] = [
            EngineSteps(
                steps=sample.steps_delta,
                busy=sample.busy_delta,
                decode_seqs=sample.decode_seqs_delta or 0.0,
                kv_read=sample.kv_read_delta or 0.0,
                prefill=sample.prefill_delta or 0.0,
                max_num_seqs=sample.max_num_seqs,
                kv_capacity=sample.kv_capacity_tokens,
            )
            for sample in group
            if sample.steps_delta and sample.busy_delta
        ]
    return InferencePoll(
        seconds=seconds,
        generated=sum(sample.generation_tokens_delta for sample in samples if sample.role != "prefill"),
        finished=finished,
        env_bound=env_bound,
        busy=busy,
        engines=engines,
        paused=paused,
    )


@dataclass(frozen=True)
class TrainerStep:
    """One shipped batch."""

    rollouts: int
    tokens: int
    """Tokens the trainer processes for the batch, padding included."""
    period: float
    """Seconds since the previous batch shipped."""
    busy: float | None
    """Seconds of ``period`` the trainer spent before requesting this batch. None when the transport did not report
    when the trainer asked."""


@dataclass(frozen=True)
class Capacity:
    rate: float
    """Train rollouts per second the pool sustains at its current size."""
    busy: float | None
    """The pool's busy share; None when not reported, and ``rate`` is then only a lower bound."""
    batch: BatchFit | None = None
    """The batch extrapolation the rate includes, for a saturated inference role."""

    @property
    def lower_bound(self) -> bool:
        return self.busy is None


@dataclass(frozen=True)
class Split:
    units: dict[str, int]
    speed: float


@dataclass(frozen=True)
class Advice:
    capacities: dict[str, Capacity]
    current: Split
    best: Split
    bottleneck: str
    """The pool with the least capacity, or ``environment``."""
    rate: float
    """Train rollouts per second shipped over the window."""
    trained_per_rollout: float
    """Tokens the trainer processes per rollout over the window, which turns rates into trained tokens per second."""

    def tokens(self, rate: float) -> float:
        """A rate in rollouts per second as trained tokens per second."""
        return rate * self.trained_per_rollout

    @property
    def gain(self) -> float:
        return self.best.speed / self.current.speed if self.current.speed > 0 else 1.0

    @property
    def changed(self) -> bool:
        return self.best.units != self.current.units


def best_split(pools: list[BalancePoolConfig], capacities: dict[str, Capacity]) -> tuple[Split, Split]:
    """The current split and the fastest one of the same units. Speed is monotone in every pool's units, so the last
    pool takes the units the others leave; ties keep the split closest to the current one."""
    total = sum(pool.units for pool in pools)

    def speed(units: dict[str, int]) -> float:
        return min(capacities[pool.name].rate * units[pool.name] / pool.units for pool in pools)

    current_units = {pool.name: pool.units for pool in pools}
    current = Split(current_units, speed(current_units))
    best = current
    best_distance = 0

    def search(index: int, room: int, units: dict[str, int]) -> None:
        nonlocal best, best_distance
        pool = pools[index]
        if index == len(pools) - 1:
            size = room // pool.step * pool.step
            if size < pool.minimum:
                return
            candidate = {**units, pool.name: size}
            value = speed(candidate)
            distance = sum(abs(candidate[name] - current_units[name]) for name in candidate)
            # A split must be clearly faster than the best so far, not faster by rounding
            if value > best.speed * (1 + 1e-9) or (math.isclose(value, best.speed) and distance < best_distance):
                best, best_distance = Split(candidate, value), distance
            return
        rest = sum(math.ceil(p.minimum / p.step) * p.step for p in pools[index + 1 :])
        first = math.ceil(pool.minimum / pool.step) * pool.step
        for size in range(first, room - rest + 1, pool.step):
            search(index + 1, room - size, {**units, pool.name: size})

    search(0, total, {})
    return current, best


class BalanceAdvisor:
    def __init__(self, config: BalanceConfig, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.config = config
        self.clock = clock
        self.window_s = config.window_minutes * 60
        self.polls: deque[tuple[float, InferencePoll]] = deque()
        self.steps: deque[tuple[float, TrainerStep]] = deque()
        # The split promising ``min_gain`` and since when, and the last notice
        self.candidate: dict[str, int] | None = None
        self.candidate_since: float | None = None
        self.noticed: dict[str, int] | None = None
        self.noticed_at: float | None = None
        self.advice: Advice | None = None

    @property
    def pools(self) -> list[BalancePoolConfig]:
        return self.config.pools or []

    def observe(self, poll: InferencePoll) -> None:
        self.polls.append((self.clock(), poll))
        self.trim()

    def observe_step(self, step: TrainerStep) -> None:
        self.steps.append((self.clock(), step))
        self.trim()

    def trim(self) -> None:
        now = self.clock()
        for entries in (self.polls, self.steps):
            while entries and now - entries[0][0] > self.window_s:
                entries.popleft()

    def utilization(self, role: str) -> float | None:
        """The role's busy share over the window's polls that report it, outside weight-update pauses. None when they
        cover too little of the window."""
        timed = [p for _, p in self.polls if p.seconds > 0 and not p.paused and p.busy.get(role) is not None]
        if sum(p.seconds for p in timed) < MIN_COVERAGE * self.window_s:
            return None
        busy = sum(p.busy[role] * p.seconds for p in timed) / sum(p.seconds for p in timed)
        return busy if busy > 0 else None

    def batch_fit(self, role: str) -> BatchFit | None:
        """Fit per-step time over the window's engine polls as a fixed cost plus a cost per sequence decoding, per
        context token read and per prefill token, and extrapolate to the batch the engines could hold: their KV cache
        over the context per sequence, or ``max_num_seqs``. None without a reliable fit."""
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
        steps = sum(e.steps for e in points)
        kv_per_seq = sum(e.kv_read for e in points) / seqs
        prefill_per_seq = sum(e.prefill for e in points) / seqs
        limits = [e.max_num_seqs for e in points if e.max_num_seqs]
        limits += [KV_HEADROOM * e.kv_capacity / kv_per_seq for e in points if e.kv_capacity and kv_per_seq > 0]
        if not limits:
            return None
        batch_now = seqs / steps
        full = max(min(limits), batch_now)

        def throughput(seqs_per_step: float) -> float:
            step = cost @ np.array([1.0, seqs_per_step, kv_per_seq * seqs_per_step, prefill_per_seq * seqs_per_step])
            return seqs_per_step / step

        gain = min(max(throughput(full) / throughput(batch_now), 1.0), MAX_BATCH_GAIN)
        return BatchFit(gain=float(gain), batch=float(batch_now), full_batch=float(full))

    def evaluate(self) -> Advice | None:
        """Measure every pool over the window and search the splits. None until the window holds enough."""
        self.trim()
        if not self.pools or len(self.steps) < MIN_STEPS or not self.polls:
            return None
        poll_seconds = sum(p.seconds for _, p in self.polls)
        if poll_seconds < MIN_COVERAGE * self.window_s:
            return None
        steps = [s for _, s in self.steps]
        rollouts = sum(s.rollouts for s in steps)
        trained = sum(s.tokens for s in steps)
        period = sum(s.period for s in steps)
        if rollouts <= 0 or trained <= 0 or period <= 0:
            return None
        rate = rollouts / period
        # Rollouts the engines finished per second: the trainer's pace decouples from it while a backlog builds or
        # drains. Weight-update pauses are part of every step, so they stay in.
        finished = sum(p.finished for _, p in self.polls) / poll_seconds
        if finished <= 0:
            return None

        busy = [s.busy for s in steps]
        trainer_busy = sum(busy) / period if None not in busy and sum(busy) > 0 else None
        capacities = {"trainer": Capacity(rate / trainer_busy if trainer_busy else rate, trainer_busy)}
        for pool in self.pools:
            if pool.name != "trainer":
                utilization = self.utilization(pool.name)
                if utilization is None:
                    capacities[pool.name] = Capacity(rate, None)
                    continue
                fit = self.batch_fit(pool.name) if utilization >= SATURATED_BUSY else None
                capacity = finished / utilization * (fit.gain if fit else 1.0)
                capacities[pool.name] = Capacity(max(capacity, rate), utilization, fit)

        current, best = best_split(self.pools, capacities)
        if sum(p.seconds for _, p in self.polls if p.env_bound) >= ENV_SHARE * poll_seconds:
            bottleneck = "environment"
            best = current
        else:
            # The bottleneck is the pool that sustains no more than the run gets. Among pools that close to the
            # slowest, a measured one is it: a lower bound only says the pool makes at least what it makes now.
            slowest = min(c.rate for c in capacities.values())
            close = [name for name, c in capacities.items() if c.rate <= (1 + BOTTLENECK_TOLERANCE) * slowest]
            known = [name for name in close if not capacities[name].lower_bound] or close
            bottleneck = min(known, key=lambda name: capacities[name].rate)
        self.advice = Advice(capacities, current, best, bottleneck, rate, trained / rollouts)
        return self.advice

    def notice(self, advice: Advice, *, remaining_s: float | None = None, restart_s: float | None = None) -> str | None:
        """The notice to log, once a split has promised ``min_gain`` for ``sustain_minutes`` (again after
        ``REMIND_SECONDS``, or when the advised split changes). None otherwise."""
        now = self.clock()
        if not advice.changed or advice.gain < 1 + self.config.min_gain:
            self.candidate = self.candidate_since = None
            return None
        if advice.best.units != self.candidate:
            self.candidate, self.candidate_since = advice.best.units, now
        if now - self.candidate_since < self.config.sustain_minutes * 60:
            return None
        if self.noticed == advice.best.units and now - self.noticed_at < REMIND_SECONDS:
            return None
        saved = remaining_s * (1 - 1 / advice.gain) if remaining_s is not None else None
        if saved is not None and restart_s is not None and saved <= restart_s:
            return None
        self.noticed, self.noticed_at = advice.best.units, now
        return self.format(advice, saved=saved, restart_s=restart_s, sustained=(now - self.candidate_since) / 60)

    def format(self, advice: Advice, *, saved: float | None, restart_s: float | None, sustained: float) -> str:
        unit = self.config.unit

        def rate(rollouts_per_s: float) -> str:
            return f"{format_num(advice.tokens(rollouts_per_s), precision=1)} tok/s"

        def shape(split: Split) -> str:
            return " / ".join(f"{name} {split.units[name]}" for name in split.units) + f" {unit}s"

        bound = any(
            advice.capacities[name].lower_bound
            for name in advice.best.units
            if advice.best.units[name] < advice.current.units[name]
        )
        lines = [f"[balance] {advice.bottleneck.capitalize()}-bound for the last {sustained:.0f} min."]
        for name, capacity in advice.capacities.items():
            busy = "busy time not reported" if capacity.lower_bound else f"busy {capacity.busy:.0%}"
            if capacity.batch is not None:
                busy += f", batch {capacity.batch.batch:.0f} -> {capacity.batch.full_batch:.0f} seqs"
            lines.append(
                f"  {name:<9} capacity {rate(capacity.rate):>11} at {advice.current.units[name]} {unit}s ({busy})"
            )
        lines.append(f"  current      {shape(advice.current)}  {rate(advice.current.speed)}")
        lines.append(
            f"  recommended  {shape(advice.best)}  {rate(advice.best.speed)} "
            f"({'at least ' if bound else ''}+{advice.gain - 1:.0%})"
        )
        if saved is not None:
            cost = f"restart ~{restart_s / 60:.0f} min, " if restart_s is not None else ""
            lines.append(
                f"  {cost}saves ~{saved / 3600:.1f} h of the ~{saved / (1 - 1 / advice.gain) / 3600:.1f} h left"
            )
        lines.append("  To apply, stop the run and resume it with: " + " ".join(self.overrides(advice.best)))
        return "\n".join(lines)

    def overrides(self, split: Split) -> list[str]:
        """The CLI overrides that set every resized pool."""
        flags = []
        for pool in self.pools:
            units = split.units[pool.name]
            if units == pool.units:
                continue
            for key, per_step in zip(pool.keys, pool.per_step):
                flags.append(f"--{key} {units // pool.step * per_step}")
        # A P/D deployment's inference nodes per replica follow from its prefill and decode replicas
        if any(split.units[pool.name] != pool.units for pool in self.pools if pool.name in ("prefill", "decode")):
            per_replica = (split.units["prefill"] + split.units["decode"]) // self.config.infer_replicas
            flags.append(f"--deployment.num_infer_nodes {per_replica}")
        return flags

    def gauges(self) -> dict[str, float]:
        advice = self.advice
        if advice is None:
            return {}
        # Trained tokens per second throughout
        gauges = {
            "balance/rate": advice.tokens(advice.rate),
            "balance/speed": advice.tokens(advice.current.speed),
            "balance/best_speed": advice.tokens(advice.best.speed),
            "balance/gain": advice.gain,
            "balance/environment/bottleneck": float(advice.bottleneck == "environment"),
        }
        for name, capacity in advice.capacities.items():
            gauges[f"balance/{name}/capacity"] = advice.tokens(capacity.rate)
            if capacity.busy is not None:
                gauges[f"balance/{name}/busy"] = capacity.busy
            if capacity.batch is not None:
                gauges[f"balance/{name}/batch_gain"] = capacity.batch.gain
            gauges[f"balance/{name}/units"] = float(advice.current.units[name])
            gauges[f"balance/{name}/best_units"] = float(advice.best.units[name])
            gauges[f"balance/{name}/bottleneck"] = float(advice.bottleneck == name)
        return gauges
