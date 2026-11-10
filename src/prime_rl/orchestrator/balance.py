"""Trainer / inference GPU split advice.

An RL run is a pipeline of GPU pools: inference engines make rollouts (in P/D, a prefill pool feeds a decode pool)
and the trainer consumes them. The slowest pool sets the run's speed; the others idle part of the time. The advisor
measures what each pool sustains at its current size, in train rollouts per second, and searches every split of the
same units into valid pool sizes for the fastest, assuming each pool's capacity scales with its units:

    speed(split) = min over pools p of  capacity_p * units'_p / units_p

Per-token rates, context lengths, KV offload, Mooncake hit rates and P/D transfers all cancel: each pool's
capacity is measured end to end.

- **Trainer:** its busy time per step is the time from receiving one batch to requesting the next, so its capacity
  is the rollouts it trained over its busy time. Always measured.
- **Inference:** measured while inference is the bottleneck: the concurrency controller sits at its knee (growth no
  longer pays), the cap binds, and nothing on the trainer side holds admission back. Its tokens per second there,
  over the tokens one train rollout costs now, is its capacity. While the trainer is the bottleneck inference runs
  below its knee, so the last knee measurement stands in for it (up to ``knee_max_age_minutes``), else the current
  rate is only a lower bound. A probe (``probe``) lifts the trainer-side bounds until the knee is measured again.
- **Prefill / decode:** at the knee, the role whose engines queue more is the one that binds. The other one's slack
  is estimated: prefill engines are compute bound, so the share of polls they run a request is their busy share;
  decode engines' GPU KV usage bounds how many more sequences they could hold (no slack once full, which offload
  makes normal).
- **Environment:** a cap held at ``concurrency.max_inflight`` while growth still pays means the episodes' external
  resources (e.g. sandboxes) bound the run, not GPUs; the advisor says so instead of advising a split.

The advice is conservative: lower bounds count as the capacity, so a split that moves units off a pool whose
capacity is only bounded from below never looks better than it is.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from prime_rl.configs.orchestrator import BalanceConfig, BalancePoolConfig
from prime_rl.orchestrator.concurrency import EngineLoadSample
from prime_rl.utils.utils import format_num

MIN_STEPS = 3
"""Trainer steps the window must hold before a pool is measured."""

MIN_COVERAGE = 0.5
"""Share of the window the polls must cover before a pool is measured."""

KNEE_SHARE = 1 / 3
"""Share of the window's polls at the inference knee for inference to count as the bottleneck now."""

KNEE_MIN_SECONDS = 300.0
"""Seconds at the knee that make a knee measurement."""

ENV_SHARE = 0.5
"""Share of the window's polls held at the configured cap while growth paid that makes the run environment-bound."""

QUEUE_EPS = 0.02
"""Queued share below which a role is taken to have no queue."""

PREFILL_BUSY_FLOOR = 0.25
"""Least prefill busy share the slack estimate divides by (the slack is at most 1 / this)."""

DECODE_KV_FULL = 0.9
"""Decode GPU KV usage at which decode engines hold no more sequences."""

DECODE_SLACK_MAX = 2.0
"""Most decode slack the KV usage estimate claims."""

BOTTLENECK_TOLERANCE = 0.1
"""Pools within this share of the slowest capacity are candidates for the bottleneck."""

REMIND_SECONDS = 7200.0
"""Seconds before an unchanged notice is repeated."""


class Kind(Enum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    LOWER_BOUND = "lower bound"


@dataclass(frozen=True)
class InferencePoll:
    """One metrics poll of the inference engines, with the concurrency state at that time."""

    seconds: float
    generated: int
    """Tokens the decode (or colocated) engines generated since the previous poll."""
    finished: int
    """Train rollouts finished since the previous poll."""
    knee: bool
    """Inference at its knee: the controller's steps run, growth did not pay, the cap binds, and nothing on the
    trainer side holds admission back."""
    env_bound: bool
    """The cap sits at the configured ``max_inflight`` and binds while growth still pays."""
    queued_share: dict[str, float] = field(default_factory=dict)
    prefill_busy: float | None = None
    """Share of prefill engines running a request."""
    decode_kv: float | None = None
    """Mean GPU KV usage of the decode engines (0-1)."""


def inference_poll(
    samples: list[EngineLoadSample],
    seconds: float,
    *,
    finished: int,
    knee: bool,
    env_bound: bool,
    queued_share: dict[str, float],
) -> InferencePoll:
    """Pool one metrics poll's engines into an :class:`InferencePoll`."""
    prefill = [sample for sample in samples if sample.role == "prefill"]
    decode_kv = [sample.kv_usage for sample in samples if sample.role == "decode" and sample.kv_usage is not None]
    return InferencePoll(
        seconds=seconds,
        generated=sum(sample.generation_tokens_delta for sample in samples if sample.role != "prefill"),
        finished=finished,
        knee=knee,
        env_bound=env_bound,
        queued_share=dict(queued_share),
        prefill_busy=sum(sample.running > 0 for sample in prefill) / len(prefill) if prefill else None,
        decode_kv=sum(decode_kv) / len(decode_kv) if decode_kv else None,
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
    """Seconds of ``period`` the trainer spent before requesting this batch (training, broadcasting, saving). None
    when the transport did not report when the trainer asked."""


@dataclass(frozen=True)
class Capacity:
    rate: float
    """Train rollouts per second the pool sustains at its current size."""
    kind: Kind
    source: str


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
    trainer_busy: float | None
    trained_per_rollout: float
    """Tokens the trainer processes per rollout over the window: shown rates are trained tokens per second, which
    do not drift as rollouts grow longer. Rollouts are only the unit the pools are matched in."""

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

    def sizes(pool: BalancePoolConfig, room: int) -> range:
        first = math.ceil(pool.minimum / pool.step) * pool.step
        return range(first, room + 1, pool.step)

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
        for size in sizes(pool, room - rest):
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
        # The last knee measurement: generated tokens per second at the knee, and when
        self.knee_throughput: float | None = None
        self.knee_at: float | None = None
        self.knee_roles: dict[str, Capacity] = {}
        # The split promising ``min_gain`` and since when, and the last notice
        self.candidate: dict[str, int] | None = None
        self.candidate_since: float | None = None
        self.noticed: dict[str, int] | None = None
        self.noticed_at: float | None = None
        self.advice: Advice | None = None
        # Probe: when the current one started, and when the last one did
        self.probe_started: float | None = None
        self.last_probe: float | None = None

    @property
    def pools(self) -> list[BalancePoolConfig]:
        return self.config.pools or []

    @property
    def roles(self) -> list[str]:
        """The inference pools: ``inference``, or ``prefill`` and ``decode``."""
        return [pool.name for pool in self.pools if pool.name != "trainer"]

    # ── inbound ──────────────────────────────────────────────────────────────

    def observe(self, poll: InferencePoll) -> None:
        now = self.clock()
        self.polls.append((now, poll))
        self.trim(now)
        if not poll.knee:
            return
        # A probe measures the knee from its own polls only
        since = self.probe_started if self.probe_started is not None else -math.inf
        knee = [p for at, p in self.polls if p.knee and at >= since]
        seconds = sum(p.seconds for p in knee)
        if seconds >= KNEE_MIN_SECONDS:
            self.knee_throughput = sum(p.generated for p in knee) / seconds
            self.knee_at = now
            self.knee_roles = self.role_slack(knee)

    def observe_step(self, step: TrainerStep) -> None:
        now = self.clock()
        self.steps.append((now, step))
        self.trim(now)

    def trim(self, now: float) -> None:
        for entries in (self.polls, self.steps):
            while entries and now - entries[0][0] > self.window_s:
                entries.popleft()

    # ── measurement ──────────────────────────────────────────────────────────

    def role_slack(self, polls: list[InferencePoll]) -> dict[str, Capacity]:
        """Per P/D role, its capacity relative to the inference pipeline's at the knee (``rate`` is the factor)."""
        if "prefill" not in self.roles:
            return {}
        n = len(polls)
        queued = {role: sum(p.queued_share.get(role, 0.0) for p in polls) / n for role in ("prefill", "decode")}
        busy = [p.prefill_busy for p in polls if p.prefill_busy is not None]
        kv = [p.decode_kv for p in polls if p.decode_kv is not None]
        prefill_busy = sum(busy) / len(busy) if busy else None
        decode_kv = sum(kv) / len(kv) if kv else None
        if max(queued.values()) >= QUEUE_EPS:
            binding = max(queued, key=queued.get)
        else:
            # No queue anywhere: the knee is a throughput plateau, and busy prefill engines are its likeliest cause
            binding = "prefill" if prefill_busy is not None and prefill_busy >= 0.9 else "decode"
        slack = {binding: Capacity(1.0, Kind.MEASURED, f"{binding} queues at the knee")}
        if binding == "decode":
            if prefill_busy is None:
                slack["prefill"] = Capacity(1.0, Kind.LOWER_BOUND, "prefill busy share unknown")
            else:
                factor = 1 / max(prefill_busy, PREFILL_BUSY_FLOOR)
                slack["prefill"] = Capacity(max(factor, 1.0), Kind.ESTIMATED, f"prefill busy {prefill_busy:.0%}")
        elif decode_kv is None or decode_kv >= DECODE_KV_FULL:
            slack["decode"] = Capacity(1.0, Kind.LOWER_BOUND, "decode KV full or unknown")
        else:
            factor = min(DECODE_KV_FULL / max(decode_kv, 1e-3), DECODE_SLACK_MAX)
            slack["decode"] = Capacity(max(factor, 1.0), Kind.ESTIMATED, f"decode KV {decode_kv:.0%}")
        return slack

    def evaluate(self) -> Advice | None:
        """Measure every pool over the window and search the splits. None until the window holds enough."""
        now = self.clock()
        self.trim(now)
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
        # What a train rollout costs the engines: by the rollouts they finished, not the ones the trainer took, which
        # a stockpile (a probe, the startup) decouples. Eval traffic counts against it, which only underestimates
        # inference capacity.
        generated = sum(p.generated for _, p in self.polls)
        finished = sum(p.finished for _, p in self.polls)
        if generated <= 0 or finished <= 0:
            return None
        generated_per_rollout = generated / finished

        capacities: dict[str, Capacity] = {}
        busy = [s.busy for s in steps if s.busy is not None]
        if len(busy) == len(steps) and sum(busy) > 0:
            capacities["trainer"] = Capacity(rollouts / sum(busy), Kind.MEASURED, "busy time per step")
            trainer_busy = sum(busy) / period
        else:
            capacities["trainer"] = Capacity(rate, Kind.LOWER_BOUND, "trainer busy time not reported")
            trainer_busy = None

        polls = [p for _, p in self.polls]
        knee_seconds = sum(p.seconds for p in polls if p.knee)
        env_seconds = sum(p.seconds for p in polls if p.env_bound)
        fresh = self.knee_at is not None and now - self.knee_at <= self.config.knee_max_age_minutes * 60
        if knee_seconds >= KNEE_SHARE * poll_seconds and self.knee_throughput is not None:
            inference = Capacity(self.knee_throughput / generated_per_rollout, Kind.MEASURED, "at the knee now")
        elif fresh and self.knee_throughput is not None:
            age = (now - self.knee_at) / 60
            inference = Capacity(
                max(self.knee_throughput / generated_per_rollout, rate), Kind.ESTIMATED, f"knee {age:.0f} min ago"
            )
        else:
            inference = Capacity(rate, Kind.LOWER_BOUND, "below the knee, no recent knee measurement")

        if "inference" in self.roles:
            capacities["inference"] = inference
        else:
            for role in ("prefill", "decode"):
                slack = self.knee_roles.get(role) if fresh or inference.kind is Kind.MEASURED else None
                if slack is None or inference.kind is Kind.LOWER_BOUND:
                    capacities[role] = Capacity(inference.rate, Kind.LOWER_BOUND, inference.source)
                    continue
                kind = inference.kind if slack.kind is Kind.MEASURED else slack.kind
                capacities[role] = Capacity(inference.rate * slack.rate, kind, f"{inference.source}; {slack.source}")

        current, best = best_split(self.pools, capacities)
        if env_seconds >= ENV_SHARE * poll_seconds:
            bottleneck = "environment"
            best = current
        else:
            # The bottleneck is the pool that sustains no more than the run gets. Among pools that close to the
            # slowest, a measured one is it: a lower bound only says the pool makes at least what it makes now.
            slowest = min(c.rate for c in capacities.values())
            close = [name for name, c in capacities.items() if c.rate <= (1 + BOTTLENECK_TOLERANCE) * slowest]
            known = [name for name in close if capacities[name].kind is not Kind.LOWER_BOUND] or close
            bottleneck = min(known, key=lambda name: capacities[name].rate)
        self.advice = Advice(capacities, current, best, bottleneck, rate, trainer_busy, trained / rollouts)
        return self.advice

    # ── notice ───────────────────────────────────────────────────────────────

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
        return self.format(advice, saved=saved, restart_s=restart_s)

    def format(self, advice: Advice, *, saved: float | None, restart_s: float | None) -> str:
        unit = self.config.unit
        sustained = (self.clock() - self.candidate_since) / 60 if self.candidate_since is not None else 0.0
        bound = any(
            advice.capacities[name].kind is Kind.LOWER_BOUND
            for name in advice.best.units
            if advice.best.units[name] < advice.current.units[name]
        )
        lines = [
            f"[balance] {advice.bottleneck.capitalize()}-bound for the last {sustained:.0f} min"
            + (f" (trainer busy {advice.trainer_busy:.0%} of each step)." if advice.trainer_busy is not None else ".")
        ]

        def rate(rollouts_per_s: float) -> str:
            return f"{format_num(advice.tokens(rollouts_per_s), precision=1)} tok/s"

        for name, capacity in advice.capacities.items():
            lines.append(
                f"  {name:<9} capacity {rate(capacity.rate):>11} at {advice.current.units[name]} "
                f"{unit}s ({capacity.kind.value}: {capacity.source})"
            )

        def shape(split: Split) -> str:
            return " / ".join(f"{name} {split.units[name]}" for name in split.units) + f" {unit}s"

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
        resized = any(split.units[pool.name] != pool.units for pool in self.pools if pool.name in ("prefill", "decode"))
        if resized:
            per_replica = (split.units["prefill"] + split.units["decode"]) // self.config.infer_replicas
            flags.append(f"--deployment.num_infer_nodes {per_replica}")
        return flags

    # ── probe ────────────────────────────────────────────────────────────────

    def should_probe(self, advice: Advice | None) -> bool:
        """Probe while the trainer is the bottleneck and inference capacity is only bounded from below."""
        if not self.config.probe or self.probe_started is not None or advice is None:
            return False
        if advice.bottleneck != "trainer":
            return False
        if not any(advice.capacities[role].kind is Kind.LOWER_BOUND for role in self.roles):
            return False
        now = self.clock()
        return self.last_probe is None or now - self.last_probe >= self.config.probe_interval_minutes * 60

    def start_probe(self) -> None:
        self.probe_started = self.last_probe = self.clock()

    def probe_done(self) -> bool:
        """The probe measured the knee, or ran out of time."""
        if self.probe_started is None:
            return False
        measured = self.knee_at is not None and self.knee_at >= self.probe_started
        timed_out = self.clock() - self.probe_started >= self.config.probe_max_minutes * 60
        return measured or timed_out

    def end_probe(self) -> None:
        self.probe_started = None

    # ── observability ────────────────────────────────────────────────────────

    def gauges(self) -> dict[str, float]:
        gauges = {"balance/probing": float(self.probe_started is not None)}
        advice = self.advice
        if advice is None:
            return gauges
        # Trained tokens per second throughout
        gauges["balance/rate"] = advice.tokens(advice.rate)
        gauges["balance/speed"] = advice.tokens(advice.current.speed)
        gauges["balance/best_speed"] = advice.tokens(advice.best.speed)
        gauges["balance/gain"] = advice.gain
        if advice.trainer_busy is not None:
            gauges["balance/trainer_busy"] = advice.trainer_busy
        for name, capacity in advice.capacities.items():
            gauges[f"balance/{name}/capacity"] = advice.tokens(capacity.rate)
            gauges[f"balance/{name}/lower_bound"] = float(capacity.kind is Kind.LOWER_BOUND)
            gauges[f"balance/{name}/units"] = float(advice.current.units[name])
            gauges[f"balance/{name}/best_units"] = float(advice.best.units[name])
            gauges[f"balance/{name}/bottleneck"] = float(advice.bottleneck == name)
        gauges["balance/environment/bottleneck"] = float(advice.bottleneck == "environment")
        return gauges
