"""GoodputController: in-flight cap that maximizes accepted tokens/s.

Treats the inference engines as a black box. The objective is

    J(P) = T(P) * eta(P)

- ``T(P)``: engine generation tokens/s at pool size ``P`` (in-flight
  episodes), *measured* per pool-size bin from ``vllm:generation_tokens``.
  Nothing about KV, MoE or attention structure is assumed.
- ``eta(P)``: fraction of generated tokens whose group trains fresh,
  *computed* from the group length law. By Little's law each in-flight
  episode progresses at ``T / P`` tokens/s; a group is dropped about
  ``K`` trainer steps after dispatch, and with inference as the bottleneck a
  step takes ``(B / G) * C(x) / (T * S(x))`` seconds. So a group trains iff
  its longest member ``M <= x``, with

      x * P * S(x) = K' * (B / G) * C(x)          (T cancels)
      S(x) = P(M <= x),  C(x) = E[sum_i min(L_i, x)],
      eta  = E[sum_i L_i * w(K' M / x) * 1{M <= x}] / C(x)

  ``w`` optionally discounts staler groups (``staleness_scale``). Goodput
  alone is unbounded when throughput scales: more in-flight work always adds
  *some* accepted tokens, while the batch skews to short groups (long ones
  age out). So bins with ``eta < min_fresh_fraction`` are infeasible. The law is
  a Kaplan-Meier estimate over train groups; dropped groups are censored at
  their estimated progress (a virtual per-episode token clock).

The cap hill-climbs over a geometric grid of pool sizes: hold a bin for about
one episode lifetime, measure ``T``, then move to the best of one bin down /
stay / up (unvisited bins scored optimistically; up-moves accelerate while
throughput scales near-linearly). Bin estimates are forgetful and carry an
uncertainty bonus, so neighbours are re-probed as the workload drifts. Fast
rail: a probe that collapses throughput, or sustained preemptions, reverts
immediately. If the pool does not fill the cap (dispatch gated by the
trainer, or a starved source), the cap never grows.

Same hooks as the legacy controller (``bind`` / ``observe`` /
``record_episode`` / ``gauges``), plus group events from the dispatcher.
"""

from __future__ import annotations

import math
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from prime_rl.configs.orchestrator import ConcurrencyConfig
from prime_rl.utils.logger import get_logger

BIN_RATIO = 2**0.25
"""Pool sizes form a geometric grid: one bin is ~19% more in-flight work."""

MIN_SETTLE_S = 30.0
MIN_MEASURE_S = 30.0
MAX_PHASE_S = 900.0
SETTLE_LIFETIMES = 1.0
"""Hold a new bin this many episode lifetimes before measuring: the pool's
age mix (context lengths, cache residency) needs a turnover to reflect it."""

MEASURE_LIFETIMES = 0.5

UCB = 1.0
"""Uncertainty bonus in standard errors on a bin's throughput estimate."""

BIN_MEMORY = 4.0
"""Effective sample count cap per bin, so estimates track drift."""

ABORT_RATIO = 0.7
"""A probe whose goodput falls below this fraction of the incumbent's for
``ABORT_POLLS`` consecutive polls reverts without waiting for the window."""

ABORT_POLLS = 3

PENALTY = 0.5
PENALTY_TTL_S = 1800.0
"""Bins that preempted or aborted score at ``PENALTY`` for this long."""

PREEMPTION_POLLS = 2
"""Consecutive polls with preemptions before stepping down."""

BINDING_FRACTION = 0.9
"""The cap binds when the measured pool fills this fraction of it."""

GROUP_WINDOW = 512
"""Train groups kept in the length law."""

MIN_GROUPS = 8
"""Below this many groups, eta is taken as 1."""

MAX_CLIMB_BINS = 4
"""Slow-start: consecutive near-linear gains double the up-step, up to 2x pool."""

LINEAR_EFFICIENCY = 0.5
"""An up-move keeps the slow-start going if dlogT/dlogP reaches this."""


def bin_size(i: int) -> int:
    return max(1, int(round(BIN_RATIO**i)))


def bin_of(p: float) -> int:
    return int(round(math.log(max(p, 1.0)) / math.log(BIN_RATIO)))


@dataclass
class _Group:
    size: int
    started_v: float
    lengths: list[float] = field(default_factory=list)
    finished_v: list[float] = field(default_factory=list)


@dataclass
class _Bin:
    mean: float
    var: float
    n: float


class GroupDurationLaw:
    """Kaplan-Meier law of a train group's *virtual duration*: how far the
    average in-flight episode's token clock advanced between the group's
    dispatch and its last member finishing. Measuring groups on that clock
    (rather than by token count at an assumed pace) folds in everything that
    makes some groups slower than others — tool time, turn structure,
    queueing — and makes the censoring point of a dropped group exact.

    Each group keeps its members' ``(tokens, virtual finish time)``; a member
    still running at virtual time ``x`` is costed pro rata. Dropped groups are
    right-censored at their virtual age; KM's mass beyond the last completed
    group is costed like those dropped groups (finished members as observed,
    the rest running to the cutoff)."""

    def __init__(self, window: int = GROUP_WINDOW) -> None:
        # (virtual duration or age, completed, member tokens, member virtual finish, unfinished count)
        self.obs: deque[tuple[float, bool, np.ndarray, np.ndarray, int]] = deque(maxlen=window)
        self.cache: tuple | None = None
        self.version = 0

    def add_complete(self, tokens: list[float], vtimes: list[float]) -> None:
        tok, vt = np.asarray(tokens, dtype=float), np.maximum(np.asarray(vtimes, dtype=float), 1e-9)
        self.append((float(vt.max()), True, tok, vt, 0))

    def add_censored(self, age: float, size: int, tokens: list[float], vtimes: list[float]) -> None:
        tok, vt = np.asarray(tokens, dtype=float), np.maximum(np.asarray(vtimes, dtype=float), 1e-9)
        self.append((float(age), False, tok, vt, max(size - len(tok), 1)))

    def append(self, obs) -> None:
        self.obs.append(obs)
        self.cache = None
        self.version += 1

    def __len__(self) -> int:
        return len(self.obs)

    @property
    def completed(self) -> int:
        return sum(1 for o in self.obs if o[1])

    def fit(self):
        """``(V, mass, tokens matrix, vtimes matrix, residual mass, mean group size, residual profile)``"""
        if self.cache is None:
            n = len(self.obs)
            key = np.array([o[0] for o in self.obs])
            event = np.array([o[1] for o in self.obs])
            order = np.lexsort((~event, key))  # events before censorings at ties
            at_risk = n - np.arange(n)
            surv = np.cumprod(np.where(event[order], 1.0 - 1.0 / at_risk, 1.0))
            mass = np.concatenate([[1.0], surv[:-1]]) - surv
            ev_idx = order[event[order]]
            width = max(len(o[2]) for o in self.obs)
            tok = np.zeros((len(ev_idx), width))
            vt = np.ones((len(ev_idx), width))
            for row, i in enumerate(ev_idx):
                m = len(self.obs[i][2])
                tok[row, :m], vt[row, :m] = self.obs[i][2], self.obs[i][3]
            size = float(np.mean([len(o[2]) + o[4] for o in self.obs]))
            last = key[ev_idx].max() if len(ev_idx) else -math.inf
            tail = [o for o in self.obs if not o[1] and o[0] >= last] or [o for o in self.obs if not o[1]]
            profile = [(o[2], o[3], o[4]) for o in tail]
            self.cache = (key[ev_idx], mass[event[order]], tok, vt, float(surv[-1]) if n else 1.0, size, profile)
        return self.cache

    def terms(self, x: float, k_eff: float, staleness_scale: float | None):
        """``S(x)`` (P[group finishes by virtual time x]), ``A_w(x)`` (tokens of
        such groups, staleness-weighted) and ``C(x)`` (tokens spent per group
        when work past x is dropped)."""
        V, mass, tok, vt, resid, size, profile = self.fit()
        fresh = V <= x
        S = float(mass[fresh].sum())
        w = np.ones_like(V) if staleness_scale is None else np.exp(-k_eff * V / max(x, 1e-9) / staleness_scale)
        A = float((mass * tok.sum(axis=1) * w)[fresh].sum())
        spent = (tok * np.minimum(1.0, x / vt)).sum(axis=1)
        C = float((mass * spent).sum())
        if resid > 0:
            # Groups slower than any that completed never train; they cost what
            # the dropped ones did up to the cutoff
            if profile:
                cost = float(np.mean([(t * np.minimum(1.0, x / v)).sum() + u * x for t, v, u in profile]))
            else:
                cost = size * x
            C += resid * cost
        return S, A, C


class GoodputController:
    def __init__(
        self,
        config: ConcurrencyConfig,
        *,
        batch_size: int | None = None,
        max_off_policy_steps: int | None = None,
    ) -> None:
        self.config = config
        self.floor = config.min_inflight
        self.batch_size = batch_size
        self.k_eff = None if max_off_policy_steps is None else max_off_policy_steps + config.deadline_offset_steps
        self.staleness_scale = config.staleness_scale
        self.min_fresh = config.min_fresh_fraction
        self.law = GroupDurationLaw()
        self.groups: dict[uuid.UUID | str, _Group] = {}
        self.model_cache: dict[tuple[int, int], tuple[float, float, float]] = {}

        # Without a user-set start, the cap is sized per decode engine on the first poll
        self.bootstrapped = config.initial_inflight is not None
        start = config.initial_inflight or config.bootstrap_inflight_per_engine
        self.cur = bin_of(self.clamp(start))
        self.max_inflight = int(self.clamp(bin_size(self.cur)))
        self.bins: dict[int, _Bin] = {}
        self.penalized: dict[int, float] = {}
        self.prev: int | None = None  # bin we came from on the last move
        self.climb = 1
        self.lifetime: float | None = None
        self.vclock = 0.0  # tokens one average in-flight episode has generated
        self.last_poll: float | None = None
        self.last_rate = 0.0
        self.preempt_polls = 0
        self.abort_polls = 0
        self.signal = "probe"
        self.now = time.monotonic()
        self.start_phase(self.now)
        # Benchmarking: a pinned cap schedule [(seconds since start, cap), ...]
        self.schedule = sorted(config.schedule or []) if config.mode == "schedule" else []
        self.t0 = self.now
        if self.schedule:
            self.max_inflight = int(self.clamp(self.schedule[0][1]))

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
        self.set_limit = set_limit
        self.get_inflight = get_inflight
        self.on_overload = on_overload

    # ── inbound: dispatcher ──────────────────────────────────────────────────

    def record_episode(self, tokens: int) -> None:
        """Legacy hook; group events carry everything this controller needs."""

    def on_group_start(self, group_id, size: int) -> None:
        self.groups[group_id] = _Group(size=size, started_v=self.vclock)

    def on_episode_done(self, group_id, output_tokens: int, wall_s: float | None = None) -> None:
        if wall_s is not None and wall_s > 0:
            self.lifetime = wall_s if self.lifetime is None else 0.98 * self.lifetime + 0.02 * wall_s
        group = self.groups.get(group_id)
        if group is None:
            return
        group.lengths.append(float(max(output_tokens, 0)))
        group.finished_v.append(self.vclock - group.started_v)
        if len(group.lengths) >= group.size:
            del self.groups[group_id]
            self.law.add_complete(group.lengths, group.finished_v)

    def on_group_drop(self, group_id, reason: str) -> None:
        group = self.groups.pop(group_id, None)
        if group is None or reason not in ("stale", "overload"):
            return
        self.law.add_censored(self.vclock - group.started_v, group.size, group.lengths, group.finished_v)

    # ── inbound: engine metrics ──────────────────────────────────────────────

    def observe(self, samples) -> None:
        samples = [s for s in samples if s.role != "prefill"]
        if not samples:
            return
        now = time.monotonic()
        self.now = now
        if not self.bootstrapped and not self.schedule:
            self.bootstrapped = True
            self.move(bin_of(self.config.bootstrap_inflight_per_engine * len(samples)), now, reason="bootstrap")
        if self.last_poll is None:
            self.last_poll = now
            return
        dt = now - self.last_poll
        self.last_poll = now
        if dt <= 0:
            return
        self.poll_dt = dt
        tokens = sum(s.generation_tokens_delta for s in samples)
        preempted = any(s.preemptions_delta > 0 for s in samples)
        inflight = self.get_inflight() if self.get_inflight is not None else 0
        rate = tokens / dt
        self.last_rate = rate
        if inflight > 0:
            self.vclock += rate / inflight * dt

        if self.schedule:
            self.follow_schedule(now, rate, inflight)
            return

        self.preempt_polls = self.preempt_polls + 1 if preempted else 0
        if self.preempt_polls >= PREEMPTION_POLLS:
            self.preempt_polls = 0
            self.penalized[self.cur] = now
            back = self.prev if self.prev is not None and self.prev < self.cur else self.cur - 1
            self.move(back, now, reason="preemptions")
            return

        if now < self.settle_until:
            return
        self.acc_tokens += tokens
        self.acc_time += dt
        self.acc_pool += inflight * dt

        if self.prev is not None and self.prev in self.bins:
            incumbent = self.score(self.prev, optimistic=False)
            if self.goodput(bin_size(self.cur), rate) < ABORT_RATIO * incumbent:
                self.abort_polls += 1
            else:
                self.abort_polls = 0
            if self.abort_polls >= ABORT_POLLS:
                self.record(self.cur, self.acc_tokens / self.acc_time)
                self.penalized[self.cur] = now
                self.move(self.prev, now, reason="probe collapsed")
                return

        if now < self.measure_until:
            return
        T = self.acc_tokens / self.acc_time
        pool = self.acc_pool / self.acc_time
        binding = pool >= BINDING_FRACTION * self.max_inflight
        held = self.cur if binding else bin_of(pool)
        self.record(held, T)
        self.decide(now, binding)

    def follow_schedule(self, now: float, rate: float, inflight: int) -> None:
        """Pin the cap to the schedule; still measure T per level, so the log
        pairs each level's throughput with the predicted fresh fraction."""
        target = self.max_inflight
        for at, cap in self.schedule:
            if now - self.t0 >= at:
                target = int(self.clamp(cap))
        if target != self.max_inflight:
            get_logger().info(f"Scheduled concurrency {self.max_inflight} -> {target}")
            self.max_inflight = target
            self.start_phase(now)
            if self.set_limit is not None:
                self.set_limit(target)
            return
        if now >= self.settle_until:
            self.acc_tokens += rate * self.poll_dt
            self.acc_time += self.poll_dt
            self.acc_pool += inflight * self.poll_dt
        if now >= self.measure_until and self.acc_time > 0:
            self.record(bin_of(self.acc_pool / self.acc_time), self.acc_tokens / self.acc_time)
            self.start_phase(now)

    # ── model ────────────────────────────────────────────────────────────────

    def cutoff(self, P: float) -> float:
        """Slowest group (virtual duration) that still trains fresh at pool ``P``."""
        V, *_, size, _ = self.law.fit()
        groups_per_batch = self.batch_size / size
        lo, hi = 1.0, 1e3 * max(float(V.max()) if len(V) else 1.0, 1.0)
        S, _, C = self.law.terms(hi, self.k_eff, None)
        if hi * P * S <= self.k_eff * groups_per_batch * C:
            return hi
        for _ in range(48):
            x = math.sqrt(lo * hi)
            S, _, C = self.law.terms(x, self.k_eff, None)
            if x * P * S > self.k_eff * groups_per_batch * C:
                hi = x
            else:
                lo = x
        return math.sqrt(lo * hi)

    def model(self, P: float) -> tuple[float, float, float]:
        """``(cutoff, predicted group survival, eta)`` at pool ``P``, cached per law update."""
        key = (int(P), self.law.version)
        if key not in self.model_cache:
            if len(self.model_cache) > 256:
                self.model_cache.clear()
            x = self.cutoff(P)
            S, _, C = self.law.terms(x, self.k_eff, None)
            _, A, C = self.law.terms(x, self.k_eff, self.staleness_scale)
            self.model_cache[key] = (x, S, A / C if C > 0 else 1.0)
        return self.model_cache[key]

    def eta(self, P: float) -> float:
        return self.model(P)[2] if self.law_ready else 1.0

    @property
    def law_ready(self) -> bool:
        return self.k_eff is not None and bool(self.batch_size) and self.law.completed >= MIN_GROUPS

    def goodput(self, P: float, T: float) -> float:
        return T * self.eta(P)

    def score(self, i: int, *, optimistic: bool = True) -> float:
        P = bin_size(i)
        b = self.bins.get(i)
        if b is not None:
            T = b.mean + (UCB * math.sqrt(b.var / b.n) if optimistic else 0.0)
        elif self.bins:
            near = min(self.bins, key=lambda j: abs(j - i))
            # Optimistic: linear scaling above the data, flat below it
            T = self.bins[near].mean * (P / bin_size(near) if i > near else 1.0)
        else:
            return 0.0
        eta = self.eta(P)
        if eta < self.min_fresh:
            # Infeasible: closer to the bound ranks higher, below any feasible bin
            return eta - self.min_fresh
        penalty = PENALTY if self.now - self.penalized.get(i, -math.inf) < PENALTY_TTL_S else 1.0
        return T * eta * penalty

    # ── control loop ─────────────────────────────────────────────────────────

    def record(self, i: int, T: float) -> None:
        b = self.bins.get(i)
        if b is None:
            self.bins[i] = _Bin(mean=T, var=(0.05 * T) ** 2, n=1.0)
            return
        n = min(b.n + 1.0, BIN_MEMORY)
        d = T - b.mean
        b.mean += d / n
        b.var = max((1 - 1 / n) * (b.var + d * d / n), (0.03 * b.mean) ** 2)
        b.n = n

    def decide(self, now: float, binding: bool) -> None:
        lo, hi = bin_of(self.floor), bin_of(self.ceiling)
        cands = {self.cur, max(lo, self.cur - 1)}
        if binding:
            cands.add(min(hi, self.cur + self.climb))
        scores = {i: self.score(i) for i in cands}
        best = max(scores, key=scores.get)
        if best > self.cur and self.prev is not None and self.prev < self.cur and self.prev in self.bins:
            # Slow start: keep doubling the step while T scales near-linearly
            T0, T1 = self.bins[self.prev].mean, self.bins[self.cur].mean
            dlogp = math.log(bin_size(self.cur) / bin_size(self.prev))
            gain = math.log(max(T1, 1e-9) / max(T0, 1e-9)) / dlogp if dlogp > 0 else 0.0
            self.climb = min(MAX_CLIMB_BINS, self.climb * 2) if gain >= LINEAR_EFFICIENCY else 1
        elif best <= self.cur:
            self.climb = 1
        self.signal = "probe" if best != self.cur else "hold"
        self.move(best, now, reason=None, scores=scores)

    def move(self, i: int, now: float, *, reason: str | None, scores: dict | None = None) -> None:
        i = min(max(i, bin_of(self.floor)), bin_of(self.ceiling))
        self.prev = self.cur if i != self.cur else None
        self.cur = i
        self.abort_polls = 0
        self.start_phase(now)
        target = int(self.clamp(bin_size(i)))
        if target != self.max_inflight:
            detail = reason or ", ".join(f"{bin_size(j)}: {s:.0f}" for j, s in sorted((scores or {}).items()))
            get_logger().info(
                f"{'Increased' if target > self.max_inflight else 'Decreased'} concurrency "
                f"{self.max_inflight} -> {target} ({detail}) - eta={self.eta(target):.3f}"
            )
            self.max_inflight = target
            if self.set_limit is not None:
                self.set_limit(target)

    def start_phase(self, now: float) -> None:
        life = self.lifetime or 0.0
        self.settle_until = now + min(MAX_PHASE_S, max(MIN_SETTLE_S, SETTLE_LIFETIMES * life))
        self.measure_until = self.settle_until + min(MAX_PHASE_S, max(MIN_MEASURE_S, MEASURE_LIFETIMES * life))
        self.acc_tokens = self.acc_time = self.acc_pool = 0.0

    @property
    def ceiling(self) -> float:
        return float(self.config.max_inflight or 2**20)

    def clamp(self, n: float) -> float:
        return min(max(n, float(self.floor)), self.ceiling)

    # ── observability ────────────────────────────────────────────────────────

    def gauges(self) -> dict[str, float]:
        b = self.bins.get(self.cur)
        P = float(self.max_inflight)
        eta = self.eta(P)
        T = b.mean if b is not None else self.last_rate
        out = {
            "concurrency/max_inflight": P,
            "concurrency/throughput": self.last_rate,
            "concurrency/throughput_est": T,
            "concurrency/eta": eta,
            "concurrency/goodput_est": T * eta,
            "concurrency/probing": float(self.signal == "probe"),
            "concurrency/law_groups": float(len(self.law)),
            "concurrency/episode_lifetime_s": self.lifetime or 0.0,
        }
        if self.law_ready:
            x, S, _ = self.model(P)
            out["concurrency/cutoff_virtual_tokens"] = x
            # Validation pair: predicted vs observed share of train groups that finish fresh
            out["concurrency/pred_group_survival"] = S
            out["concurrency/obs_group_survival"] = self.law.completed / len(self.law)
        return out


def make_concurrency_controller(
    config: ConcurrencyConfig,
    *,
    fallback_cost: int,
    batch_size: int | None = None,
    max_off_policy_steps: int | None = None,
):
    """The controller selected by ``config.mode``."""
    if config.mode == "legacy":
        from prime_rl.orchestrator.concurrency import ConcurrencyController

        return ConcurrencyController(config, fallback_cost=fallback_cost)
    return GoodputController(config, batch_size=batch_size, max_off_policy_steps=max_off_policy_steps)
