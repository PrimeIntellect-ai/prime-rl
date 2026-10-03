"""Training source selection, sample mixing, and curriculum lifecycle."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import verifiers.v1 as vf

from prime_rl.orchestrator.curriculum import Curriculum
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.types import CancelReason, TaskRequest
from prime_rl.orchestrator.utils import episode_env_name

MIXER_ALPHA = 0.5
"""Blend of the target term (long-run dispatch ∝ demand) and the deficit term
(push envs behind on the current batch) in the env weights."""
MIN_ACCEPTANCE_RATE = 0.05
"""Below this acceptance rate an env's dispatch correction tapers back to 1×
(see :func:`acceptance_correction`)."""
PRIOR_ACCEPTANCE_RATE = 0.5
PRIOR_GROUPS = 4
"""Finalized groups an env needs before its measured acceptance rate replaces the prior."""
ESTIMATE_WINDOW = 20
"""Samples (groups or episodes) in the acceptance-rate and duration moving averages."""
CAP_HEADROOM = 1.25
"""In-flight cap of an env relative to its Little's-law share of the train slots."""


def trace_quotas(shares: dict[str, float], batch_size: int) -> dict[str, int]:
    """Per-env trace counts of one batch: ``share * batch_size`` rounded by
    largest remainder, so the quotas sum to ``batch_size``."""
    exact = {env: share * batch_size for env, share in shares.items()}
    quotas = {env: int(value) for env, value in exact.items()}
    by_remainder = sorted(exact, key=lambda env: exact[env] - quotas[env], reverse=True)
    for env in by_remainder[: batch_size - sum(quotas.values())]:
        quotas[env] += 1
    return quotas


def acceptance_correction(acceptance_rate: float) -> float:
    """Groups to generate per accepted group: ``1 / acceptance_rate``, peaking
    at ``1 / MIN_ACCEPTANCE_RATE``. Below that rate the env's share is not
    reachable at a sane cost, so the correction tapers back to 1× at zero
    acceptance: an env that yields nothing is dispatched no more than an env
    that accepts everything, instead of starving the other envs."""
    if acceptance_rate >= MIN_ACCEPTANCE_RATE:
        return 1 / acceptance_rate
    return max(acceptance_rate / MIN_ACCEPTANCE_RATE**2, 1.0)


def mixer_weight(target: float, pending: float, acceptance_rate: float) -> float:
    """Dispatch weight of one env (MiMo-V2.6 Eq. 7), in groups to generate:
    ``target`` accepted groups per batch, ``pending`` accepted groups not yet
    shipped."""
    deficit = max(target - pending, 0.0)
    return (MIXER_ALPHA * target + (1 - MIXER_ALPHA) * deficit) * acceptance_correction(acceptance_rate)


def inflight_caps(
    demand: dict[str, float], durations: dict[str, float], capacity: int, max_off_policy_steps: int | None
) -> dict[str, float]:
    """Per-env in-flight episode caps: ``CAP_HEADROOM`` times the env's
    Little's-law share of ``capacity`` (``demand`` episodes per step times
    mean episode duration), clipped at the ``1 + max_off_policy_steps`` steps
    of demand that can still train."""
    need = {env: durations[env] * demand[env] for env in demand}
    total = sum(need.values())
    caps = {env: CAP_HEADROOM * capacity * need[env] / total for env in demand}
    if max_off_policy_steps is not None:
        caps = {env: min(cap, (1 + max_off_policy_steps) * demand[env]) for env, cap in caps.items()}
    return caps


def smooth_round_robin(current: dict[str, float], weights: dict[str, float]) -> str:
    """Smooth weighted round-robin: deterministic picks in proportion to
    ``weights``, interleaved rather than bunched. Mutates ``current``."""
    for env, weight in weights.items():
        current[env] += weight
    picked = max(current, key=current.__getitem__)
    current[picked] -= sum(weights.values())
    return picked


class TrainSource:
    """Mix train envs and host one user-authored curriculum per env.

    ``ratio`` is each env's target share of the prompts (groups) in a shipped
    batch. Envs are picked by smooth weighted round-robin over
    :func:`mixer_weight`, which corrects for the share of each env's groups
    that end up in the batch (acceptance rate); ``TrainSink`` enforces the
    share per batch with :meth:`quotas`, in traces. Envs at their in-flight cap (:func:`inflight_caps`) are
    skipped unless every env is at its cap, so a slow or stalled env cannot
    take every train slot."""

    def __init__(
        self, train_envs: TrainEnvs, batch_size: int | None = None, max_off_policy_steps: int | None = None
    ) -> None:
        self.envs = list(train_envs)
        if not self.envs:
            raise ValueError("TrainSource needs at least one train env")

        self.curricula: dict[str, Curriculum] = {}
        for env in self.envs:
            if env.tasks is None:
                raise RuntimeError(f"env {env.name} not started")
            tasks = env.tasks if env.num_tasks is None else list(env.tasks)
            self.curricula[env.name] = Curriculum(env.config.curriculum, tasks)

        self.env_names = [env.name for env in self.envs]
        total_ratio = sum(env.config.ratio for env in self.envs)
        self.shares = {env.name: env.config.ratio / total_ratio for env in self.envs}
        self.group_sizes = {env.name: env.config.group_size for env in self.envs}
        # An env's share of the batch's traces is its prompt share weighted by group size.
        total_traces = sum(share * self.group_sizes[name] for name, share in self.shares.items())
        self.trace_shares = {name: share * self.group_sizes[name] / total_traces for name, share in self.shares.items()}
        self.batch_size = batch_size
        # The staleness clip on the caps needs absolute targets, i.e. a trace batch.
        self.max_off_policy_steps = max_off_policy_steps if batch_size is not None else None
        # Accepted groups per batch. Token batches have no trace count, so the
        # targets are only relative and the deficit term is left out.
        self.targets = {name: share * (batch_size or 1) / total_traces for name, share in self.shares.items()}
        self.current = {name: 0.0 for name in self.env_names}
        self.acceptance = {name: PRIOR_ACCEPTANCE_RATE for name in self.env_names}
        self.finalized: Counter[str] = Counter()
        self.pending: Counter[str] = Counter()
        """Accepted traces per env waiting in the sink for a batch."""
        self.durations = {name: 0.0 for name in self.env_names}
        self.completed: Counter[str] = Counter()
        self.inflight: dict[str, int] = {}
        self.caps: dict[str, float] | None = None
        self._admitted: dict[str, int] = defaultdict(int)
        self._rejected: dict[str, int] = defaultdict(int)

    def quotas(self, batch_size: int) -> dict[str, int]:
        return trace_quotas(self.trace_shares, batch_size)

    def acceptance_rate(self, env_name: str) -> float:
        if self.finalized[env_name] < PRIOR_GROUPS:
            return PRIOR_ACCEPTANCE_RATE
        return self.acceptance[env_name]

    def weights(self) -> dict[str, float]:
        weights = {}
        for name, target in self.targets.items():
            pending = self.pending[name] / self.group_sizes[name] if self.batch_size is not None else 0.0
            weights[name] = mixer_weight(target, pending, self.acceptance_rate(name))
        return weights

    def next_task(self, *, step: int, capacity: int, inflight: dict[str, int]) -> TaskRequest:
        """``capacity`` is the train share of the dispatcher's in-flight cap,
        ``inflight`` the train episodes in flight per env."""
        weights = self.weights()
        self.inflight = inflight
        # Caps need a duration estimate for every env.
        self.caps = None
        if all(self.completed[name] for name in self.env_names):
            demand = {
                name: target * acceptance_correction(self.acceptance_rate(name)) * self.group_sizes[name]
                for name, target in self.targets.items()
            }
            self.caps = inflight_caps(demand, self.durations, capacity, self.max_off_policy_steps)
            uncapped = {
                name: weight if inflight.get(name, 0) < self.caps[name] else 0.0 for name, weight in weights.items()
            }
            if any(uncapped.values()):
                weights = uncapped
        env_name = smooth_round_robin(self.current, weights)
        return TaskRequest(env_name=env_name, task=next(self.curricula[env_name].sampler), step=step)

    def on_result(self, group: list[vf.Episode]) -> bool:
        """Report a finalized group and return whether it should train."""
        if not group:
            raise ValueError("Cannot report an empty rollout group")
        env_name = episode_env_name(group[0])
        admitted = self.curricula[env_name].on_result(group)
        if not isinstance(admitted, bool):
            raise TypeError(f"Curriculum.on_result() must return bool, got {type(admitted).__name__}")
        if admitted:
            self._admitted[env_name] += 1
        else:
            self._rejected[env_name] += 1
        return admitted

    def on_group_finalized(self, env_name: str, *, accepted: bool, cancel_reason: CancelReason | None) -> None:
        """Update the acceptance rate: ``accepted`` means the group queued
        traces for a batch. Groups cut by an overload or superseded
        cancellation are a pipeline decision, not env yield, and are skipped;
        stale groups count as not accepted, so a slow env that loses groups to
        the staleness bound is dispatched more."""
        if cancel_reason in ("overload", "superseded"):
            return
        self.finalized[env_name] += 1
        window = min(self.finalized[env_name], ESTIMATE_WINDOW)
        self.acceptance[env_name] += (float(accepted) - self.acceptance[env_name]) / window

    def on_episode_complete(self, env_name: str, duration: float) -> None:
        """Update the env's mean episode duration (seconds an episode holds a slot)."""
        self.completed[env_name] += 1
        window = min(self.completed[env_name], ESTIMATE_WINDOW)
        self.durations[env_name] += (duration - self.durations[env_name]) / window

    def metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        weights = self.weights()
        total_weight = sum(weights.values())
        for env_name, curriculum in self.curricula.items():
            admitted = self._admitted.pop(env_name, 0)
            rejected = self._rejected.pop(env_name, 0)
            total = admitted + rejected
            if total:
                metrics[f"curriculum/{env_name}/admission_rate"] = admitted / total
            metrics |= {f"curriculum/{env_name}/{name}": float(value) for name, value in curriculum.metrics().items()}
            metrics |= {
                f"mixer/{env_name}/target_prompt_share": self.shares[env_name],
                f"mixer/{env_name}/acceptance_rate": self.acceptance_rate(env_name),
                f"mixer/{env_name}/weight": weights[env_name] / total_weight,
                f"mixer/{env_name}/surplus_groups": self.pending[env_name] / self.group_sizes[env_name],
                f"mixer/{env_name}/inflight": float(self.inflight.get(env_name, 0)),
            }
            if self.caps is not None:
                metrics[f"mixer/{env_name}/cap"] = self.caps[env_name]
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "envs": {name: curriculum.state_dict() for name, curriculum in self.curricula.items()},
            "mixer": {
                "current": self.current,
                "acceptance": self.acceptance,
                "finalized": dict(self.finalized),
                "durations": self.durations,
                "completed": dict(self.completed),
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        env_states = state_dict["envs"]
        if set(env_states) != set(self.curricula):
            raise ValueError(
                f"Train-source checkpoint envs {sorted(env_states)} do not match configured envs "
                f"{sorted(self.curricula)}"
            )
        for name, curriculum in self.curricula.items():
            curriculum.load_state_dict(env_states[name])
        # Checkpoints without mixer state start the estimates from the prior.
        mixer = state_dict.get("mixer")
        if mixer is not None:
            self.current |= mixer["current"]
            self.acceptance |= mixer["acceptance"]
            self.finalized.update(mixer["finalized"])
            self.durations |= mixer["durations"]
            self.completed.update(mixer["completed"])
