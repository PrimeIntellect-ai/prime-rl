"""Training source selection, sample mixing, and curriculum lifecycle."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict, deque
from statistics import fmean
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
ESTIMATE_WINDOW = 20
"""Samples (groups, episodes or steps) in the acceptance-rate, duration and step-time estimates."""
CAP_HEADROOM = 1.25
"""In-flight cap of an env relative to its Little's-law share of the train slots."""


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
    demand: dict[str, float],
    durations: dict[str, deque[float]],
    capacity: int,
    step_time: float | None,
    max_off_policy_steps: dict[str, int],
) -> dict[str, float]:
    """Per-env in-flight episode caps: ``CAP_HEADROOM`` times the env's
    Little's-law share of ``capacity`` (``demand`` episodes per step times
    mean episode duration). An env whose p90 episode outlasts the
    ``1 + max_off_policy_steps`` steps of ``step_time`` it may run and still
    train has its cap scaled down in proportion, so it cannot hold more work
    than can finish in time."""
    need = {env: fmean(durations[env]) * demand[env] for env in demand}
    total = sum(need.values())
    caps = {env: CAP_HEADROOM * capacity * need[env] / total for env in demand}
    if step_time:
        for env in caps:
            p90_steps = sorted(durations[env])[math.ceil(0.9 * len(durations[env])) - 1] / step_time
            caps[env] *= min(1.0, (1 + max_off_policy_steps[env]) / p90_steps)
    return caps


class TrainSource:
    """Mix train envs and host one user-authored curriculum per env.

    ``ratio`` is each env's target share of the prompts (groups) in a shipped
    batch. Envs are picked at random in proportion to
    :func:`mixer_weight`, which corrects for the share of each env's groups
    that end up in the batch (acceptance rate); the share holds on average,
    batches ship whatever is queued (``TrainSink``). Envs at their
    in-flight cap (:func:`inflight_caps`) are skipped unless every env is at
    its cap, so a slow or stalled env cannot take every train slot."""

    def __init__(self, train_envs: TrainEnvs, batch_size: int | None = None) -> None:
        self.rng = random.Random(42)
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
        self.max_off_policy_steps = {env.name: env.config.max_off_policy_steps for env in self.envs}
        # A prompt contributes ``group_size`` traces to the batch.
        total_traces = sum(share * self.group_sizes[name] for name, share in self.shares.items())
        self.batch_size = batch_size
        # Accepted groups per batch. Token batches have no trace count, so the
        # targets are only relative and the deficit term is left out.
        self.targets = {name: share * (batch_size or 1) / total_traces for name, share in self.shares.items()}
        self.acceptance = {name: PRIOR_ACCEPTANCE_RATE for name in self.env_names}
        self.pending: Counter[str] = Counter()
        """Accepted traces per env waiting in the sink for a batch."""
        self.pending_groups: Counter[str] = Counter()
        """Accepted groups per env with traces waiting in the sink."""
        self.durations = {name: deque(maxlen=ESTIMATE_WINDOW) for name in self.env_names}
        """Recent episode durations per env, in seconds an episode holds a slot."""
        self.step_times: deque[float] = deque(maxlen=ESTIMATE_WINDOW)
        self.inflight: dict[str, int] = {}
        self.caps: dict[str, float] | None = None
        self.dispatch_weights: dict[str, float] = {}
        self._admitted: dict[str, int] = defaultdict(int)
        self._rejected: dict[str, int] = defaultdict(int)

    def weights(self) -> dict[str, float]:
        weights = {}
        for name, target in self.targets.items():
            pending = self.pending_groups[name] if self.batch_size is not None else 0.0
            weights[name] = mixer_weight(target, pending, self.acceptance[name])
        return weights

    def next_task(self, *, step: int, capacity: int, inflight: dict[str, int]) -> TaskRequest:
        """``capacity`` is the train share of the dispatcher's in-flight cap,
        ``inflight`` the train episodes in flight per env."""
        weights = self.weights()
        self.inflight = inflight
        # Caps need a duration estimate for every env.
        self.caps = None
        if all(self.durations.values()):
            demand = {
                name: target * acceptance_correction(self.acceptance[name]) * self.group_sizes[name]
                for name, target in self.targets.items()
            }
            step_time = fmean(self.step_times) if self.step_times else None
            self.caps = inflight_caps(demand, self.durations, capacity, step_time, self.max_off_policy_steps)
            uncapped = {
                name: weight if inflight.get(name, 0) < self.caps[name] else 0.0 for name, weight in weights.items()
            }
            if any(uncapped.values()):
                weights = uncapped
        self.dispatch_weights = weights
        env_name = self.rng.choices(list(weights), weights=list(weights.values()), k=1)[0]
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
        traces for a batch. Cancelled groups (stale, overload, superseded)
        say nothing about the env's yield and are skipped: counting stale
        groups as rejected would dispatch more of an env the more of its work
        goes stale."""
        if cancel_reason is not None:
            return
        self.acceptance[env_name] += (float(accepted) - self.acceptance[env_name]) / ESTIMATE_WINDOW

    def on_episode_complete(self, env_name: str, duration: float) -> None:
        self.durations[env_name].append(duration)

    def on_step(self, step_time: float) -> None:
        """Record the seconds between two shipped batches."""
        self.step_times.append(step_time)

    def metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        total_weight = sum(self.dispatch_weights.values())
        for env_name, curriculum in self.curricula.items():
            admitted = self._admitted.pop(env_name, 0)
            rejected = self._rejected.pop(env_name, 0)
            total = admitted + rejected
            if total:
                metrics[f"curriculum/{env_name}/admission_rate"] = admitted / total
            metrics |= {f"curriculum/{env_name}/{name}": float(value) for name, value in curriculum.metrics().items()}
            metrics |= {
                f"mixer/{env_name}/target_prompt_share": self.shares[env_name],
                f"mixer/{env_name}/acceptance_rate": self.acceptance[env_name],
                f"mixer/{env_name}/surplus_groups": float(self.pending_groups[env_name]),
                f"mixer/{env_name}/inflight": float(self.inflight.get(env_name, 0)),
            }
            if total_weight:
                # Share of the last dispatch decision, after the in-flight caps
                metrics[f"mixer/{env_name}/weight"] = self.dispatch_weights[env_name] / total_weight
            if self.caps is not None:
                metrics[f"mixer/{env_name}/cap"] = self.caps[env_name]
            if self.durations[env_name]:
                # Mean seconds an episode holds a slot; what the in-flight caps are sized from
                metrics[f"mixer/{env_name}/episode_duration"] = fmean(self.durations[env_name])
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng": self.rng.getstate(),
            "envs": {name: curriculum.state_dict() for name, curriculum in self.curricula.items()},
            "mixer": {
                "acceptance": self.acceptance,
                "durations": {name: list(durations) for name, durations in self.durations.items()},
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not {"rng", "envs"} <= set(state_dict) <= {"rng", "envs", "mixer"}:
            raise ValueError("Train-source checkpoint fields must be rng, envs and optionally mixer")
        env_states = state_dict["envs"]
        if set(env_states) != set(self.curricula):
            raise ValueError(
                f"Train-source checkpoint envs {sorted(env_states)} do not match configured envs "
                f"{sorted(self.curricula)}"
            )
        self.rng.setstate(state_dict["rng"])
        for name, curriculum in self.curricula.items():
            curriculum.load_state_dict(env_states[name])
        # Checkpoints without mixer state start the estimates from the prior.
        mixer = state_dict.get("mixer")
        if mixer is not None:
            self.acceptance = dict(mixer["acceptance"])
            self.durations = {
                name: deque(durations, maxlen=ESTIMATE_WINDOW) for name, durations in mixer["durations"].items()
            }
