"""Training source selection, sample mixing, and curriculum lifecycle."""

from __future__ import annotations

import random
import uuid
from collections import Counter, defaultdict
from typing import Any

import verifiers.v1 as vf

from prime_rl.configs.orchestrator import PrefixSourceConfig
from prime_rl.orchestrator.curriculum import Curriculum
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.prefix import PrefixBuffer
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
"""Samples (groups or episodes) in the acceptance-rate and duration moving averages."""
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
    durations: dict[str, float],
    capacity: int,
    max_off_policy_steps: dict[str, int] | None,
) -> dict[str, float]:
    """Per-env in-flight episode caps: ``CAP_HEADROOM`` times the env's
    Little's-law share of ``capacity`` (``demand`` episodes per step times
    mean episode duration), clipped at the ``1 + max_off_policy_steps`` steps
    of demand that can still train (the env's own bound)."""
    need = {env: durations[env] * demand[env] for env in demand}
    total = sum(need.values())
    caps = {env: CAP_HEADROOM * capacity * need[env] / total for env in demand}
    if max_off_policy_steps is not None:
        caps = {env: min(cap, (1 + max_off_policy_steps[env]) * demand[env]) for env, cap in caps.items()}
    return caps


class TrainSource:
    """Mix train envs and host one user-authored curriculum per env.

    ``ratio`` is each env's target share of the prompts (groups) in a shipped
    batch. Envs are picked at random in proportion to
    :func:`mixer_weight`, which corrects for the share of each env's groups
    that end up in the batch (acceptance rate); the share holds on average,
    batches ship whatever is queued (``TrainSink``). Envs at their
    in-flight cap (:func:`inflight_caps`) are skipped unless every env is at
    its cap, so a slow or stalled env cannot take every train slot.

    Prefix sources (``PrefixSourceConfig``) are mixer sources next to the envs:
    their groups run on their env, continue episodes from the env's
    ``PrefixBuffer``, and skip its curriculum. A prefix source with no eligible
    episode gets weight 0."""

    def __init__(
        self, train_envs: TrainEnvs, batch_size: int | None = None, prefixes: list[PrefixSourceConfig] | None = None
    ) -> None:
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

        prefixes = prefixes or []
        self.buffers = {prefix.name: PrefixBuffer(prefix) for prefix in prefixes}
        self.prefix_envs = {prefix.name: prefix.env for prefix in prefixes}
        self.fresh_tasks: dict[str, vf.Task] = {}
        """Task of each in-flight fresh group of an env with a prefix source, by group id."""
        self.prefix_stats: dict[str, list[dict[str, float]]] = defaultdict(list)
        self.step = 0
        ratios = {env.name: env.config.ratio for env in self.envs} | {prefix.name: prefix.ratio for prefix in prefixes}
        self.env_names = list(ratios)
        """Mixer source names: the train envs, then the prefix sources."""
        total_ratio = sum(ratios.values())
        self.shares = {name: ratio / total_ratio for name, ratio in ratios.items()}
        self.group_sizes = {env.name: env.config.group_size for env in self.envs}
        self.group_sizes |= {name: self.group_sizes[env] for name, env in self.prefix_envs.items()}
        # A prompt contributes ``group_size`` traces to the batch.
        total_traces = sum(share * self.group_sizes[name] for name, share in self.shares.items())
        self.batch_size = batch_size
        # The staleness clip on the caps needs absolute targets, i.e. a trace batch.
        max_off_policy_steps = {env.name: env.config.max_off_policy_steps for env in self.envs}
        max_off_policy_steps |= {name: max_off_policy_steps[env] for name, env in self.prefix_envs.items()}
        self.max_off_policy_steps = max_off_policy_steps if batch_size is not None else None
        # Accepted groups per batch. Token batches have no trace count, so the
        # targets are only relative and the deficit term is left out.
        self.targets = {name: share * (batch_size or 1) / total_traces for name, share in self.shares.items()}
        self.acceptance = {name: PRIOR_ACCEPTANCE_RATE for name in self.env_names}
        self.pending: Counter[str] = Counter()
        """Accepted traces per env waiting in the sink for a batch."""
        self.pending_groups: Counter[str] = Counter()
        """Accepted groups per env with traces waiting in the sink."""
        self.durations = {name: 0.0 for name in self.env_names}
        self.completed: Counter[str] = Counter()
        self.inflight: dict[str, int] = {}
        self.caps: dict[str, float] | None = None
        self.dispatch_weights: dict[str, float] = {}
        self._admitted: dict[str, int] = defaultdict(int)
        self._rejected: dict[str, int] = defaultdict(int)

    def weights(self, step: int = 0) -> dict[str, float]:
        """A prefix source with nothing to continue gets weight 0."""
        self.step = step
        weights = {}
        for name, target in self.targets.items():
            if name in self.buffers and not self.buffers[name].available(step):
                weights[name] = 0.0
                continue
            pending = self.pending_groups[name] if self.batch_size is not None else 0.0
            weights[name] = mixer_weight(target, pending, self.acceptance[name])
        return weights

    def next_task(self, *, step: int, capacity: int, inflight: dict[str, int]) -> TaskRequest:
        """``capacity`` is the train share of the dispatcher's in-flight cap,
        ``inflight`` the train episodes in flight per env."""
        weights = self.weights(step)
        self.inflight = inflight
        # Caps need a duration estimate for every env; a prefix source without one has its env's.
        self.caps = None
        if all(self.completed[env.name] for env in self.envs):
            demand = {
                name: target * acceptance_correction(self.acceptance[name]) * self.group_sizes[name]
                for name, target in self.targets.items()
            }
            durations = {
                name: self.durations[name if self.completed[name] else self.prefix_envs.get(name, name)]
                for name in self.env_names
            }
            self.caps = inflight_caps(demand, durations, capacity, self.max_off_policy_steps)
            uncapped = {
                name: weight if inflight.get(name, 0) < self.caps[name] else 0.0 for name, weight in weights.items()
            }
            if any(uncapped.values()):
                weights = uncapped
        self.dispatch_weights = weights
        env_name = self.rng.choices(list(weights), weights=list(weights.values()), k=1)[0]
        if env_name in self.buffers:
            task, prefix = self.buffers[env_name].sample(env_name)
            return TaskRequest(env_name=env_name, task=task, step=step, prefix=prefix)
        task = next(self.curricula[env_name].sampler)
        if env_name not in self.prefix_envs.values():
            return TaskRequest(env_name=env_name, task=task, step=step)
        group_id = str(uuid.uuid4())
        self.fresh_tasks[group_id] = task
        return TaskRequest(env_name=env_name, task=task, step=step, group_id=group_id)

    def on_group(self, group_id: str, env_name: str, group: list[vf.Episode]) -> None:
        """See a finalized group, trained or not. A fresh group of an env with
        prefix sources feeds their buffers; a prefix group records its stats."""
        task = self.fresh_tasks.pop(group_id, None)
        if task is not None:
            for name, env in self.prefix_envs.items():
                if env == env_name:
                    self.buffers[name].admit(task, group)
        if env_name not in self.buffers:
            return
        for episode in group:
            for trace in episode.traces:
                if trace.ok and "prefix" in trace.info:
                    info = trace.info["prefix"]
                    calls = info["source"]["calls"]
                    self.prefix_stats[env_name].append(
                        {
                            "continuation_reward": trace.reward,
                            "source_reward": info["source"]["reward"],
                            "cut_frac": info["cut"] / calls,
                            "realized_cut_frac": info["realized_cut"] / calls,
                            "obs_changed_frac": trace.metrics["prefix/obs_changed_frac"],
                        }
                    )

    def on_result(self, group: list[vf.Episode]) -> bool:
        """Report a finalized group and return whether it should train. Prefix
        groups bypass the curriculum: their pass rates are not the task's."""
        if not group:
            raise ValueError("Cannot report an empty rollout group")
        env_name = episode_env_name(group[0])
        if env_name in self.buffers:
            return True
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
        self.acceptance[env_name] += (float(accepted) - self.acceptance[env_name]) / ESTIMATE_WINDOW

    def on_episode_complete(self, env_name: str, duration: float) -> None:
        """Update the env's mean episode duration (seconds an episode holds a slot)."""
        self.completed[env_name] += 1
        window = min(self.completed[env_name], ESTIMATE_WINDOW)
        self.durations[env_name] += (duration - self.durations[env_name]) / window

    def metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name, buffer in self.buffers.items():
            metrics |= {f"prefix/{name}/{key}": value for key, value in buffer.metrics(self.step).items()}
            stats = self.prefix_stats.pop(name, [])
            for key in stats[0] if stats else ():
                metrics[f"prefix/{name}/{key}"] = sum(stat[key] for stat in stats) / len(stats)
        total_weight = sum(self.dispatch_weights.values())
        for env_name, curriculum in self.curricula.items():
            admitted = self._admitted.pop(env_name, 0)
            rejected = self._rejected.pop(env_name, 0)
            total = admitted + rejected
            if total:
                metrics[f"curriculum/{env_name}/admission_rate"] = admitted / total
            metrics |= {f"curriculum/{env_name}/{name}": float(value) for name, value in curriculum.metrics().items()}
        for env_name in self.env_names:
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
            if self.completed[env_name]:
                # Mean seconds an episode holds a slot; what the in-flight caps are sized from
                metrics[f"mixer/{env_name}/episode_duration"] = self.durations[env_name]
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng": self.rng.getstate(),
            "envs": {name: curriculum.state_dict() for name, curriculum in self.curricula.items()},
            "mixer": {
                "acceptance": self.acceptance,
                "durations": self.durations,
                "completed": dict(self.completed),
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
            # Sources added since the checkpoint keep their fresh state.
            self.acceptance |= {name: value for name, value in mixer["acceptance"].items() if name in self.acceptance}
            self.durations |= {name: value for name, value in mixer["durations"].items() if name in self.durations}
            self.completed = Counter({name: n for name, n in mixer["completed"].items() if name in self.durations})
