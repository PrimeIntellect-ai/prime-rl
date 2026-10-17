"""Training source selection and curriculum lifecycle."""

from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import Any

import verifiers.v1 as vf

from prime_rl.orchestrator.curriculum import Curriculum
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.types import TaskRequest
from prime_rl.orchestrator.utils import episode_env_name
from prime_rl.utils.logger import get_logger

TASK_ERRORS = {"TaskError", "ToolsetError"}
"""Error types raised by a task's own code (setup, finalize, rewards) or its toolset."""
CONFIRM_GROUPS = 4
"""Groups of the env before and after a failing task's group that decide whether it is
quarantined (most on each side must succeed) or the env was failing as a whole."""


class TrainSource:
    """Mix train envs and host one user-authored curriculum per env."""

    def __init__(self, train_envs: TrainEnvs, quarantine_after: int | None = None) -> None:
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
        self.weights = [float(env.config.ratio) for env in self.envs]
        self._admitted: dict[str, int] = defaultdict(int)
        self._rejected: dict[str, int] = defaultdict(int)

        self.quarantine_after = quarantine_after
        self.quarantined: dict[str, dict[str, str]] = {name: {} for name in self.env_names}
        """Per env, the keys of tasks no longer sampled, with the error that put them there."""
        self._failed_groups: dict[tuple[str, str], int] = defaultdict(int)
        self._candidates: dict[str, dict[str, tuple[str, list[bool]]]] = {name: {} for name in self.env_names}
        """Per env, tasks waiting for confirmation: their error and whether each later group of the env succeeded."""
        self._recent_ok: dict[str, deque[bool]] = {name: deque(maxlen=CONFIRM_GROUPS) for name in self.env_names}
        self.num_tasks = {env.name: env.num_tasks for env in self.envs}

    def next_task(self, *, step: int) -> TaskRequest:
        env_name = self.rng.choices(self.env_names, weights=self.weights, k=1)[0]
        quarantined = self.quarantined[env_name]
        if quarantined and len(quarantined) == self.num_tasks[env_name]:
            raise RuntimeError(f"Every task of env {env_name} is quarantined: {quarantined}")
        task = next(self.curricula[env_name].sampler)
        while task.key in quarantined:
            task = next(self.curricula[env_name].sampler)
        return TaskRequest(env_name=env_name, task=task, step=step)

    def _track_failures(self, env_name: str, group: list[vf.Episode]) -> None:
        """Quarantine a task after ``quarantine_after`` consecutive groups in which every
        episode failed in the task's own code or toolset. Infra errors (sandbox, provider,
        env server) say nothing about the task; timeouts and zero rewards are task results.
        A task that qualifies is quarantined only if most of the env's ``CONFIRM_GROUPS``
        groups before it and most of the ``CONFIRM_GROUPS`` groups of other tasks after it
        did not fully error, so neither the start nor the end of an outage (e.g. a judge that
        every reward calls) quarantines anything."""
        errors = [
            [*episode.errors, *(error for trace in episode.traces for error in trace.errors)] for episode in group
        ]
        failed = all(errors)
        key = group[0].task.key
        candidates = self._candidates[env_name]
        for other in [other for other in candidates if other != key]:
            reason, outcomes = candidates[other]
            outcomes.append(not failed)
            if len(outcomes) == CONFIRM_GROUPS:
                del candidates[other]
                if 2 * sum(outcomes) > CONFIRM_GROUPS:
                    self._quarantine(env_name, other, reason)
        recent = self._recent_ok[env_name]
        env_healthy = len(recent) == CONFIRM_GROUPS and 2 * sum(recent) > CONFIRM_GROUPS
        recent.append(not failed)
        if not failed:
            self._failed_groups.pop((env_name, key), None)
            return
        if not all(episode_errors[-1].type in TASK_ERRORS for episode_errors in errors):
            return
        self._failed_groups[(env_name, key)] += 1
        if self._failed_groups[(env_name, key)] >= self.quarantine_after and env_healthy and key not in candidates:
            del self._failed_groups[(env_name, key)]
            last = errors[-1][-1]
            candidates[key] = (f"{last.type}: {last.message}"[:500], [])

    def _quarantine(self, env_name: str, key: str, reason: str) -> None:
        self.quarantined[env_name][key] = reason
        get_logger().warning(
            f"Quarantined task {key} of env {env_name}: every episode of its last {self.quarantine_after} "
            f"group(s) failed in task code (last: {reason}). It is no longer sampled "
            f"({len(self.quarantined[env_name])} quarantined in {env_name})."
        )

    def on_result(self, group: list[vf.Episode]) -> bool:
        """Report a finalized group and return whether it should train."""
        if not group:
            raise ValueError("Cannot report an empty rollout group")
        env_name = episode_env_name(group[0])
        if self.quarantine_after is not None:
            self._track_failures(env_name, group)
        admitted = self.curricula[env_name].on_result(group)
        if not isinstance(admitted, bool):
            raise TypeError(f"Curriculum.on_result() must return bool, got {type(admitted).__name__}")
        if admitted:
            self._admitted[env_name] += 1
        else:
            self._rejected[env_name] += 1
        return admitted

    def metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for env_name, curriculum in self.curricula.items():
            admitted = self._admitted.pop(env_name, 0)
            rejected = self._rejected.pop(env_name, 0)
            total = admitted + rejected
            if total:
                metrics[f"curriculum/{env_name}/admission_rate"] = admitted / total
            metrics |= {f"curriculum/{env_name}/{name}": float(value) for name, value in curriculum.metrics().items()}
            if self.quarantine_after is not None:
                metrics[f"curriculum/{env_name}/quarantined"] = float(len(self.quarantined[env_name]))
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng": self.rng.getstate(),
            "envs": {name: curriculum.state_dict() for name, curriculum in self.curricula.items()},
            "quarantined": self.quarantined,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Checkpoints written before quarantine have no "quarantined".
        expected_fields = {"rng", "envs", "quarantined"}
        if not {"rng", "envs"} <= set(state_dict) <= expected_fields:
            raise ValueError(f"Train-source checkpoint fields must be {sorted(expected_fields)}")
        env_states = state_dict["envs"]
        if set(env_states) != set(self.curricula):
            raise ValueError(
                f"Train-source checkpoint envs {sorted(env_states)} do not match configured envs "
                f"{sorted(self.curricula)}"
            )
        self.rng.setstate(state_dict["rng"])
        for name, curriculum in self.curricula.items():
            curriculum.load_state_dict(env_states[name])
        self.quarantined.update(state_dict.get("quarantined", {}))
        for name, quarantined in self.quarantined.items():
            if quarantined:
                get_logger().info(f"Resumed {len(quarantined)} quarantined tasks of env {name}")
