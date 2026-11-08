"""Training source selection and curriculum lifecycle."""

from __future__ import annotations

import random
import uuid
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from typing import Any

import verifiers.v1 as vf

from prime_rl.control.types import ConfigUpdate, Reconfigurable
from prime_rl.orchestrator.curriculum import Curriculum
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.types import TaskRequest
from prime_rl.orchestrator.utils import episode_env_name
from prime_rl.utils.logger import get_logger


class TrainSource(Reconfigurable):
    """Mix train envs and host one user-authored curriculum per env.

    Each draw opens a train group, which stays open until its samples first ship or
    it finalizes without training. The checkpoint saves the open groups, and a
    resume re-runs them before it draws new tasks, so a task's sampling count is
    the number of its groups that finished plus the open groups the resume re-runs.
    """

    def __init__(self, train_envs: TrainEnvs) -> None:
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

        self.streaming = {env.name for env in self.envs if env.num_tasks is None}
        self.env_names = [env.name for env in self.envs]
        self.weights = [float(env.config.ratio) for env in self.envs]
        self._admitted: dict[str, int] = defaultdict(int)
        self._rejected: dict[str, int] = defaultdict(int)
        # Group id -> (env name, task key), in draw order.
        self.open_groups: dict[str, tuple[str, str]] = {}
        self.reopened: deque[tuple[str, vf.Task]] = deque()

    def reconfigure(self, update: ConfigUpdate) -> None:
        ratios = {source.resolved_name: source.ratio for source in update.config.train.source}
        self.weights = [float(ratios[name]) for name in self.env_names]

    def next_task(self, *, step: int) -> TaskRequest:
        # A reopened group was already counted by its sampler, and its env is fixed.
        if self.reopened:
            env_name, task = self.reopened.popleft()
        else:
            env_name = self.rng.choices(self.env_names, weights=self.weights, k=1)[0]
            task = next(self.curricula[env_name].sampler)
        group_id = str(uuid.uuid4())
        self.open_groups[group_id] = (env_name, task.key)
        return TaskRequest(env_name=env_name, task=task, step=step, group_id=group_id)

    def close_groups(self, group_ids: Iterable[str]) -> None:
        """Close groups that shipped samples or finalized without training."""
        for group_id in group_ids:
            self.open_groups.pop(group_id, None)

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

    def metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for env_name, curriculum in self.curricula.items():
            admitted = self._admitted.pop(env_name, 0)
            rejected = self._rejected.pop(env_name, 0)
            total = admitted + rejected
            if total:
                metrics[f"curriculum/{env_name}/admission_rate"] = admitted / total
            metrics |= {f"curriculum/{env_name}/{name}": float(value) for name, value in curriculum.metrics().items()}
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng": self.rng.getstate(),
            "envs": {name: curriculum.state_dict() for name, curriculum in self.curricula.items()},
            # Groups still waiting to re-run come last: no new task is drawn before they all re-run.
            "open_groups": [*self.open_groups.values(), *((env_name, task.key) for env_name, task in self.reopened)],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore each configured env's curriculum by env name. An env missing from the
        checkpoint starts fresh; state of an env that is no longer configured is dropped.
        Open groups re-run, except those of a removed env or task or a streaming taskset."""
        required_fields = {"rng", "envs"}
        if not required_fields <= set(state_dict) <= required_fields | {"open_groups"}:
            raise ValueError(f"Unexpected train-source checkpoint fields: {sorted(state_dict)}")
        env_states = state_dict["envs"]
        self.rng.setstate(state_dict["rng"])
        for name, curriculum in self.curricula.items():
            if name in env_states:
                get_logger().info(f"Resuming curriculum state for env {name}")
                curriculum.load_state_dict(env_states[name])
            else:
                get_logger().info(f"Env {name} is not in the checkpoint, so its curriculum starts fresh")
        for name in sorted(env_states.keys() - self.curricula.keys()):
            get_logger().info(f"Dropped the checkpointed curriculum state of env {name}, which is no longer configured")
        self._reopen(state_dict.get("open_groups", []))

    def _reopen(self, open_groups: list[tuple[str, str]]) -> None:
        removed_env: Counter[str] = Counter()
        removed_task: Counter[str] = Counter()
        for env_name, task_key in open_groups:
            if env_name not in self.curricula:
                removed_env[env_name] += 1
            elif (task := self.curricula[env_name].sampler.find_task(task_key)) is None:
                removed_task[env_name] += 1
            else:
                self.reopened.append((env_name, task))
        for name, count in sorted(removed_env.items()):
            get_logger().info(f"Dropped {count} open group(s) of env {name}, which is no longer configured")
        for name, count in sorted(removed_task.items()):
            if name in self.streaming:
                get_logger().info(f"Dropped {count} open group(s) of env {name}: a streaming taskset cannot re-run")
            else:
                get_logger().info(f"Dropped {count} open group(s) of env {name} whose task is no longer in the taskset")
        if self.reopened:
            get_logger().info(f"Re-running {len(self.reopened)} open group(s) from the checkpoint")
