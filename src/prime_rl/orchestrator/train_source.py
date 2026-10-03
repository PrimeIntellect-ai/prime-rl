"""Training source selection and per-env task samplers."""

from __future__ import annotations

import random
from typing import Any

import verifiers.v1 as vf

from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.samplers import TaskSampler, setup_sampler
from prime_rl.orchestrator.types import TaskRequest
from prime_rl.orchestrator.utils import episode_env_name


class TrainSource:
    """Mix train envs and host one task sampler per env."""

    def __init__(self, train_envs: TrainEnvs) -> None:
        self.rng = random.Random(42)
        self.envs = list(train_envs)
        if not self.envs:
            raise ValueError("TrainSource needs at least one train env")

        self.samplers: dict[str, TaskSampler] = {}
        for env in self.envs:
            if env.tasks is None:
                raise RuntimeError(f"env {env.name} not started")
            tasks = env.tasks if env.num_tasks is None else list(env.tasks)
            self.samplers[env.name] = setup_sampler(env.config.sampler, tasks)

        self.env_names = [env.name for env in self.envs]
        self.weights = [float(env.config.ratio) for env in self.envs]

    def next_task(self, *, step: int) -> TaskRequest:
        env_name = self.rng.choices(self.env_names, weights=self.weights, k=1)[0]
        return TaskRequest(env_name=env_name, task=next(self.samplers[env_name]), step=step)

    def observe(self, group: list[vf.Episode]) -> None:
        """Report a finalized group to its env's sampler."""
        task_keys = {episode.task.key for episode in group}
        if None in task_keys:
            raise ValueError("A finalized group is missing Task.key")
        if len(task_keys) != 1:
            raise ValueError(f"A finalized group contains multiple task keys: {task_keys}")
        self.samplers[episode_env_name(group[0])].observe(group)

    def metrics(self) -> dict[str, float]:
        return {
            f"sampler/{env_name}/{name}": float(value)
            for env_name, sampler in self.samplers.items()
            for name, value in sampler.metrics().items()
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng": self.rng.getstate(),
            "envs": {name: {"sampler": sampler.state_dict()} for name, sampler in self.samplers.items()},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        expected_fields = {"rng", "envs"}
        if set(state_dict) != expected_fields:
            raise ValueError(f"Train-source checkpoint fields must be {sorted(expected_fields)}")
        env_states = state_dict["envs"]
        if set(env_states) != set(self.samplers):
            raise ValueError(
                f"Train-source checkpoint envs {sorted(env_states)} do not match configured envs "
                f"{sorted(self.samplers)}"
            )
        self.rng.setstate(state_dict["rng"])
        for name, sampler in self.samplers.items():
            sampler.load_state_dict(env_states[name]["sampler"])
