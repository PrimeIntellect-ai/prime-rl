"""Training source selection and curriculum lifecycle."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from prime_rl.orchestrator.curriculum import Curriculum
from prime_rl.orchestrator.envs import TrainEnvs

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout


class TrainSource:
    """Mix train envs and host one user-authored curriculum per env."""

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
            self.curricula[env.name] = Curriculum.from_config(env.config.curriculum, tasks)

        self.env_names = [env.name for env in self.envs]
        self.weights = [float(env.config.ratio) for env in self.envs]
        self._admitted: dict[str, int] = defaultdict(int)
        self._rejected: dict[str, int] = defaultdict(int)

    def next_example(self) -> dict[str, Any]:
        env_name = self.rng.choices(self.env_names, weights=self.weights, k=1)[0]
        return {
            "env_name": env_name,
            "task": next(self.curricula[env_name].sampler),
        }

    def on_result(self, group: list[Rollout]) -> bool:
        """Report a finalized group and return whether it should train."""
        if not group:
            raise ValueError("Cannot report an empty rollout group")
        env_name = group[0].env_name
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
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.rng.setstate(state_dict["rng"])
        for name, curriculum_state in state_dict["envs"].items():
            curriculum = self.curricula.get(name)
            if curriculum is not None:
                curriculum.load_state_dict(curriculum_state)
