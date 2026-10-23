"""Training source selection and curriculum lifecycle."""

from __future__ import annotations

import random
import uuid
from collections import defaultdict
from typing import Any

import verifiers.v1 as vf

from prime_rl.orchestrator.curriculum import Curriculum
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.types import TaskRequest
from prime_rl.orchestrator.utils import episode_env_name


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
            self.curricula[env.name] = Curriculum(env.config.curriculum, tasks)
            perfect_size = getattr(self.curricula[env.name].sampler, "fully_solved_group_size", None)
            if perfect_size is not None and perfect_size != env.config.group_size:
                raise ValueError(f"{env.name}: fully_solved_group_size must match group_size")

        self.env_names = [env.name for env in self.envs]
        self.weights = [float(env.config.ratio) for env in self.envs]
        self._admitted: dict[str, int] = defaultdict(int)
        self._rejected: dict[str, int] = defaultdict(int)
        self.reservations: dict[str, dict[str, Any]] = {}

    def next_task(self, *, step: int) -> TaskRequest | None:
        excluded = {
            name: {item["task_key"] for item in self.reservations.values() if item["env_name"] == name}
            for name in self.env_names
        }
        available = [
            (name, weight)
            for name, weight in zip(self.env_names, self.weights, strict=True)
            if weight > 0 and self.curricula[name].sampler.has_available(excluded[name])
        ]
        if not available:
            return None
        env_name = self.rng.choices([name for name, _ in available], weights=[weight for _, weight in available], k=1)[
            0
        ]
        sampler = self.curricula[env_name].sampler
        initial_sweep = bool(getattr(sampler, "first_cycle", False))
        task = sampler.next_available(excluded[env_name])
        assert task is not None
        group_id = None
        if sampler.tracks_inflight:
            group_id = str(uuid.uuid4())
            self.reservations[group_id] = {"env_name": env_name, "task_key": task.key, "initial_sweep": initial_sweep}
        return TaskRequest(env_name=env_name, task=task, step=step, group_id=group_id)

    def release_group(self, group_id: str) -> None:
        self.reservations.pop(group_id, None)

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
        never_solved: list[float] = []
        for env_name, curriculum in self.curricula.items():
            if curriculum.sampler.tracks_inflight:
                metrics[f"curriculum/{env_name}/sampling/active_groups"] = float(
                    sum(item["env_name"] == env_name for item in self.reservations.values())
                )
            admitted = self._admitted.pop(env_name, 0)
            rejected = self._rejected.pop(env_name, 0)
            total = admitted + rejected
            if total:
                metrics[f"curriculum/{env_name}/admission_rate"] = admitted / total
            curriculum_metrics = curriculum.metrics()
            metrics |= {f"curriculum/{env_name}/{name}": float(value) for name, value in curriculum_metrics.items()}
            if "sampler/sampling/never_solved" in curriculum_metrics:
                never_solved.append(curriculum_metrics["sampler/sampling/never_solved"])
        if len(never_solved) == len(self.curricula):
            metrics["curriculum/agg/sampler/sampling/never_solved"] = sum(never_solved)
        return metrics

    def on_episode(self, episode: vf.Episode) -> None:
        self.curricula[episode_env_name(episode)].sampler.observe_episode(episode)

    def on_train_batch(self, episodes: list[vf.Episode]) -> None:
        """Count contributions only after the effective cohort is sent to training."""
        by_env: dict[str, list[vf.Episode]] = defaultdict(list)
        for episode in episodes:
            by_env[episode_env_name(episode)].append(episode)
        for env_name, cohort in by_env.items():
            self.curricula[env_name].sampler.on_train_batch(cohort)

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng": self.rng.getstate(),
            "envs": {name: curriculum.state_dict() for name, curriculum in self.curricula.items()},
            "reservations": {key: dict(value) for key, value in self.reservations.items()},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        expected_fields = {"rng", "envs", "reservations"}
        if set(state_dict) != expected_fields:
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
        # Network calls are not restored. Do not retain orphaned reservations:
        # return their unfinished sweep entries to the sampler instead.
        reservations = state_dict["reservations"]
        for item in reservations.values():
            name = item["env_name"]
            if name not in self.curricula or type(item["initial_sweep"]) is not bool:
                raise ValueError("Invalid checkpoint reservation")
            self.curricula[name].sampler.requeue_abandoned(item["task_key"], item["initial_sweep"])
        self.reservations = {}
