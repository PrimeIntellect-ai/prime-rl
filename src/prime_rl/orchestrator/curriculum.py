"""User-authored task sampling and admission policies."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import verifiers.v1 as vf

from prime_rl.utils.utils import import_object

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import CurriculumComponentConfig, CurriculumConfig
    from prime_rl.orchestrator.types import Rollout


@dataclass(frozen=True)
class CurriculumResult:
    """One finalized task group, after its training samples and credit are built."""

    task_key: str
    rollouts: tuple[Rollout, ...]

    @classmethod
    def from_rollouts(cls, rollouts: list[Rollout]) -> CurriculumResult:
        if not rollouts:
            raise ValueError("A curriculum result needs at least one rollout")
        keys = {rollout.task.key for rollout in rollouts}
        if None in keys:
            raise ValueError("A curriculum result is missing Task.key")
        if len(keys) != 1:
            raise ValueError(f"A curriculum result contains multiple task keys: {keys}")
        task_key = keys.pop()
        assert task_key is not None
        return cls(task_key=task_key, rollouts=tuple(rollouts))


class TaskSampler:
    """Default task sampler and base class for user-authored selection policies."""

    def __init__(self, tasks: Sequence[vf.Task] | Iterator[vf.Task], *, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.epoch = 1
        self.cursor = 0

        if isinstance(tasks, Sequence):
            if not tasks:
                raise ValueError("A finite curriculum needs at least one task")
            self.tasks: tuple[vf.Task, ...] | None = tuple(tasks)
            self.task_iterator: Iterator[vf.Task] | None = None
            keys = [task.key for task in self.tasks]
            duplicates = {key for key, count in Counter(keys).items() if count > 1}
            if duplicates:
                raise ValueError(f"Task keys must be unique within a taskset: {sorted(duplicates)}")
            self.tasks_by_key = dict(zip(keys, self.tasks))
            self._epoch_tasks = self._shuffle()
        else:
            self.tasks = None
            self.task_iterator = tasks
            self.tasks_by_key: dict[str, vf.Task] = {}
            self._epoch_tasks: list[vf.Task] | None = None

    def _shuffle(self) -> list[vf.Task] | None:
        if self.tasks is None:
            return None
        tasks = list(self.tasks)
        random.Random(self.epoch).shuffle(tasks)
        return tasks

    def sample(self) -> vf.Task:
        """Choose the next task.

        The default preserves prime-rl's epoch-shuffled finite iteration and
        sequential infinite-taskset iteration.
        """
        if self._epoch_tasks is None:
            assert self.task_iterator is not None
            task = next(self.task_iterator)
            self.cursor += 1
            return task
        if self.cursor >= len(self._epoch_tasks):
            self.epoch += 1
            self.cursor = 0
            self._epoch_tasks = self._shuffle()
            assert self._epoch_tasks is not None
        task = self._epoch_tasks[self.cursor]
        self.cursor += 1
        return task

    def observe(self, result: CurriculumResult) -> None:
        """Update sampling state from a finalized group."""

    def state_dict(self) -> dict[str, Any]:
        """Return checkpoint state owned by this sampler."""
        return {
            "rng": self.rng.getstate(),
            "epoch": self.epoch,
            "cursor": self.cursor,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore checkpoint state before sampling resumes."""
        if "rng" in state_dict:
            self.rng.setstate(state_dict["rng"])
        self.epoch = state_dict["epoch"]
        self.cursor = state_dict["cursor"]
        if self.tasks is None:
            assert self.task_iterator is not None
            for _ in range(self.cursor):
                next(self.task_iterator)
        else:
            self._epoch_tasks = self._shuffle()

    def metrics(self) -> dict[str, float]:
        """Return metrics relative to this sampler's namespace."""
        return {}


class AdmissionGate:
    """Base class for user-authored training-sample admission policies."""

    def admit(self, result: CurriculumResult) -> bool:
        """Return whether a finalized group should enter the training batch."""
        return True

    def state_dict(self) -> dict[str, Any]:
        """Return checkpoint state owned by this gate."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore checkpoint state before results resume."""

    def metrics(self) -> dict[str, float]:
        """Return metrics relative to this gate's namespace."""
        return {}


class Curriculum:
    """One task sampler composed with zero or more admission gates."""

    def __init__(self, sampler: TaskSampler, gates: dict[str, AdmissionGate] | None = None) -> None:
        self.sampler = sampler
        self.gates = {} if gates is None else dict(gates)

    def sample(self) -> vf.Task:
        return self.sampler.sample()

    def on_result(self, result: CurriculumResult) -> bool:
        """Observe every result, evaluate every gate, and combine with AND."""
        self.sampler.observe(result)
        decisions: list[bool] = []
        for name, gate in self.gates.items():
            decision = gate.admit(result)
            if not isinstance(decision, bool):
                raise TypeError(f"AdmissionGate {name!r}.admit() must return bool, got {type(decision).__name__}")
            decisions.append(decision)
        return all(decisions)

    def state_dict(self) -> dict[str, Any]:
        return {
            "sampler": self.sampler.state_dict(),
            "gates": {name: gate.state_dict() for name, gate in self.gates.items()},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "sampler" not in state_dict:
            self.sampler.load_state_dict(state_dict)
            return
        self.sampler.load_state_dict(state_dict["sampler"])
        for name, gate_state in state_dict["gates"].items():
            gate = self.gates.get(name)
            if gate is not None:
                gate.load_state_dict(gate_state)

    def metrics(self) -> dict[str, float]:
        metrics = {f"sampler/{name}": float(value) for name, value in self.sampler.metrics().items()}
        for gate_name, gate in self.gates.items():
            metrics |= {f"gate/{gate_name}/{name}": float(value) for name, value in gate.metrics().items()}
        return metrics


def _setup_component(config: CurriculumComponentConfig, base_type: type, *args: Any) -> Any:
    component_type = import_object(config.import_path)
    component = component_type(*args, **config.kwargs)
    if not isinstance(component, base_type):
        raise TypeError(f"{config.import_path} must subclass {base_type.__name__}")
    return component


def setup_curriculum(
    config: CurriculumConfig | None,
    tasks: Sequence[vf.Task] | Iterator[vf.Task],
) -> Curriculum:
    sampler = (
        TaskSampler(tasks)
        if config is None or config.sampler is None
        else _setup_component(config.sampler, TaskSampler, tasks)
    )
    gates = (
        {} if config is None else {name: _setup_component(gate, AdmissionGate) for name, gate in config.gates.items()}
    )
    return Curriculum(sampler, gates)
