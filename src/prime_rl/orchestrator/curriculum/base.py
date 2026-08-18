"""User-authored task sampling and admission interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import verifiers.v1 as vf

from prime_rl.utils.utils import import_object

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import AdmissionGateConfig, CurriculumConfig, TaskSamplerConfig
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


class TaskSampler(ABC):
    """Base class for user-authored task selection policies."""

    @abstractmethod
    def sample(self) -> vf.Task:
        """Choose the next task."""
        raise NotImplementedError

    def observe(self, result: CurriculumResult) -> None:
        """Update sampling state from a finalized group."""

    def state_dict(self) -> dict[str, Any]:
        """Return checkpoint state owned by this sampler."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore checkpoint state before sampling resumes."""

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


def _setup_sampler(
    config: TaskSamplerConfig,
    tasks: Sequence[vf.Task] | Iterator[vf.Task],
) -> TaskSampler:
    sampler_type = import_object(config.import_path)
    sampler = sampler_type(tasks, **config.kwargs)
    if not isinstance(sampler, TaskSampler):
        raise TypeError(f"{config.import_path} must subclass TaskSampler")
    return sampler


def _setup_gate(config: AdmissionGateConfig) -> AdmissionGate:
    gate_type = import_object(config.import_path)
    gate = gate_type(**config.kwargs)
    if not isinstance(gate, AdmissionGate):
        raise TypeError(f"{config.import_path} must subclass AdmissionGate")
    return gate


def setup_curriculum(
    config: CurriculumConfig | None,
    tasks: Sequence[vf.Task] | Iterator[vf.Task],
) -> Curriculum:
    from prime_rl.orchestrator.curriculum.standard_sampler import StandardSampler

    sampler = (
        StandardSampler(tasks) if config is None or config.sampler is None else _setup_sampler(config.sampler, tasks)
    )
    gates = {} if config is None else {name: _setup_gate(gate) for name, gate in config.gates.items()}
    return Curriculum(sampler, gates)
