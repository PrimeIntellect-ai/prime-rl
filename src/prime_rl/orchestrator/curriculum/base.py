"""Task sampling and admission interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import verifiers.v1 as vf

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import CurriculumConfig
    from prime_rl.orchestrator.types import Rollout


class TaskSampler(Iterator[vf.Task], ABC):
    """Base class for user-authored task selection policies."""

    @abstractmethod
    def __next__(self) -> vf.Task:
        """Choose the next task."""
        raise NotImplementedError

    def observe(self, group: list[Rollout]) -> None:
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

    def admit(self, group: list[Rollout]) -> bool:
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

    def __init__(
        self,
        config: CurriculumConfig | None,
        tasks: Sequence[vf.Task] | Iterator[vf.Task],
    ) -> None:
        from prime_rl.configs.orchestrator import (
            AdvRangeGateConfig,
            CurriculumConfig,
            DifficultyPoolSamplerConfig,
            StandardSamplerConfig,
        )
        from prime_rl.orchestrator.curriculum.adv_range_gate import AdvRangeGate
        from prime_rl.orchestrator.curriculum.difficulty_pool_sampler import DifficultyPoolSampler
        from prime_rl.orchestrator.curriculum.standard_sampler import StandardSampler

        config = CurriculumConfig() if config is None else config
        if isinstance(config.sampler, StandardSamplerConfig):
            self.sampler: TaskSampler = StandardSampler(tasks)
        elif isinstance(config.sampler, DifficultyPoolSamplerConfig):
            self.sampler = DifficultyPoolSampler(config.sampler, tasks)
        else:
            raise TypeError(f"Unsupported task sampler config: {type(config.sampler).__name__}")

        self.gates: dict[str, AdmissionGate] = {}
        for name, gate_config in config.gates.items():
            if isinstance(gate_config, AdvRangeGateConfig):
                gate: AdmissionGate = AdvRangeGate(gate_config)
            else:
                raise TypeError(f"Unsupported admission gate config: {type(gate_config).__name__}")
            self.gates[name] = gate

    def on_result(self, group: list[Rollout]) -> bool:
        """Observe every result, evaluate every gate, and combine with AND."""
        if not group:
            raise ValueError("Cannot report an empty rollout group")
        task_keys = {rollout.task.key for rollout in group}
        if None in task_keys:
            raise ValueError("A finalized group is missing Task.key")
        if len(task_keys) != 1:
            raise ValueError(f"A finalized group contains multiple task keys: {task_keys}")
        self.sampler.observe(group)
        decisions: list[bool] = []
        for name, gate in self.gates.items():
            decision = gate.admit(group)
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
