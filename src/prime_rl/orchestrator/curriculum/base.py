"""Curriculum composition and lifecycle."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import verifiers.v1 as vf

from prime_rl.orchestrator.curriculum.gates import AdmissionGate
from prime_rl.orchestrator.curriculum.samplers import TaskSampler

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import CurriculumConfig


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
        from prime_rl.orchestrator.curriculum.gates import AdvRangeGate
        from prime_rl.orchestrator.curriculum.samplers import DifficultyPoolSampler, StandardSampler

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

    def on_result(self, group: list[vf.Episode]) -> bool:
        """Observe every result, evaluate every gate, and combine with AND."""
        if not group:
            raise ValueError("Cannot report an empty rollout group")
        task_keys = {episode.task.key for episode in group}
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
        expected_fields = {"sampler", "gates"}
        if set(state_dict) != expected_fields:
            raise ValueError(f"Curriculum checkpoint fields must be {sorted(expected_fields)}")
        gate_states = state_dict["gates"]
        if set(gate_states) != set(self.gates):
            raise ValueError(
                f"Curriculum checkpoint gates {sorted(gate_states)} do not match configured gates {sorted(self.gates)}"
            )
        self.sampler.load_state_dict(state_dict["sampler"])
        for name, gate in self.gates.items():
            gate.load_state_dict(gate_states[name])

    def metrics(self) -> dict[str, float]:
        metrics = {f"sampler/{name}": float(value) for name, value in self.sampler.metrics().items()}
        for gate_name, gate in self.gates.items():
            metrics |= {f"gate/{gate_name}/{name}": float(value) for name, value in gate.metrics().items()}
        return metrics
