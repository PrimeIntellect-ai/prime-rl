"""Curriculum composition and lifecycle."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import verifiers.v1 as vf

from prime_rl.orchestrator.algo.base import iter_trainable_traces
from prime_rl.orchestrator.curriculum.gates import AdmissionGate
from prime_rl.orchestrator.curriculum.samplers import TaskSampler
from prime_rl.utils.logger import get_logger

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
        self.sampler_type = config.sampler.type
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
        """Observe every result and admit trainable groups that pass every gate."""
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
        return all(decisions) and any(iter_trainable_traces(group))

    def state_dict(self) -> dict[str, Any]:
        return {
            "sampler_type": self.sampler_type,
            "sampler": self.sampler.state_dict(),
            "gates": {name: gate.state_dict() for name, gate in self.gates.items()},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the sampler unless its type changed, and each gate by name. A new
        gate or a sampler of a new type starts fresh; state of a removed gate is dropped."""
        sampler_state = state_dict["sampler"]
        # Checkpoints from before the sampler type was recorded only knew two samplers.
        saved_type = state_dict.get("sampler_type", "standard" if "cursor" in sampler_state else "difficulty_pool")
        if saved_type == self.sampler_type:
            self.sampler.load_state_dict(sampler_state)
        else:
            get_logger().warning(
                f"The task sampler changed from {saved_type!r} to {self.sampler_type!r}, so its state starts fresh"
            )
        gate_states = state_dict["gates"]
        for name, gate in self.gates.items():
            if name in gate_states:
                gate.load_state_dict(gate_states[name])
            else:
                get_logger().info(f"Admission gate {name!r} is not in the checkpoint, so it starts fresh")
        for name in sorted(gate_states.keys() - self.gates.keys()):
            get_logger().info(
                f"Dropped the checkpointed state of admission gate {name!r}, which is no longer configured"
            )

    def metrics(self) -> dict[str, float]:
        metrics = {f"sampler/{name}": float(value) for name, value in self.sampler.metrics().items()}
        for gate_name, gate in self.gates.items():
            metrics |= {f"gate/{gate_name}/{name}": float(value) for name, value in gate.metrics().items()}
        return metrics
