"""Advantage-based training-sample admission."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.orchestrator.curriculum.gates.base import AdmissionGate

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import AdvRangeGateConfig
    from prime_rl.orchestrator.types import Rollout


class AdvRangeGate(AdmissionGate):
    """Reject groups whose trainable-token advantages all fall inside a range.

    The default ``[0, 0]`` interval filters groups with no online learning
    signal. Groups without an advantage stream are admitted.
    """

    def __init__(self, config: AdvRangeGateConfig) -> None:
        self.config = config

    def admit(self, group: list[Rollout]) -> bool:
        advantages: list[float] = []
        for rollout in group:
            if rollout.advantages is None:
                continue
            trainable = [value for sample in rollout.samples for value in sample.mask]
            advantages.extend(advantage for advantage, keep in zip(rollout.advantages, trainable, strict=True) if keep)
        if not advantages:
            return True
        return not all(self.config.reject_min <= advantage <= self.config.reject_max for advantage in advantages)
