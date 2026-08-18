"""Advantage-based training-sample admission."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.orchestrator.curriculum.gates.base import AdmissionGate

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import AdvRangeGateConfig
    from prime_rl.orchestrator.types import EpisodeRun


class AdvRangeGate(AdmissionGate):
    """Reject groups whose trainable-token advantages all fall inside a range.

    The default ``[0, 0]`` interval filters groups with no online learning
    signal. Groups without an advantage stream are admitted.
    """

    def __init__(self, config: AdvRangeGateConfig) -> None:
        self.config = config

    def admit(self, group: list[EpisodeRun]) -> bool:
        advantages: list[float] = []
        for run in group:
            for trace in run.training:
                if trace.advantages is None:
                    continue
                trainable = [value for sample in trace.samples for value in sample.mask]
                advantages.extend(
                    advantage for advantage, keep in zip(trace.advantages, trainable, strict=True) if keep
                )
        if not advantages:
            return True
        return not all(self.config.reject_min <= advantage <= self.config.reject_max for advantage in advantages)
