"""Advantage-based training-sample admission."""

from __future__ import annotations

from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.orchestrator.credit import trace_advantage
from prime_rl.orchestrator.curriculum.gates.base import AdmissionGate

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import AdvRangeGateConfig


class AdvRangeGate(AdmissionGate):
    """Reject groups whose trace advantages all fall inside a range.

    The default ``[0, 0]`` interval filters groups with no online learning
    signal. Groups without advantages are admitted.
    """

    def __init__(self, config: AdvRangeGateConfig) -> None:
        self.config = config

    def admit(self, group: list[vf.Episode]) -> bool:
        advantages = [
            advantage
            for episode in group
            for trace in episode.traces
            if episode.ok and not trace.has_error and trace.agent.trainable
            if (advantage := trace_advantage(trace)) is not None
        ]
        if not advantages:
            return True
        return not all(self.config.reject_min <= advantage <= self.config.reject_max for advantage in advantages)
