"""Advantage-based training-sample admission."""

from __future__ import annotations

from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.orchestrator.curriculum.gates.base import AdmissionGate

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import AdvRangeGateConfig


class AdvRangeGate(AdmissionGate):
    """Reject groups whose trainable-token advantages all fall inside a range.

    The default ``[0, 0]`` interval filters groups with no online learning
    signal. Groups without an advantage stream are admitted.
    """

    def __init__(self, config: AdvRangeGateConfig) -> None:
        self.config = config

    def admit(self, group: list[vf.Episode]) -> bool:
        advantages = [
            advantage
            for episode in group
            for trace in episode.traces
            if not trace.has_error and trace.agent.trainable
            for node in trace.nodes
            for advantage in node.advantages or []
        ]
        if not advantages:
            return True
        return not all(self.config.reject_min <= advantage <= self.config.reject_max for advantage in advantages)
