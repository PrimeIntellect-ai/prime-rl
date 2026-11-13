"""Advantage-based training-sample admission."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
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
        parts = [
            np.asarray(node.advantages, dtype=np.float32)
            for episode in group
            for trace in episode.traces
            if episode.ok and not trace.has_error and trace.agent.trainable
            for node in trace.nodes
            if node.advantages is not None
        ]
        advantages = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        if not len(advantages):
            return True
        in_range = (advantages >= self.config.reject_min) & (advantages <= self.config.reject_max)
        return not bool(in_range.all())
