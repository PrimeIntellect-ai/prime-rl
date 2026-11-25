"""Advantage-based training-sample admission."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import verifiers.v1 as vf

from prime_rl.orchestrator.credit import advantage_values
from prime_rl.orchestrator.curriculum.gates.base import AdmissionGate

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import AdvRangeGateConfig


class AdvRangeGate(AdmissionGate):
    """Reject groups whose advantages (one per trace, or per token) all fall inside a range.

    The default ``[0, 0]`` interval filters groups with no online learning
    signal. Groups without advantages are admitted.
    """

    def __init__(self, config: AdvRangeGateConfig) -> None:
        self.config = config

    def admit(self, group: list[vf.Episode]) -> bool:
        parts = [
            values
            for episode in group
            for trace in episode.traces
            if episode.ok and not trace.has_error and trace.agent.trainable
            if (values := advantage_values(trace)) is not None
        ]
        if not parts:
            return True
        advantages = np.concatenate(parts)
        return not bool(((advantages >= self.config.reject_min) & (advantages <= self.config.reject_max)).all())
