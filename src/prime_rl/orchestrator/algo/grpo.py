from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import AlgorithmConfig, GRPOAdvantageConfig
from prime_rl.orchestrator.algo.advantage import assign_group_norm
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.orchestrator.types import TrainRollout
    from prime_rl.utils.client import InferencePool


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the group mean (optionally
    length-shaped); action tokens feed the ``rl`` loss."""

    def __init__(self, config: AlgorithmConfig, policy_pool: InferencePool, renderer: Renderer | None):
        super().__init__(config, policy_pool, renderer)
        assert isinstance(config.advantage, GRPOAdvantageConfig)
        self.length_penalty = config.advantage.length_penalty

    def assign(self, rollouts: list[TrainRollout]) -> None:
        assign_group_norm(rollouts, self.length_penalty)
