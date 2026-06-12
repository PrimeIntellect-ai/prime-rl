from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import AlgorithmConfig, CustomAdvantageConfig
from prime_rl.orchestrator.algo.advantage import AdvantageInputs, AdvantageOutputs, assign_advantages
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.utils.utils import import_object

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.orchestrator.types import TrainRollout
    from prime_rl.utils.client import InferencePool


class CustomAlgorithm(Algorithm):
    """User-supplied advantage function: one scalar per rollout, optionally
    with per-token advantages aligned to each rollout's completion tokens."""

    def __init__(self, config: AlgorithmConfig, policy_pool: InferencePool, renderer: Renderer | None):
        super().__init__(config, policy_pool, renderer)
        assert isinstance(config.advantage, CustomAdvantageConfig)
        custom_fn = import_object(config.advantage.import_path)
        kwargs = config.advantage.kwargs

        def advantage_fn(inputs: AdvantageInputs) -> AdvantageOutputs:
            return custom_fn(inputs, **kwargs)

        self.advantage_fn = advantage_fn

    def assign(self, rollouts: list[TrainRollout]) -> None:
        assign_advantages(rollouts, self.advantage_fn)
