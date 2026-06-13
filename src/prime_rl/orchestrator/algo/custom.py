from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import AdvantageConfig, CustomAdvantageConfig
from prime_rl.orchestrator.algo.advantage import apply_advantage_fn
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.utils.utils import import_object

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.orchestrator.types import TrainRollout
    from prime_rl.utils.client import InferencePool


class CustomAlgorithm(Algorithm):
    """User-supplied advantage function — the ``assign_advantages`` hook body
    without the class: receives the group's ``TrainRollout``\\ s, returns
    per-token advantages, one list per rollout aligned to its completion
    tokens (``broadcast`` spreads scalar group credit)."""

    def __init__(self, advantage: AdvantageConfig, policy_pool: InferencePool, renderer: Renderer | None):
        super().__init__(advantage, policy_pool, renderer)
        assert isinstance(advantage, CustomAdvantageConfig)
        custom_fn = import_object(advantage.import_path)
        kwargs = advantage.kwargs

        def advantage_fn(rollouts: list[TrainRollout]) -> list[list[float]]:
            return custom_fn(rollouts, **kwargs)

        self.advantage_fn = advantage_fn

    def assign_advantages(self, rollouts: list[TrainRollout]) -> None:
        apply_advantage_fn(rollouts, self.advantage_fn)
