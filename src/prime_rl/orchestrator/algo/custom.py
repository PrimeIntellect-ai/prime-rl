from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import AlgorithmConfig, CustomAlgorithmConfig
from prime_rl.orchestrator.algo.advantage import apply_advantage_fn
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.utils.utils import import_object

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


class CustomAlgorithm(Algorithm):
    """User-supplied advantage function — the ``score_group`` hook body without
    the class: receives the group's ``Rollout``\\ s, returns one value per
    rollout (a scalar broadcast over its completion tokens, or a per-token
    list)."""

    def __init__(self, config: AlgorithmConfig, policy_pool: InferencePool, renderer: Renderer | None):
        super().__init__(config, policy_pool, renderer)
        assert isinstance(config, CustomAlgorithmConfig)
        custom_fn = import_object(config.import_path)
        kwargs = config.kwargs

        def advantage_fn(group: list[Rollout]) -> list[float | list[float]]:
            return custom_fn(group, **kwargs)

        self.advantage_fn = advantage_fn

    async def score_group(self, group: list[Rollout]) -> None:
        apply_advantage_fn(group, self.advantage_fn)
