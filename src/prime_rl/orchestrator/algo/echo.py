from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import AlgorithmConfig, EchoAdvantageConfig
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.utils.utils import import_object

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.utils.client import InferencePool


class EchoAlgorithm(GRPOAlgorithm):
    """GRPO on action tokens, plus weighted CE on env-provided tokens of
    later turns (tool output, user feedback), selected by message role —
    tool-response bodies at the vetted default. Selected tokens feed the
    ``ce`` loss component at their role's ``alpha`` and stay outside the rl
    mask and its denominator. An optional user filter narrows the selection
    per rollout (e.g. dropping tool-output warnings)."""

    def __init__(self, config: AlgorithmConfig, policy_pool: InferencePool, renderer: Renderer | None):
        super().__init__(config, policy_pool, renderer)
        advantage = config.advantage
        assert isinstance(advantage, EchoAdvantageConfig)
        self.echo_roles = {
            role: role_config.alpha
            for role in ("system", "user", "assistant", "tool")
            if (role_config := getattr(advantage.roles, role)) is not None
        }
        if advantage.filter is not None:
            filter_fn = import_object(advantage.filter.import_path)
            self.echo_filter_fn = partial(filter_fn, **advantage.filter.kwargs)
