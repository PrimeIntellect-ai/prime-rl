from __future__ import annotations

from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.configs.algorithm import DebugAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


class DebugAlgorithm(Algorithm):
    """Debugging credit assignment: the same constant advantage (default 1.0)
    on every sampled token of every clean trainable trace, ignoring rewards
    entirely. For infra work — every rollout is trainable and every action
    token carries gradient signal, so the full RL path (advantage transport,
    importance ratios, trust region, optimizer, weight update) can be exercised
    regardless of the reward function. Not a real training signal: a policy
    pushed by it drifts monotonically."""

    def __init__(self, config: DebugAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.advantage = config.advantage

    async def score_episode(self, episode: vf.Episode) -> None:
        for _, trace in iter_trainable_traces([episode]):
            assign_advantages(trace, self.advantage)
