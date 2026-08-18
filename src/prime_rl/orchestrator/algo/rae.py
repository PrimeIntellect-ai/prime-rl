from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import RAEAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import TrainingTrace
    from prime_rl.utils.client import InferencePool


class RAEAlgorithm(Algorithm):
    """Role-conditioned advantage estimation (SPIRAL, arXiv:2506.24119):
    credit = reward minus a per-agent EMA baseline of that agent's rewards.

    The advantage estimator for multi-agent self-play envs (kuhn-poker and
    friends): in a zero-sum game the group mean is ~0 whatever the policy does,
    and centering all agents against it mixes their opposite reward scales — a
    structural first-mover edge would read as permanent credit for one agent.
    A per-agent baseline tracks each agent's own expected reward instead, so
    credit measures improvement over that agent's usual result. The instance is
    per-env, so baselines are keyed per (env, agent) — the paper's per
    (game, role). A single-agent env degrades to REINFORCE with an EMA baseline.

    Each trace is scored against the baseline *before* its reward updates it
    (the reference implementation's unbiased order). Baselines start at 0 and
    live in orchestrator memory: a restart re-warms them over ~1/(1 − decay)
    traces per agent."""

    def __init__(self, config: RAEAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.decay = config.decay
        self.baselines: dict[str, float] = defaultdict(float)

    async def score_group(self, group: list[TrainingTrace]) -> None:
        for rollout in group:
            trace = rollout.trace
            baseline = self.baselines[trace.agent.name]
            rollout.assign_advantages(trace.reward - baseline)
            self.baselines[trace.agent.name] = self.decay * baseline + (1.0 - self.decay) * trace.reward
