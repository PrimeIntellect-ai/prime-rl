from __future__ import annotations

from prime_rl.orchestrator.algo.base import Algorithm


class SFTDistillAlgorithm(Algorithm):
    """Hard distillation. Needs a teacher: the frozen model that generates the
    rollouts (``sampling.source``); the policy trains with CE on its tokens.

    Assigns no advantage — the ``ce`` loss ignores credit, and SFT trains on
    every sampled token. Reward-based filtering, if wanted, is an explicit
    filter, not smuggled through an unused advantage stream."""

    action_loss_type = "ce"


class StaticSFTAlgorithm(Algorithm):
    """Static SFT: the rollout's tokens come from a dataset (a dataset env),
    not a model — there is nothing to score, so no hooks and no advantages;
    the mask-True tokens feed the ``ce`` loss."""

    action_loss_type = "ce"
