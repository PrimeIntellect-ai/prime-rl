from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.configs.algorithm import OPDAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_reference_logprobs
from prime_rl.orchestrator.trajectories import iter_trainable_branches

if TYPE_CHECKING:
    from prime_rl.utils.client import InferencePool


class OPDAlgorithm(Algorithm):
    """On-policy distillation. Needs a teacher: the frozen reference model the
    per-token reverse KL is computed against.

    The policy samples its own rollouts; at ship time each sample's full
    context is prefill-scored under the teacher (``ref_logprobs`` on the
    wire), and the trainer evaluates the KL against the live policy. No
    credit is assigned — rollouts keep ``advantages=None`` and samples ship no
    advantage stream; ``group_size`` only fans out sampling."""

    action_loss_type = "ref_kl"

    def __init__(self, config: OPDAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.teacher = config.teacher
        self.teacher_pool: InferencePool | None = None  # frozen teacher endpoint, connected in setup()

    async def setup(self) -> None:
        self.teacher_pool = await self.connect(self.teacher)

    async def score_episode(self, episode: vf.Episode) -> None:
        pool = self.teacher_pool
        assert pool is not None, "teacher pool not connected — Algorithm.setup() must run first"
        branches = [
            branch for _, trace in iter_trainable_traces([episode]) for branch, _ in iter_trainable_branches(trace)
        ]
        scores = await asyncio.gather(*(pool.score(branch.token_ids) for branch in branches))
        for branch, logprobs in zip(branches, scores, strict=True):
            assign_reference_logprobs(branch, logprobs)
