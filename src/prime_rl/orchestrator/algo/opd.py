from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import OPDAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.utils.client import StaticInferencePool

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.transport import TrainingSample
    from prime_rl.utils.client import InferencePool


class OPDAlgorithm(Algorithm):
    """On-policy distillation. Needs a teacher: the frozen reference model the
    per-token reverse KL is computed against.

    The policy samples its own rollouts; at ship time each sample's full
    context is prefill-scored under the teacher (``ref_logprobs`` on the
    wire), and the trainer evaluates the KL against the live policy. No
    credit is assigned — rollouts keep ``advantages=None`` (advantage-based
    filters never fire) and samples ship no advantage stream; ``group_size``
    only fans out sampling."""

    action_loss_type = "ref_kl"

    def __init__(self, config: OPDAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.max_concurrent = config.max_concurrent
        self.teacher = config.teacher
        self.teacher_pool: StaticInferencePool | None = None  # static teacher endpoint, connected in setup()
        self._semaphore: asyncio.Semaphore | None = None  # bounds concurrent teacher queries; created in setup()

    async def setup(self) -> None:
        pool = await self.connect(self.teacher)
        if not isinstance(pool, StaticInferencePool):
            raise TypeError("opd teacher must be a static endpoint — prefill scoring needs fixed endpoints")
        self.teacher_pool = pool
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    async def score_rollout(self, rollout: Rollout) -> None:
        pool = self.teacher_pool
        semaphore = self._semaphore
        assert pool is not None and semaphore is not None, "Algorithm.setup() must run first"

        async def score_sample(sample: TrainingSample) -> None:
            async with semaphore:
                sample.ref_logprobs = await pool.score(list(sample.token_ids))

        await asyncio.gather(*(score_sample(sample) for sample in rollout.samples))
