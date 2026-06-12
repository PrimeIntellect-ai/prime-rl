from __future__ import annotations

import asyncio
from itertools import cycle
from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import AdvantageConfig, OPDAdvantageConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.utils import compute_prefill_logprobs

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.orchestrator.types import TrainRollout
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
    model_role = "teacher"

    def __init__(self, advantage: AdvantageConfig, policy_pool: InferencePool, renderer: Renderer | None):
        super().__init__(advantage, policy_pool, renderer)
        assert isinstance(advantage, OPDAdvantageConfig)
        self.max_concurrent = advantage.max_concurrent
        self.teacher = advantage.model
        self.teacher_pool: InferencePool | None = None  # connected in setup()

    async def setup(self) -> None:
        self.teacher_pool = await self.connect(self.teacher)

    async def score(self, rollouts: list[TrainRollout]) -> None:
        pool = self.teacher_pool
        assert pool is not None, "teacher pool not connected — Algorithm.setup() must run first"
        semaphore = asyncio.Semaphore(self.max_concurrent)
        samples = [sample for rollout in rollouts for sample in rollout.samples]

        async def score_sample(client, sample: TrainingSample) -> None:
            async with semaphore:
                token_ids = list(sample.prompt_ids) + list(sample.completion_ids)
                sample.ref_logprobs = await compute_prefill_logprobs(client, pool.model_name, token_ids)

        await asyncio.gather(
            *[score_sample(client, sample) for client, sample in zip(cycle(pool.train_clients), samples)]
        )
