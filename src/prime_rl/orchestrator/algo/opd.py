from __future__ import annotations

import asyncio
from itertools import cycle
from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import AlgorithmConfig, OPDAlgorithmConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.utils.client import PrefillClient

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.orchestrator.types import RolloutView
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

    def __init__(self, config: AlgorithmConfig, policy_pool: InferencePool, renderer: Renderer | None):
        super().__init__(config, policy_pool, renderer)
        assert isinstance(config, OPDAlgorithmConfig)
        self.max_concurrent = config.max_concurrent
        self.teacher = config.model
        self.teacher_pool: InferencePool | None = None  # connected in setup()

    async def setup(self) -> None:
        self.teacher_pool = await self.connect(self.teacher)

    async def score_batch(self, batch: list[RolloutView]) -> None:
        pool = self.teacher_pool
        assert pool is not None, "teacher pool not connected — Algorithm.setup() must run first"
        semaphore = asyncio.Semaphore(self.max_concurrent)
        samples = [sample for view in batch for sample in view.samples]
        clients = [PrefillClient(c) for c in pool.train_clients]

        async def score_sample(client: PrefillClient, sample: TrainingSample) -> None:
            async with semaphore:
                sample.ref_logprobs = await client.score(pool.model_name, list(sample.token_ids))

        try:
            await asyncio.gather(*[score_sample(client, sample) for client, sample in zip(cycle(clients), samples)])
        finally:
            await asyncio.gather(*(client.close() for client in clients))
