from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING

import torch

from prime_rl.configs.algorithm import TOPDAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.utils.client import StaticInferencePool

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.transport import TrainingSample
    from prime_rl.utils.client import InferencePool


class TOPDAlgorithm(Algorithm):
    """Trust Region Policy Distillation (TOP-D, arXiv:2607.04751). Needs a
    teacher, like opd — but where opd's unbounded log-ratio reward diverges
    when the teacher rejects a sampled token, TOP-D distills toward a *proximal
    teacher*, the probability-space interpolation ``α·π_teacher +
    (1−α)·π_sampler``. That construction never needs to be materialized: the
    per-token reward reduces to ``r̃ = log(α·ρ + 1−α)`` with ``ρ`` the
    teacher/sampler probability ratio, floored at ``log(1−α)``.

    The signal compiles to ordinary rl credit rather than a trainer-side KL:
    each sample's trainable tokens get a return ``R̃_k = r̃_k + mean(r̃_{k+1:})``
    (the paper's length-normalized future return, Eq. 8), z-normalized
    *token-level* across the whole group, and shipped as a per-token advantage
    stream on the ``rl`` loss — whose importance ratio against the sampling
    logprobs and trust-region masking are exactly the paper's internal trust
    region iterations. Advantage-based filters apply as they would for grpo."""

    def __init__(self, config: TOPDAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.teacher = config.teacher
        self.alpha = config.alpha
        self.teacher_pool: StaticInferencePool | None = None  # static teacher endpoint, connected in setup()

    async def setup(self) -> None:
        pool = await self.connect(self.teacher)
        if not isinstance(pool, StaticInferencePool):
            raise TypeError("topd teacher must be a static endpoint — prefill scoring needs fixed endpoints")
        self.teacher_pool = pool

    async def score_rollout(self, rollout: Rollout) -> None:
        pool = self.teacher_pool
        assert pool is not None, "teacher pool not connected — Algorithm.setup() must run first"

        async def score_sample(sample: TrainingSample) -> None:
            sample.ref_logprobs = await pool.score(list(sample.token_ids))

        await asyncio.gather(*(score_sample(sample) for sample in rollout.samples))

    def _sample_returns(self, sample: TrainingSample) -> torch.Tensor:
        """Token-level returns over the sample's trainable tokens: the TOP-D
        reward plus the mean of the later trainable tokens' rewards — the
        length-normalized future return of the paper's Eq. 8."""
        assert sample.ref_logprobs is not None, "sample not teacher-scored — score_rollout must run first"
        mask = torch.tensor(sample.mask, dtype=torch.bool)
        teacher_logprobs = torch.tensor(sample.ref_logprobs, dtype=torch.float32)[mask]
        sampler_logprobs = torch.tensor(sample.logprobs, dtype=torch.float32)[mask]
        # r̃ = log(α·ρ + 1−α), via logaddexp so large log-ratios never overflow
        log_ratio = teacher_logprobs - sampler_logprobs
        rewards = torch.logaddexp(log_ratio + math.log(self.alpha), torch.full_like(log_ratio, math.log1p(-self.alpha)))
        future_sum = rewards.flip(0).cumsum(0).flip(0) - rewards
        num_future = torch.arange(len(rewards) - 1, -1, -1, dtype=torch.float32)
        return rewards + future_sum / num_future.clamp(min=1.0)

    async def score_group(self, group: list[Rollout]) -> None:
        returns: list[list[torch.Tensor]] = []
        for rollout in group:
            rollout_returns = []
            for sample in rollout.samples:
                rollout_returns.append(self._sample_returns(sample))
                sample.ref_logprobs = None  # consumed — the rl loss reads advantages, not reference logprobs
            returns.append(rollout_returns)

        flat = [r for rollout_returns in returns for r in rollout_returns]
        all_returns = torch.cat(flat) if flat else torch.empty(0)
        if all_returns.numel() == 0:
            return
        mean = all_returns.mean()
        std = all_returns.std() if all_returns.numel() > 1 else all_returns.new_zeros(())
        std = std.clamp(min=1e-6)  # a group with no spread carries all-zero advantages, like a uniform grpo group

        for rollout, rollout_returns in zip(group, returns, strict=True):
            stream: list[float] = []
            for sample, sample_returns in zip(rollout.samples, rollout_returns, strict=True):
                advantages = iter(((sample_returns - mean) / std).tolist())
                stream.extend(next(advantages) if trainable else 0.0 for trainable in sample.mask)
            rollout.assign_advantages(stream)
