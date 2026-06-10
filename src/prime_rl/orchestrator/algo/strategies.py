"""Runtime advantage strategies — one object per ``AdvantageConfig`` union member.

Each strategy owns both execution points of the training signal: group-time
scalar assignment (``assign``, cheap and synchronous) and ship-time reference
scoring (``score``, async inference against the strategy's reference pool with
bounded concurrency).
"""

from __future__ import annotations

import asyncio
from itertools import cycle
from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import (
    AdvantageConfig,
    CustomAdvantageConfig,
    DemoRefKLAdvantageConfig,
    GroupNormAdvantageConfig,
    LengthPenaltyConfig,
    RefKLAdvantageConfig,
    RewardAdvantageConfig,
    SupervisedAdvantageConfig,
)
from prime_rl.orchestrator.algo.advantage import (
    AdvantageInputs,
    AdvantageOutputs,
    assign_advantages,
    default_advantage_fn,
)
from prime_rl.orchestrator.utils import compute_prefill_logprobs
from prime_rl.transport import TrainingSample
from prime_rl.utils.utils import import_object

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

    from prime_rl.orchestrator.types import TrainRollout
    from prime_rl.utils.client import InferencePool


def _assign_group_norm(rollouts: list[TrainRollout], length_penalty: LengthPenaltyConfig | None) -> None:
    assign_advantages(rollouts, lambda inputs: default_advantage_fn(inputs, length_penalty=length_penalty))


class AdvantageStrategy:
    """Runtime counterpart of one ``AdvantageConfig`` union member.

    Two execution points: ``assign`` runs at group finalization and sets
    rollout-level scalars; ``score`` runs at batch-ship time and attaches
    per-token reference data by querying its reference model. Subclasses
    override what they use — the defaults assign no scalars (rollouts keep
    ``advantage=None``, so advantage-based filters skip them) and score
    nothing."""

    def assign(self, rollouts: list[TrainRollout]) -> None:
        pass

    async def score(self, rollouts: list[TrainRollout]) -> None:
        pass


class GroupNormAdvantage(AdvantageStrategy):
    """GRPO: scalar advantage = reward minus the per-group mean baseline."""

    def __init__(self, config: GroupNormAdvantageConfig):
        self.length_penalty = config.length_penalty

    def assign(self, rollouts: list[TrainRollout]) -> None:
        _assign_group_norm(rollouts, self.length_penalty)


class RewardAdvantage(AdvantageStrategy):
    """Scalar advantage = raw reward, no group baseline."""

    def assign(self, rollouts: list[TrainRollout]) -> None:
        assign_advantages(rollouts, None)


class CustomAdvantage(AdvantageStrategy):
    """User-supplied advantage function: one scalar per rollout, optionally
    with per-token advantages aligned to each rollout's completion tokens."""

    def __init__(self, config: CustomAdvantageConfig):
        custom_fn = import_object(config.import_path)
        kwargs = config.kwargs

        def advantage_fn(inputs: AdvantageInputs) -> AdvantageOutputs:
            return custom_fn(inputs, **kwargs)

        self.advantage_fn = advantage_fn

    def assign(self, rollouts: list[TrainRollout]) -> None:
        assign_advantages(rollouts, self.advantage_fn)


class SupervisedAdvantage(AdvantageStrategy):
    """SFT distillation: the CE loss type ignores scalars, but group-relative
    scalars are still assigned so reward-based filtering keeps working."""

    def assign(self, rollouts: list[TrainRollout]) -> None:
        _assign_group_norm(rollouts, None)


class RefKLAdvantage(AdvantageStrategy):
    """On-policy distillation: group-relative scalars (their sign steers the
    DPPO masking direction in the ``ref_kl`` loss type) plus
    ``TrainingSample.ref_logprobs`` from scoring each sample's own context
    under the reference model."""

    def __init__(self, config: RefKLAdvantageConfig):
        assert config.model is not None
        self.config = config
        self.pool: InferencePool | None = None  # resolved by Algorithm.setup

    def assign(self, rollouts: list[TrainRollout]) -> None:
        _assign_group_norm(rollouts, self.config.length_penalty)

    async def score(self, rollouts: list[TrainRollout]) -> None:
        pool = self.pool
        assert pool is not None, "reference pool not set — Algorithm.setup() must run before scoring"
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        samples = [sample for rollout in rollouts for sample in rollout.samples]

        async def score_sample(client, sample: TrainingSample) -> None:
            async with semaphore:
                token_ids = list(sample.prompt_ids) + list(sample.completion_ids)
                sample.ref_logprobs = await compute_prefill_logprobs(client, pool.model_name, token_ids)

        await asyncio.gather(
            *[score_sample(client, sample) for client, sample in zip(cycle(pool.train_clients), samples)]
        )


class DemoRefKLAdvantage(AdvantageStrategy):
    """Self-distillation (SDFT): fill ``TrainingSample.ref_logprobs`` by
    scoring each sample's completion under the reference model conditioned on
    an expert demonstration. No scalar advantage is assigned.

    The scoring prefix is rebuilt from the rollout's first-turn prompt
    messages with the demonstration woven into the last user message; the
    returned completion logprobs are aligned back onto the sample (the
    sample's prompt positions are 0.0 and stay outside the loss mask).
    """

    def __init__(self, config: DemoRefKLAdvantageConfig, tokenizer: PreTrainedTokenizer):
        assert config.model is not None
        self.config = config
        self.tokenizer = tokenizer
        self.pool: InferencePool | None = None  # resolved by Algorithm.setup

    def _ref_prefix_ids(self, rollout: TrainRollout) -> list[int]:
        trajectory = rollout.raw.get("trajectory") or []
        if len(trajectory) != 1:
            raise ValueError(
                f"demo_ref_kl supports single-step trajectories only; "
                f"env '{rollout.env_name}' produced {len(trajectory)} steps."
            )
        info = rollout.raw.get("info") or {}
        demonstration = info.get(self.config.demo_key) if isinstance(info, dict) else None
        if demonstration is None:
            demonstration = rollout.raw.get(self.config.demo_key)
        if demonstration is None:
            raise ValueError(
                f"demo_ref_kl requires '{self.config.demo_key}' in the example's info dict or as a "
                f"top-level rollout field (env '{rollout.env_name}', example {rollout.example_id})."
            )

        messages = [dict(m) for m in trajectory[0]["prompt"]]
        user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if not user_indices:
            raise ValueError(f"demo_ref_kl found no user message to condition (env '{rollout.env_name}').")
        last_user = messages[user_indices[-1]]
        question = last_user.get("content")
        if not isinstance(question, str):
            raise ValueError("demo_ref_kl supports text-only prompts (user content must be a string).")
        last_user["content"] = self.config.template.format(question=question, demonstration=demonstration)

        # ``return_dict=False`` pins the flat token-id list — newer
        # transformers default to returning a BatchEncoding.
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=False
        )

    async def score(self, rollouts: list[TrainRollout]) -> None:
        pool = self.pool
        assert pool is not None, "reference pool not set — Algorithm.setup() must run before scoring"
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def score_rollout(client, rollout: TrainRollout) -> None:
            prefix_ids = self._ref_prefix_ids(rollout)
            assert len(rollout.samples) == 1  # single-step trajectory → one sample
            sample = rollout.samples[0]
            async with semaphore:
                full_logprobs = await compute_prefill_logprobs(
                    client, pool.model_name, prefix_ids + list(sample.completion_ids)
                )
            completion_logprobs = full_logprobs[-len(sample.completion_ids) :]
            sample.ref_logprobs = [0.0] * len(sample.prompt_ids) + completion_logprobs

        await asyncio.gather(
            *[score_rollout(client, rollout) for client, rollout in zip(cycle(pool.train_clients), rollouts)]
        )


def setup_advantage_strategy(config: AdvantageConfig, tokenizer: PreTrainedTokenizer) -> AdvantageStrategy:
    if isinstance(config, GroupNormAdvantageConfig):
        return GroupNormAdvantage(config)
    if isinstance(config, RewardAdvantageConfig):
        return RewardAdvantage()
    if isinstance(config, CustomAdvantageConfig):
        return CustomAdvantage(config)
    if isinstance(config, SupervisedAdvantageConfig):
        return SupervisedAdvantage()
    if isinstance(config, RefKLAdvantageConfig):
        return RefKLAdvantage(config)
    assert isinstance(config, DemoRefKLAdvantageConfig)
    return DemoRefKLAdvantage(config, tokenizer)
