"""Orchestrator-side algorithm runtime.

The config side (``prime_rl.configs.algorithm``) defines *what* an algorithm
is — a preset of sampling, scoring (advantage + token scorer), and loss
routing. This module turns that declaration into runtime objects:

- :class:`ModelRegistry` — named inference pools. ``"policy"`` is the live
  policy; every other entry is a frozen hosted model from
  ``[orchestrator.models]``. Liveness (cache salting, sampling logprobs,
  off-policy aging) is a property of the entry, not of any model role.
- :class:`Algorithm` — one strategy object per env, the only orchestrator
  component that interprets ``AlgorithmConfig``. The pipeline (dispatcher,
  train sink, orchestrator) calls its hooks and reads its properties; it
  never branches on algorithm config fields.
- **Token scorers** — async per-sample scoring that attaches per-token data
  by querying a reference model from the registry (bounded concurrency).
  Runs at batch-ship time via :func:`score_train_batch`.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from itertools import cycle
from typing import TYPE_CHECKING, Protocol

from prime_rl.configs.algorithm import (
    POLICY_MODEL,
    AlgorithmConfig,
    DemoLogprobsScorerConfig,
    LogprobsScorerConfig,
    LossRoutingConfig,
    TokenScorerConfig,
)
from prime_rl.orchestrator.advantage import assign_advantages, setup_advantage_fn
from prime_rl.orchestrator.utils import compute_prefill_logprobs
from prime_rl.transport import TrainingSample
from prime_rl.transport.types import LOSS_CORE_CE, LOSS_CORE_REF_KL, LOSS_CORE_RL

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

    from prime_rl.orchestrator.envs import TrainEnvs
    from prime_rl.orchestrator.types import TrainRollout
    from prime_rl.utils.client import InferencePool

ACTION_LOSS_CORES = {"rl": LOSS_CORE_RL, "ce": LOSS_CORE_CE, "ref_kl": LOSS_CORE_REF_KL}


class ModelRegistry:
    """Named inference pools, registered during orchestrator setup.

    Algorithms hold names and resolve pools lazily, so they can be constructed
    before the pools are ready."""

    def __init__(self) -> None:
        self.pools: dict[str, InferencePool] = {}

    def register(self, name: str, pool: InferencePool) -> None:
        self.pools[name] = pool

    def get(self, name: str) -> InferencePool:
        return self.pools[name]

    def is_live(self, name: str) -> bool:
        """Live entries are weight-updated as training advances: their prefix
        caches must be salted per policy version, their rollouts age
        off-policy, and importance ratios need their sampling logprobs.
        Frozen entries need none of that."""
        return name == POLICY_MODEL


def stamp_loss_routing(sample: TrainingSample, loss: LossRoutingConfig) -> None:
    """Stamp the env's loss routing onto one sample's wire fields.

    Action tokens (the trainable completion tokens) get the configured action
    core. When the algorithm trains on observations, env-provided tokens
    (tagged by ``interleave_rollout`` in ``completion_obs_mask``) flip from
    masked-out to trainable on the CE core with ``observation_weight``.
    ``completion_obs_mask`` is orchestrator-internal and cleared here so it
    never ships.
    """
    action_core = ACTION_LOSS_CORES[loss.action]
    sample.loss_core = action_core
    obs_mask = sample.completion_obs_mask
    sample.completion_obs_mask = None
    if loss.observation == "none" or obs_mask is None or not any(obs_mask):
        return

    prompt_len = len(sample.prompt_ids)
    seq_len = prompt_len + len(sample.completion_ids)
    cores = [action_core] * seq_len
    weights = [1.0] * seq_len
    completion_mask = list(sample.completion_mask)
    for i, is_obs in enumerate(obs_mask):
        if is_obs:
            cores[prompt_len + i] = LOSS_CORE_CE
            weights[prompt_len + i] = loss.observation_weight
            completion_mask[i] = True
    sample.completion_mask = completion_mask
    sample.token_loss_cores = cores
    sample.token_loss_weights = weights


class TokenScorer(Protocol):
    async def score(self, rollouts: list[TrainRollout]) -> None: ...


class LogprobsScorer:
    """Fill ``TrainingSample.ref_logprobs`` by scoring each sample's own
    context under the reference model (on-policy distillation)."""

    def __init__(self, config: LogprobsScorerConfig, registry: ModelRegistry):
        assert config.model is not None
        self.config = config
        self.registry = registry

    async def score(self, rollouts: list[TrainRollout]) -> None:
        pool = self.registry.get(self.config.model)
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        samples = [sample for rollout in rollouts for sample in rollout.samples]

        async def score_sample(client, sample: TrainingSample) -> None:
            async with semaphore:
                token_ids = list(sample.prompt_ids) + list(sample.completion_ids)
                sample.ref_logprobs = await compute_prefill_logprobs(client, pool.model_name, token_ids)

        await asyncio.gather(
            *[score_sample(client, sample) for client, sample in zip(cycle(pool.train_clients), samples)]
        )


class DemoLogprobsScorer:
    """Fill ``TrainingSample.ref_logprobs`` by scoring each sample's completion
    under the reference model conditioned on an expert demonstration (SDFT).

    The scoring prefix is rebuilt from the rollout's first-turn prompt
    messages with the demonstration woven into the last user message; the
    returned completion logprobs are aligned back onto the sample (the
    sample's prompt positions are 0.0 and stay outside the loss mask).
    """

    def __init__(self, config: DemoLogprobsScorerConfig, registry: ModelRegistry, tokenizer: PreTrainedTokenizer):
        assert config.model is not None
        self.config = config
        self.registry = registry
        self.tokenizer = tokenizer

    def _ref_prefix_ids(self, rollout: TrainRollout) -> list[int]:
        trajectory = rollout.raw.get("trajectory") or []
        if len(trajectory) != 1:
            raise ValueError(
                f"demo_logprobs supports single-step trajectories only; "
                f"env '{rollout.env_name}' produced {len(trajectory)} steps."
            )
        info = rollout.raw.get("info") or {}
        demonstration = info.get(self.config.demo_key) if isinstance(info, dict) else None
        if demonstration is None:
            demonstration = rollout.raw.get(self.config.demo_key)
        if demonstration is None:
            raise ValueError(
                f"demo_logprobs requires '{self.config.demo_key}' in the example's info dict or as a "
                f"top-level rollout field (env '{rollout.env_name}', example {rollout.example_id})."
            )

        messages = [dict(m) for m in trajectory[0]["prompt"]]
        user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if not user_indices:
            raise ValueError(f"demo_logprobs found no user message to condition (env '{rollout.env_name}').")
        last_user = messages[user_indices[-1]]
        question = last_user.get("content")
        if not isinstance(question, str):
            raise ValueError("demo_logprobs supports text-only prompts (user content must be a string).")
        last_user["content"] = self.config.template.format(question=question, demonstration=demonstration)

        # ``return_dict=False`` pins the flat token-id list — newer
        # transformers default to returning a BatchEncoding.
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=False
        )

    async def score(self, rollouts: list[TrainRollout]) -> None:
        pool = self.registry.get(self.config.model)
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


def setup_token_scorer(
    config: TokenScorerConfig | None, registry: ModelRegistry, tokenizer: PreTrainedTokenizer
) -> TokenScorer | None:
    if config is None:
        return None
    if isinstance(config, LogprobsScorerConfig):
        return LogprobsScorer(config, registry)
    return DemoLogprobsScorer(config, registry, tokenizer)


class Algorithm:
    """Runtime strategy object for one env — the sole interpreter of
    ``AlgorithmConfig`` in the orchestrator."""

    def __init__(self, config: AlgorithmConfig, registry: ModelRegistry, tokenizer: PreTrainedTokenizer):
        assert config.sampling is not None and config.sampling.source is not None
        assert config.advantage is not None and config.loss is not None
        self.config = config
        self.registry = registry
        self.sampling_source: str = config.sampling.source
        self.loss = config.loss
        self.advantage_fn = setup_advantage_fn(config.advantage)
        self.token_scorer = setup_token_scorer(config.token_scorer, registry, tokenizer)

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def samples_from_live_policy(self) -> bool:
        return self.registry.is_live(self.sampling_source)

    @property
    def tag_observation_tokens(self) -> bool:
        """``interleave_rollout`` marks env-provided tokens when the loss
        routing trains on them."""
        return self.loss.observation != "none"

    def sampling_args(self, args: dict) -> dict:
        """Algorithm-specific sampling-arg overrides. Sampling logprobs are
        only needed for importance ratios on policy-sampled tokens — frozen
        endpoints may reject the knob."""
        if not self.samples_from_live_policy:
            args.pop("logprobs", None)
        return args

    def finalize_group(self, rollouts: list[TrainRollout]) -> None:
        """Score one finalized group: assign group-level advantages, then
        stamp each sample's wire fields (advantage + loss routing)."""
        assign_advantages(rollouts, self.advantage_fn)
        for rollout in rollouts:
            for sample in rollout.samples:
                # ``advantage=None`` (NoAdvantageConfig) ships as neutral 0.0;
                # the rollout keeps None so advantage-based filters skip it.
                sample.advantage = rollout.advantage if rollout.advantage is not None else 0.0
                sample.reward = rollout.reward
                sample.env_name = rollout.env_name
                stamp_loss_routing(sample, self.loss)

    async def score_batch(self, rollouts: list[TrainRollout]) -> None:
        """Run the token scorer over this env's rollouts at batch-ship time.
        No-op without a scorer."""
        if self.token_scorer is None or not rollouts:
            return
        await self.token_scorer.score(rollouts)


async def score_train_batch(train_envs: TrainEnvs, rollouts: list[TrainRollout]) -> None:
    """Run each env's ``score_batch`` over its unfiltered rollouts,
    concurrently across envs. Per-env concurrency is bounded by the scorer's
    own config; envs without a scorer return immediately."""
    by_env: dict[str, list[TrainRollout]] = defaultdict(list)
    for rollout in rollouts:
        if not rollout.is_filtered:
            by_env[rollout.env_name].append(rollout)
    await asyncio.gather(
        *(train_envs.get(env_name).algorithm.score_batch(env_rollouts) for env_name, env_rollouts in by_env.items())
    )
