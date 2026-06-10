"""The per-env algorithm runtime classes.

Each named class below *is* one training algorithm: it owns the algorithm's
methods directly — ``assign`` (group-time credit) and ``score`` (ship-time
reference scoring) — and declares what it needs (``action_loss_type``, a
``model_role`` like "teacher"). Reading a class top to bottom reads the
algorithm; writing your own is subclassing :class:`Algorithm` and overriding
the same methods. Shared math (group normalization, prefill alignment) lives
as plain functions in ``advantage.py``; duplication of orchestration between
similar algorithms (e.g. OPD and OPSD) is accepted so each class stays
self-contained.

The pipeline (dispatcher, train sink, orchestrator) calls the base-class hooks
and reads its properties; it never branches on algorithm config fields or
model roles — liveness of a reference is the only runtime distinction.
prime-rl hosts exactly one model — the trainable policy, whose pool is passed
in; every frozen model reference is an external endpoint the algorithm
*connects to* (never launches) in :meth:`Algorithm.setup`.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from itertools import cycle
from typing import TYPE_CHECKING, ClassVar

from prime_rl.configs.algorithm import (
    ActionLossType,
    AlgorithmConfig,
    CustomAdvantageConfig,
    DemoRefKLAdvantageConfig,
    FrozenModelConfig,
    GroupNormAdvantageConfig,
    LengthPenaltyConfig,
    RefKLAdvantageConfig,
)
from prime_rl.orchestrator.algo.advantage import (
    AdvantageInputs,
    AdvantageOutputs,
    assign_advantages,
    default_advantage_fn,
)
from prime_rl.orchestrator.algo.routing import spread_token_advantages, stamp_loss_routing
from prime_rl.orchestrator.utils import compute_prefill_logprobs
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import import_object

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

    from prime_rl.orchestrator.envs import TrainEnvs
    from prime_rl.orchestrator.types import TrainRollout
    from prime_rl.transport import TrainingSample
    from prime_rl.utils.client import InferencePool


async def connect_frozen_pool(config: FrozenModelConfig) -> InferencePool:
    """Connect a client pool to an inline frozen model and wait for it to be
    ready. The endpoint is externally hosted — prime-rl connects and waits,
    never launches."""
    from prime_rl.utils.client import setup_inference_pool

    get_logger().info(f"Initializing frozen model pool (model={config.name}, base_url={', '.join(config.base_url)})")
    pool = await setup_inference_pool(config, model_name=config.name)
    await pool.wait_for_ready(config.name)
    return pool


def _assign_group_norm(rollouts: list[TrainRollout], length_penalty: LengthPenaltyConfig | None) -> None:
    assign_advantages(rollouts, lambda inputs: default_advantage_fn(inputs, length_penalty=length_penalty))


class Algorithm:
    """Base class for one env's training algorithm — the sole interpreter of
    ``AlgorithmConfig`` in the orchestrator.

    Subclass and override the two execution points of the training signal:

    - :meth:`assign` — group finalization, cheap and synchronous; set
      rollout-level scalar (and optionally per-token) advantages. The default
      assigns nothing (rollouts keep ``advantage=None``, so advantage-based
      filters skip them).
    - :meth:`score` — batch-ship time, async; attach per-token reference data
      by querying ``self.reference_pool``. The default scores nothing.

    Class-level declarations say what the algorithm needs: which loss
    component its action tokens feed (``action_loss_type``) and what it calls
    its reference model, if it has one (``model_role``, e.g. "teacher").
    Holds the policy pool (built once by the orchestrator) and connects client
    pools to any inline frozen model references in :meth:`setup`."""

    action_loss_type: ClassVar[ActionLossType] = "rl"
    model_role: ClassVar[str | None] = None

    def __init__(self, config: AlgorithmConfig, policy_pool: InferencePool, tokenizer: PreTrainedTokenizer):
        assert config.sampling.source is not None
        self.config = config
        self.tokenizer = tokenizer
        self.policy_pool = policy_pool
        self.sampling_pool: InferencePool = policy_pool  # frozen sources swap this in setup()
        self.reference_pool: InferencePool | None = None  # resolved in setup() when the algorithm declares a model
        self.connected_pools: list[InferencePool] = []  # client pools connected in setup(); closed at shutdown
        self.loss = config.loss

    async def setup(self) -> None:
        """Connect client pools to the algorithm's frozen model references and
        wait for readiness. Must run before dispatching or scoring."""
        source = self.config.sampling.source
        if isinstance(source, FrozenModelConfig):
            self.sampling_pool = await connect_frozen_pool(source)
            self.connected_pools.append(self.sampling_pool)
        reference = getattr(self.config.advantage, "model", None)
        if reference is not None:
            if reference == "policy":
                self.reference_pool = self.policy_pool
            else:
                self.reference_pool = await connect_frozen_pool(reference)
                self.connected_pools.append(self.reference_pool)

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def samples_from_live_policy(self) -> bool:
        return self.config.sampling.source == "policy"

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

    def assign(self, rollouts: list[TrainRollout]) -> None:
        """Assign credit to one finalized group of rollouts."""

    async def score(self, rollouts: list[TrainRollout]) -> None:
        """Attach per-token reference data to a batch of rollouts at ship time."""

    def finalize_group(self, rollouts: list[TrainRollout]) -> None:
        """Score one finalized group: assign credit, then stamp each sample's
        wire fields (advantage + loss routing)."""
        self.assign(rollouts)
        for rollout in rollouts:
            if rollout.token_advantages is not None:
                spread_token_advantages(rollout)
            for sample in rollout.samples:
                # Algorithms without scalars leave ``rollout.advantage=None``
                # (advantage-based filters skip it); the wire ships a
                # neutral 0.0.
                sample.advantage = rollout.advantage if rollout.advantage is not None else 0.0
                sample.reward = rollout.reward
                sample.env_name = rollout.env_name
                stamp_loss_routing(sample, self.action_loss_type, self.loss)

    async def score_batch(self, rollouts: list[TrainRollout]) -> None:
        """Run :meth:`score` over this env's rollouts. No-op for algorithms
        without reference scoring."""
        if not rollouts:
            return
        await self.score(rollouts)

    def _reference_pool(self) -> InferencePool:
        pool = self.reference_pool
        assert pool is not None, f"{self.model_role or 'reference'} pool not set — Algorithm.setup() must run first"
        return pool


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the group mean (optionally
    length-shaped); action tokens feed the ``rl`` loss."""

    def __init__(self, config: AlgorithmConfig, policy_pool: InferencePool, tokenizer: PreTrainedTokenizer):
        super().__init__(config, policy_pool, tokenizer)
        assert isinstance(config.advantage, GroupNormAdvantageConfig)
        self.length_penalty = config.advantage.length_penalty

    def assign(self, rollouts: list[TrainRollout]) -> None:
        _assign_group_norm(rollouts, self.length_penalty)


class OPDAlgorithm(Algorithm):
    """On-policy distillation. Needs a teacher: the frozen reference model the
    per-token reverse KL is computed against.

    The policy samples its own rollouts; at ship time each sample's full
    context is prefill-scored under the teacher (``ref_logprobs`` on the
    wire), and the trainer evaluates the KL against the live policy. Group
    scalars are still assigned: their sign steers the DPPO masking direction
    in the ``ref_kl`` loss, and the zero-advantage filter reads them."""

    action_loss_type = "ref_kl"
    model_role = "teacher"

    def __init__(self, config: AlgorithmConfig, policy_pool: InferencePool, tokenizer: PreTrainedTokenizer):
        super().__init__(config, policy_pool, tokenizer)
        assert isinstance(config.advantage, RefKLAdvantageConfig)
        self.length_penalty = config.advantage.length_penalty
        self.max_concurrent = config.advantage.max_concurrent

    def assign(self, rollouts: list[TrainRollout]) -> None:
        _assign_group_norm(rollouts, self.length_penalty)

    async def score(self, rollouts: list[TrainRollout]) -> None:
        pool = self._reference_pool()
        semaphore = asyncio.Semaphore(self.max_concurrent)
        samples = [sample for rollout in rollouts for sample in rollout.samples]

        async def score_sample(client, sample: TrainingSample) -> None:
            async with semaphore:
                token_ids = list(sample.prompt_ids) + list(sample.completion_ids)
                sample.ref_logprobs = await compute_prefill_logprobs(client, pool.model_name, token_ids)

        await asyncio.gather(
            *[score_sample(client, sample) for client, sample in zip(cycle(pool.train_clients), samples)]
        )


class OPSDAlgorithm(Algorithm):
    """On-policy self-distillation (SDFT). The teacher defaults to the policy
    itself, conditioned on an expert demonstration — no extra deployment.

    The scoring prefix is rebuilt from the rollout's first-turn prompt
    messages with the demonstration woven into the last user message; the
    returned completion logprobs are aligned back onto the sample (the
    sample's prompt positions are 0.0 and stay outside the loss mask). No
    scalar advantage is assigned."""

    action_loss_type = "ref_kl"
    model_role = "teacher"

    def __init__(self, config: AlgorithmConfig, policy_pool: InferencePool, tokenizer: PreTrainedTokenizer):
        super().__init__(config, policy_pool, tokenizer)
        assert isinstance(config.advantage, DemoRefKLAdvantageConfig)
        self.demo_key = config.advantage.demo_key
        self.template = config.advantage.template
        self.max_concurrent = config.advantage.max_concurrent

    def _ref_prefix_ids(self, rollout: TrainRollout) -> list[int]:
        trajectory = rollout.raw.get("trajectory") or []
        if len(trajectory) != 1:
            raise ValueError(
                f"demo_ref_kl supports single-step trajectories only; "
                f"env '{rollout.env_name}' produced {len(trajectory)} steps."
            )
        info = rollout.raw.get("info") or {}
        demonstration = info.get(self.demo_key) if isinstance(info, dict) else None
        if demonstration is None:
            demonstration = rollout.raw.get(self.demo_key)
        if demonstration is None:
            raise ValueError(
                f"demo_ref_kl requires '{self.demo_key}' in the example's info dict or as a "
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
        last_user["content"] = self.template.format(question=question, demonstration=demonstration)

        # ``return_dict=False`` pins the flat token-id list — newer
        # transformers default to returning a BatchEncoding.
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=False
        )

    async def score(self, rollouts: list[TrainRollout]) -> None:
        pool = self._reference_pool()
        semaphore = asyncio.Semaphore(self.max_concurrent)

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


class SFTDistillAlgorithm(Algorithm):
    """Hard distillation. Needs a teacher: the frozen model that generates the
    rollouts (``sampling.source``); the policy trains with CE on its tokens.

    The ``ce`` loss ignores scalars, but group-relative scalars are still
    assigned so reward-based filtering keeps working."""

    action_loss_type = "ce"

    def assign(self, rollouts: list[TrainRollout]) -> None:
        _assign_group_norm(rollouts, None)


class RewardAlgorithm(Algorithm):
    """REINFORCE-style: credit = raw reward, no group baseline; action tokens
    feed the ``rl`` loss."""

    def assign(self, rollouts: list[TrainRollout]) -> None:
        assign_advantages(rollouts, None)


class CustomAlgorithm(Algorithm):
    """User-supplied advantage function: one scalar per rollout, optionally
    with per-token advantages aligned to each rollout's completion tokens."""

    def __init__(self, config: AlgorithmConfig, policy_pool: InferencePool, tokenizer: PreTrainedTokenizer):
        super().__init__(config, policy_pool, tokenizer)
        assert isinstance(config.advantage, CustomAdvantageConfig)
        custom_fn = import_object(config.advantage.import_path)
        kwargs = config.advantage.kwargs

        def advantage_fn(inputs: AdvantageInputs) -> AdvantageOutputs:
            return custom_fn(inputs, **kwargs)

        self.advantage_fn = advantage_fn

    def assign(self, rollouts: list[TrainRollout]) -> None:
        assign_advantages(rollouts, self.advantage_fn)


# Runtime dispatch is keyed on the advantage type — the axis along which
# behavior actually differs. Preset names are vetted parameterizations of
# these classes (e.g. ``echo`` builds GRPOAlgorithm with observation routing).
ALGORITHM_CLASSES: dict[str, type[Algorithm]] = {
    "group_norm": GRPOAlgorithm,
    "ref_kl": OPDAlgorithm,
    "demo_ref_kl": OPSDAlgorithm,
    "supervised": SFTDistillAlgorithm,
    "reward": RewardAlgorithm,
    "custom": CustomAlgorithm,
}


def build_algorithm(config: AlgorithmConfig, policy_pool: InferencePool, tokenizer: PreTrainedTokenizer) -> Algorithm:
    cls = ALGORITHM_CLASSES[config.advantage.type]
    assert cls.action_loss_type == config.advantage.action_loss_type  # config and runtime declare in two places
    return cls(config, policy_pool, tokenizer)


async def score_train_batch(train_envs: TrainEnvs, rollouts: list[TrainRollout]) -> None:
    """Run each env's ``score_batch`` over its unfiltered rollouts,
    concurrently across envs. Per-env concurrency is bounded by the
    algorithm's own config; envs without reference scoring return
    immediately."""
    by_env: dict[str, list[TrainRollout]] = defaultdict(list)
    for rollout in rollouts:
        if not rollout.is_filtered:
            by_env[rollout.env_name].append(rollout)
    await asyncio.gather(
        *(train_envs.get(env_name).algorithm.score_batch(env_rollouts) for env_name, env_rollouts in by_env.items())
    )
