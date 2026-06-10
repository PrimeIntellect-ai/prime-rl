"""The per-env algorithm runtime object.

:class:`Algorithm` is the only orchestrator component that interprets
``AlgorithmConfig``. The pipeline (dispatcher, train sink, orchestrator) calls
its hooks and reads its properties; it never branches on algorithm config
fields. prime-rl hosts exactly one model — the trainable policy, whose pool is
passed in; every frozen model reference is an external endpoint the algorithm
*connects to* (never launches) in :meth:`Algorithm.setup`.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import AlgorithmConfig, FrozenModelConfig
from prime_rl.orchestrator.algo.routing import spread_token_advantages, stamp_loss_routing
from prime_rl.orchestrator.algo.strategies import setup_advantage_strategy
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

    from prime_rl.orchestrator.envs import TrainEnvs
    from prime_rl.orchestrator.types import TrainRollout
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


class Algorithm:
    """Runtime strategy object for one env — the sole interpreter of
    ``AlgorithmConfig`` in the orchestrator.

    Holds the policy pool (built once by the orchestrator) and connects client
    pools to any inline frozen model references in :meth:`setup`."""

    def __init__(self, config: AlgorithmConfig, policy_pool: InferencePool, tokenizer: PreTrainedTokenizer):
        assert config.sampling.source is not None
        self.config = config
        self.policy_pool = policy_pool
        self.sampling_pool: InferencePool = policy_pool  # frozen sources swap this in setup()
        self.connected_pools: list[InferencePool] = []  # client pools connected in setup(); closed at shutdown
        self.loss = config.loss
        self.action_loss_type = config.advantage.action_loss_type
        self.advantage = setup_advantage_strategy(config.advantage, tokenizer)

    async def setup(self) -> None:
        """Connect client pools to the algorithm's frozen model references and
        wait for readiness. Must run before dispatching or scoring."""
        source = self.config.sampling.source
        if isinstance(source, FrozenModelConfig):
            self.sampling_pool = await connect_frozen_pool(source)
            self.connected_pools.append(self.sampling_pool)
        reference = getattr(self.config.advantage, "model", None)
        if reference is not None:
            assert hasattr(self.advantage, "pool")
            if reference == "policy":
                self.advantage.pool = self.policy_pool
            else:
                self.advantage.pool = await connect_frozen_pool(reference)
                self.connected_pools.append(self.advantage.pool)

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

    def finalize_group(self, rollouts: list[TrainRollout]) -> None:
        """Score one finalized group: assign scalar advantages, then stamp
        each sample's wire fields (advantage + loss routing)."""
        self.advantage.assign(rollouts)
        for rollout in rollouts:
            if rollout.token_advantages is not None:
                spread_token_advantages(rollout)
            for sample in rollout.samples:
                # Strategies without scalars leave ``rollout.advantage=None``
                # (advantage-based filters skip it); the wire ships a
                # neutral 0.0.
                sample.advantage = rollout.advantage if rollout.advantage is not None else 0.0
                sample.reward = rollout.reward
                sample.env_name = rollout.env_name
                stamp_loss_routing(sample, self.action_loss_type, self.loss)

    async def score_batch(self, rollouts: list[TrainRollout]) -> None:
        """Run the advantage strategy's ship-time scoring over this env's
        rollouts. No-op for strategies without reference scoring."""
        if not rollouts:
            return
        await self.advantage.score(rollouts)


async def score_train_batch(train_envs: TrainEnvs, rollouts: list[TrainRollout]) -> None:
    """Run each env's ``score_batch`` over its unfiltered rollouts,
    concurrently across envs. Per-env concurrency is bounded by the strategy's
    own config; envs without reference scoring return immediately."""
    by_env: dict[str, list[TrainRollout]] = defaultdict(list)
    for rollout in rollouts:
        if not rollout.is_filtered:
            by_env[rollout.env_name].append(rollout)
    await asyncio.gather(
        *(train_envs.get(env_name).algorithm.score_batch(env_rollouts) for env_name, env_rollouts in by_env.items())
    )
