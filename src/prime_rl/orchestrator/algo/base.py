"""Episode-native algorithm hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import verifiers.v1 as vf

from prime_rl.configs.algorithm import ActionLossType, AlgoConfig, FrozenModelConfig
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from renderers import RendererConfig

    from prime_rl.utils.client import InferencePool


async def connect_frozen_pool(
    config: FrozenModelConfig, *, renderer_config: RendererConfig | None = None
) -> InferencePool:
    """Connect to an externally hosted frozen model and wait for it."""
    from prime_rl.utils.client import InferencePool

    get_logger().info(f"Initializing frozen model pool (model={config.name}, base_url={config.base_url})")
    if renderer_config is not None:
        pool = InferencePool(
            config, model_name=config.name, train_client_type="renderer", renderer_config=renderer_config
        )
    else:
        pool = InferencePool(config, model_name=config.name)
    await pool.wait_for_ready(config.name)
    return pool


def iter_trainable_traces(episodes: list[vf.Episode]):
    """Yield clean trainable traces that contain sampled tokens."""
    for episode in episodes:
        for trace in episode.traces:
            if trace.has_error or not trace.agent.trainable:
                continue
            if any(any(node.mask) for node in trace.nodes):
                yield episode, trace


class Algorithm:
    """Assign training annotations directly to verifier episodes.

    Override :meth:`score_episode` for rollout-local work and
    :meth:`score_group` for cohort-relative work. The train sink compiles the
    annotated traces into transport samples only after admission.
    """

    action_loss_type: ClassVar[ActionLossType] = "rl"

    def __init__(self, config: AlgoConfig, policy_pool: InferencePool):
        self.policy_pool = policy_pool
        self.connected_pools: list[InferencePool] = []

    async def setup(self) -> None:
        """Connect resources owned by the algorithm."""

    async def connect(self, reference: FrozenModelConfig) -> InferencePool:
        """Connect and track a frozen model pool owned by this algorithm."""
        pool = await connect_frozen_pool(reference)
        self.connected_pools.append(pool)
        return pool

    async def score_episode(self, episode: vf.Episode) -> None:
        """Assign rollout-local annotations to one finalized episode."""

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        """Assign group-relative annotations to a finalized cohort."""

    async def finalize_episode(self, episode: vf.Episode) -> None:
        """Run rollout-local scoring when the episode has trainable traces."""
        if any(True for _ in iter_trainable_traces([episode])):
            await self.score_episode(episode)

    async def finalize_group(self, episodes: list[vf.Episode]) -> None:
        """Run group-relative scoring over the native episodes."""
        await self.score_group(episodes)
