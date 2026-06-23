"""How an env's train rollouts are produced — the sample strategy.

The algorithm (``algo/``) consumes finalized rollouts and compiles them into
per-token loss-component weights; the sampler owns where those rollouts come
from. Today that is one question — which model generates them — and its
consequences (sampling logprobs, prefix-cache salting, and off-policy
staleness are all liveness questions about the source). Future sampling
strategies (replay buffers, branching) extend here, not the algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import FrozenModelConfig, SamplingConfig
from prime_rl.orchestrator.algo import connect_frozen_pool

if TYPE_CHECKING:
    from renderers import RendererConfig

    from prime_rl.utils.client import InferencePool


class Sampler:
    """One env's rollout source.

    ``pool`` is the pool train rollouts are generated from: the policy pool,
    swapped for a connected frozen pool in :meth:`setup` when the source is an
    inline frozen model. A frozen source *generates* rollouts, so its pool uses
    the renderer (token-in/out) client (built from ``renderer_config``) — the
    rollout must carry tokens for training."""

    def __init__(
        self, config: SamplingConfig, policy_pool: InferencePool, renderer_config: RendererConfig | None = None
    ):
        assert config.source is not None, "sampling.source must be resolved by config validation"
        self.config = config
        self.pool: InferencePool = policy_pool
        self.renderer_config = renderer_config
        self.connected_pools: list[InferencePool] = []  # client pools connected in setup(); closed at shutdown

    async def setup(self) -> None:
        """Connect a client pool to a frozen sampling source and wait for
        readiness. Must run before dispatching."""
        if isinstance(self.config.source, FrozenModelConfig):
            self.pool = await connect_frozen_pool(self.config.source, renderer_config=self.renderer_config)
            self.connected_pools.append(self.pool)

    @property
    def samples_from_live_policy(self) -> bool:
        return self.config.source == "policy"

    def sampling_args(self, args: dict) -> dict:
        """Source-specific sampling-arg overrides. Sampling logprobs are only
        needed for importance ratios on policy-sampled tokens — frozen
        endpoints may reject the knob."""
        if not self.samples_from_live_policy:
            args.pop("logprobs", None)
        return args
