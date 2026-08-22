"""Resolve the model source used to generate an env's train episodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import FrozenModelConfig, SamplingConfig
from prime_rl.orchestrator.algo import connect_frozen_client

if TYPE_CHECKING:
    from renderers import RendererConfig

    from prime_rl.orchestrator.clients import InferenceClient


class GenerationSource:
    """One env's generation model and inference clients.

    ``clients`` is what train rollouts are generated from: the policy clients,
    swapped for a connected frozen endpoint in :meth:`setup` when the source is
    an inline frozen model. A frozen source *generates* rollouts, so it uses
    the renderer (token-in/out) client (built from ``renderer_config``) — the
    rollout must carry tokens for training."""

    def __init__(self, config: SamplingConfig, clients: InferenceClient, renderer_config: RendererConfig | None = None):
        assert config.source is not None, "sampling.source must be resolved by config validation"
        self.config = config
        self.clients: InferenceClient = clients
        self.renderer_config = renderer_config
        self.connected: InferenceClient | None = None  # frozen clients connected in setup(); closed at shutdown

    async def setup(self) -> None:
        """Connect clients to a frozen generation source and wait for
        readiness. Must run before dispatching."""
        if isinstance(self.config.source, FrozenModelConfig):
            self.clients = await connect_frozen_client(self.config.source, renderer_config=self.renderer_config)
            self.connected = self.clients

    @property
    def uses_live_policy(self) -> bool:
        return self.config.source == "policy"

    def sampling_args(self, args: dict) -> dict:
        """Source-specific sampling-arg overrides. Sampling logprobs are only
        needed for importance ratios on policy-sampled tokens — frozen
        endpoints may reject the knob."""
        if not self.uses_live_policy:
            args.pop("logprobs", None)
        return args
