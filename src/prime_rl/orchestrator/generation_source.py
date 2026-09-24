"""Resolve the model source used to generate an env's train episodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import FrozenModelConfig, SamplingConfig
from prime_rl.orchestrator.algo import connect_frozen_client

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


class GenerationSource:
    """The policy or frozen endpoint that generates an env's training episodes."""

    def __init__(self, config: SamplingConfig, clients: InferenceClient):
        assert config.source is not None, "sampling.source must be resolved by config validation"
        self.config = config
        self.clients: InferenceClient = clients
        self.connected: InferenceClient | None = None  # frozen clients connected in setup(); closed at shutdown

    async def setup(self) -> None:
        """Connect clients to a frozen generation source and wait for
        readiness. Must run before dispatching."""
        if isinstance(self.config.source, FrozenModelConfig):
            self.clients = await connect_frozen_client(self.config.source)
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
            args.pop("top_logprobs", None)
        return args
