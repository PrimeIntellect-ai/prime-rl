"""Resolve the model source used to generate an env's train episodes."""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import FrozenModelConfig, SamplingConfig
from prime_rl.orchestrator.algo import connect_frozen_client
from prime_rl.orchestrator.clients import setup_admin_clients

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

    from prime_rl.orchestrator.clients import InferenceClient


class GenerationSource:
    """The policy or frozen endpoint that generates an env's training episodes."""

    def __init__(self, config: SamplingConfig, clients: InferenceClient):
        assert config.source is not None, "sampling.source must be resolved by config validation"
        self.config = config
        self.clients: InferenceClient = clients
        self.connected: InferenceClient | None = None  # frozen clients connected in setup(); closed at shutdown

    async def setup(self, tokenizer: PreTrainedTokenizer) -> None:
        """Connect a frozen source and verify its token IDs match the policy vocabulary before dispatch."""
        if isinstance(self.config.source, FrozenModelConfig):
            clients = await connect_frozen_client(self.config.source)
            try:
                async with AsyncExitStack() as stack:
                    admin_clients = [
                        await stack.enter_async_context(client) for client in setup_admin_clients(self.config.source)
                    ]
                    # Frozen samples train their native IDs directly, so every ID must retain its meaning.
                    policy_vocab = tokenizer.get_vocab()
                    for client in admin_clients:
                        response = await client.get("/v1/tokenizer", timeout=self.config.source.wait_for_ready_timeout)
                        if response.status_code != 200:
                            raise ValueError(
                                f"Frozen generation requires GET /v1/tokenizer; {client.base_url} returned "
                                f"HTTP {response.status_code}. If the router does not forward this endpoint, "
                                "set admin_base_url to the inference engine URLs."
                            )
                        vocab = response.json()
                        if (
                            not isinstance(vocab, dict)
                            or not vocab
                            or any(type(index) is not int for index in vocab.values())
                        ):
                            raise ValueError(
                                f"Frozen generation endpoint {client.base_url} returned an invalid vocabulary"
                            )
                        if vocab != policy_vocab:
                            raise ValueError(
                                f"Frozen generation endpoint {client.base_url} must use the same token-to-ID mapping "
                                "as the policy tokenizer; training across different tokenizers is not supported."
                            )
            except BaseException:
                await clients.aclose()
                raise
            self.clients = self.connected = clients

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
