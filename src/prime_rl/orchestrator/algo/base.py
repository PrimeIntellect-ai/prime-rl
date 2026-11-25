"""Episode-native algorithm hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import verifiers.v1 as vf

from prime_rl.configs.algorithm import AlgoConfig, FrozenModelConfig
from prime_rl.orchestrator.clients import InferenceClient, connect_frozen_client

if TYPE_CHECKING:
    from prime_rl.env.heads import EpisodeHead, TraceHead


def has_sampled_tokens(trace: vf.Trace | TraceHead) -> bool:
    """Whether any node of the trace has trainable (model-sampled) tokens."""
    if isinstance(trace, vf.Trace):
        return any(np.any(node.mask) for node in trace.nodes)
    return trace.has_sampled_tokens


def iter_trainable_traces(episodes: list[vf.Episode] | list[EpisodeHead]):
    """Yield clean, kept, trainable traces that contain sampled tokens: of full episodes in
    the env worker, of their heads on the orchestrator."""
    for episode in episodes:
        for trace in episode.traces:
            if (
                not episode.ok
                or trace.has_error
                or trace.is_dropped
                or not trace.agent.trainable
                or trace.num_turns == 0
            ):
                continue
            if has_sampled_tokens(trace):
                yield episode, trace


class Algorithm:
    """An algorithm's training signal, computed where each part's data lives.

    - :meth:`score_episode`: rollout-local work on the trace graph (loss weights), in the env
      worker that ran the rollout, before it keeps the rollout's training samples.
    - :meth:`score_sample`: model-calling work on a kept sample (reference logprobs), in the
      replica's sample server, with :meth:`sample_context` gathered in the env worker.
    - :meth:`score_group`: cohort-relative credit on the orchestrator, one advantage per trace.
    """

    def __init__(self, config: AlgoConfig, clients: InferenceClient):
        self.action_loss_type = config.action_loss_type
        self.clients = clients
        self.connected: InferenceClient | None = None

    async def setup(self) -> None:
        """Connect resources owned by the algorithm, on the orchestrator."""

    def bind(self, client: vf.ClientConfig, model: str) -> None:
        """Connect what :meth:`score_sample` needs in an sample server, given the client and
        model the replica's rollouts sample from."""

    def sample_context(self, episode: vf.Episode, trace: vf.Trace) -> dict:
        """What :meth:`score_sample` needs of a trace beyond its samples' tokens."""
        return {}

    async def score_sample(self, token_ids: np.ndarray, sampled: np.ndarray, context: dict) -> np.ndarray | None:
        """Reference logprobs of a kept sample's tokens (0.0 on tokens the policy did not
        sample), or None for an algorithm that scores no samples."""
        return None

    @property
    def scores_samples(self) -> bool:
        return type(self).score_sample is not Algorithm.score_sample

    async def connect(self, reference: FrozenModelConfig) -> InferenceClient:
        """Connect and track the frozen model pool owned by this algorithm."""
        self.connected = await connect_frozen_client(reference)
        return self.connected

    async def score_episode(self, episode: vf.Episode) -> None:
        """Assign rollout-local, per-token annotations (loss weights) to the nodes of one
        finalized episode, in the env worker that ran it."""

    async def score_group(self, episodes: list[EpisodeHead]) -> None:
        """Credit a finalized cohort: one advantage (``assign_advantage``) and optionally one rl
        weight (``assign_rl_weight``) per trace."""

    def reconfigure(self, config: AlgoConfig) -> None:
        """Adopt a live config update, in which only fields marked LIVE can differ."""

    async def finalize_episode(self, episode: vf.Episode) -> None:
        """Run rollout-local scoring when the episode has trainable traces."""
        if any(True for _ in iter_trainable_traces([episode])):
            await self.score_episode(episode)
