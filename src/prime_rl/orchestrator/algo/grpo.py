from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, Callable

import verifiers.v1 as vf

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages
from prime_rl.orchestrator.trajectories import iter_trainable_branches
from prime_rl.utils.utils import import_object

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the group mean (optionally
    length-shaped, mean-normalized, or baselined per agent); action tokens feed
    the ``rl`` loss. With ``echo``, env-provided observation tokens also feed
    the ``ce`` loss."""

    def __init__(self, config: GRPOAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.length_penalty = config.length_penalty
        self.normalize_by_mean = config.normalize_by_mean
        self.episode_agents = set(config.episode_agents) if config.episode_agents is not None else None
        self.role_weights: dict[str, float] = {}
        self.filter_fn: Callable[..., list[list[bool]]] | None = None
        if config.echo is not None:
            self.role_weights = {
                role: role_config.alpha
                for role in ("system", "user", "assistant", "tool")
                if (role_config := getattr(config.echo.roles, role)) is not None
            }
            if config.echo.filter is not None:
                self.filter_fn = partial(import_object(config.echo.filter.import_path), **config.echo.filter.kwargs)

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        if self.episode_agents is not None:
            self._score_per_agent(episodes)
            return

        import torch  # only the trainer-side extras ship torch; an eval process never scores a group

        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        rewards = torch.tensor([trace.reward for trace in traces], dtype=torch.float32)
        length_penalty = self.length_penalty
        if length_penalty is None:
            advantages = rewards - rewards.mean()
        else:
            output = torch.tensor([trace.num_output_tokens for trace in traces], dtype=rewards.dtype)
            total = torch.tensor([trace.num_total_tokens for trace in traces], dtype=rewards.dtype)
            turns = torch.tensor([trace.num_turns for trace in traces], dtype=rewards.dtype)
            input = total - output
            penalty_frac = (
                length_penalty.num_output_tokens_weight * (output / output.max().clamp(min=1))
                + length_penalty.num_input_tokens_weight * (input / input.max().clamp(min=1))
                + length_penalty.num_turns_weight * (turns / turns.max().clamp(min=1))
            )
            penalty = rewards.mean() * penalty_frac
            shaped_rewards = rewards - penalty
            advantages = shaped_rewards - shaped_rewards.mean()
        if self.normalize_by_mean:
            mean = rewards.mean()
            advantages = torch.zeros_like(rewards) if mean <= 0 else advantages / mean
        for trace, advantage in zip(traces, advantages.tolist(), strict=True):
            assign_advantages(trace, advantage)

    def _score_per_agent(self, episodes: list[vf.Episode]) -> None:
        """Baseline each trace against its own agent's traces: within the
        episode for ``episode_agents``, across the group for the others."""
        peers: dict[tuple[str, str | None], list[vf.Trace]] = defaultdict(list)
        for episode, trace in iter_trainable_traces(episodes):
            episode_scoped = trace.agent.name in self.episode_agents
            key = (trace.agent.name, episode.id if episode_scoped else None)
            peers[key].append(trace)
        for members in peers.values():
            baseline = sum(trace.reward for trace in members) / len(members)
            for trace in members:
                assign_advantages(trace, trace.reward - baseline)

    async def score_episode(self, episode: vf.Episode) -> None:
        if not self.role_weights:
            return
        for trace in episode.traces:
            if not trace.has_error and trace.agent.trainable:
                self._weight_observations(trace)

    def _weight_observations(self, trace: vf.Trace) -> None:
        """Write graph-native ``ce`` weights over the env-provided
        observation tokens of later turns. Provenance is structural under v1:
        within a branch, the non-sampled nodes that follow the first model
        response (tool output, user feedback) are the env-provided
        observations — each such node's tokens get its message role's weight,
        narrowed by the optional user filter. The initial prompt (before the
        first response) is excluded. Selected tokens have ``mask`` False, so ce
        is the only component that trains them; samples where nothing is
        selected ship no ce stream.

        Content granularity: when a node carries the renderer's per-token
        ``is_content`` (``MessageNode.is_content``, parallel to ``token_ids``),
        only the message-body tokens are weighted — the chat-template scaffold
        (role tags, separators, tool-response wraps) is excluded. Nodes without
        attribution (the default renderer, or relay turns with no token ids)
        fall back to weighting the whole non-sampled span."""
        trainable_branches = [branch for branch, _ in iter_trainable_branches(trace)]
        filter_masks = self._filter_masks(trace, trainable_branches) if self.filter_fn is not None else None
        for branch_idx, branch in enumerate(trainable_branches):
            weights = [0.0] * len(branch.token_ids)
            offset = 0
            seen_response = False
            for node in branch.nodes:
                span = len(node.token_ids)
                role = node.message.role
                if seen_response and not node.sampled and role in self.role_weights:
                    weight = self.role_weights[role]
                    keep_mask = filter_masks[branch_idx] if filter_masks is not None else None
                    # Per-token content granularity when the renderer attributed it; otherwise
                    # the whole node span (is_content empty -> fall back to current behavior).
                    has_content = len(node.is_content) == span
                    for i in range(offset, offset + span):
                        if has_content and not node.is_content[i - offset]:
                            continue
                        if keep_mask is None or keep_mask[i]:
                            weights[i] = weight
                if node.sampled:
                    seen_response = True
                offset += span
            offset = 0
            for node in branch.nodes:
                end = offset + len(node.token_ids)
                node_weights = weights[offset:end]
                if any(node_weights):
                    streams = dict(node.loss_weights or {})
                    current = streams.get("ce", [0.0] * len(node.token_ids))
                    streams["ce"] = [max(old, new) for old, new in zip(current, node_weights, strict=True)]
                    node.loss_weights = streams
                offset = end

    def _filter_masks(self, trace: vf.Trace, trainable_branches: list) -> list[list[bool]]:
        """Invoke the user echo filter and validate its shape: one keep-mask
        per trainable branch, each spanning that branch's ``token_ids``."""
        assert self.filter_fn is not None
        masks = self.filter_fn(trace)
        if not isinstance(masks, list) or len(masks) != len(trainable_branches):
            got = len(masks) if isinstance(masks, list) else type(masks).__name__
            raise ValueError(
                f"echo filter must return one keep-mask per trainable branch: got {got}, expected {len(trainable_branches)}"
            )
        for branch_idx, (branch, mask) in enumerate(zip(trainable_branches, masks)):
            expected = len(branch.token_ids)
            if not isinstance(mask, list) or len(mask) != expected:
                got = len(mask) if isinstance(mask, list) else type(mask).__name__
                raise ValueError(
                    f"echo filter mask for branch {branch_idx} must span the branch's tokens: "
                    f"got {got}, expected {expected}"
                )
        return masks
