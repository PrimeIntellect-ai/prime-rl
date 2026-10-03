from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.configs.algorithm import CriPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages
from prime_rl.orchestrator.trajectories import iter_trainable_branches

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.orchestrator.clients import InferenceClient


def _reward_value(trace: vf.Trace, name: str) -> float:
    reward = trace.rewards.get(name)
    return reward.value if reward is not None else 0.0


def criterion_advantages(traces: list[vf.Trace]) -> dict[str, list[float]]:
    """Return per-criterion group-relative credit in trace order."""
    names = sorted({name for trace in traces for name in trace.rewards})
    return {
        name: [
            _reward_value(trace, name) - sum(_reward_value(other, name) for other in traces) / len(traces)
            for trace in traces
        ]
        for name in names
    }


def _criterion_text(trace: vf.Trace, name: str, key: str) -> str:
    criteria = trace.info.get(key)
    if isinstance(criteria, dict) and name in criteria:
        return str(criteria[name])
    return name


def _add_branch_bonus(
    branch: vf.Branch,
    branch_mask: list[bool],
    selected: list[bool],
    bonus: float,
) -> None:
    """Add a branch-aligned bonus without double-training shared graph nodes."""
    offset = 0
    for node in branch.nodes:
        end = offset + len(node.token_ids)
        node_mask = branch_mask[offset:end]
        node_selected = selected[offset:end]
        if node.advantages is not None:
            sampled_index = 0
            for is_sampled, is_trainable, is_selected in zip(node.mask, node_mask, node_selected, strict=True):
                if is_sampled:
                    if is_trainable and is_selected:
                        node.advantages[sampled_index] += bonus
                    sampled_index += 1
        offset = end


class CriPOAlgorithm(Algorithm):
    """Criterion-level credit assignment for rubric-based RL.

    CriPO-S starts with GRPO's response-level advantage. For criteria that are
    positive for one rollout but weak across its group, it asks the live policy
    to identify that criterion and compares the conditioned next-token scores
    with the original rollout. Tokens that become more likely yet remain far
    from the best token receive a small criterion-specific bonus.

    This is intentionally a separate algorithm because it adds policy prefills
    and changes the credit assignment semantics of a run.
    """

    def __init__(self, config: CriPOAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.config = config
        self.renderer: Renderer | None = None

    async def setup(self) -> None:
        from renderers.base import create_renderer, load_tokenizer

        self.renderer = create_renderer(load_tokenizer(self.clients.model_name), self.config.renderer)

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        renderer = self.renderer
        assert renderer is not None, "renderer not built — Algorithm.setup() must run first"

        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        if not traces:
            return

        rewards = [trace.reward for trace in traces]
        mean_reward = sum(rewards) / len(rewards)
        for trace, reward in zip(traces, rewards, strict=True):
            assign_advantages(trace, reward - mean_reward)

        per_criterion = criterion_advantages(traces)
        active: list[tuple[str, float]] = []
        for name, values in per_criterion.items():
            largest_positive = max((value for value in values if value > 0), default=0.0)
            if largest_positive:
                active.append((name, largest_positive))
        active = sorted(active, key=lambda item: item[1], reverse=True)[: self.config.max_criteria]

        for criterion, _ in active:
            positive_trace = next(
                trace for trace, value in zip(traces, per_criterion[criterion], strict=True) if value > 0
            )
            text = _criterion_text(positive_trace, criterion, self.config.criteria_key)
            hint = self.config.template.format(criterion=text)
            hint_block = renderer.render_ids([{"role": "system", "content": hint}], add_generation_prompt=False)
            candidates = [
                (trace, value) for trace, value in zip(traces, per_criterion[criterion], strict=True) if value > 0
            ]

            async def score_candidate(trace: vf.Trace, value: float) -> None:
                for branch, branch_mask in iter_trainable_branches(trace):
                    conditioned, maxima = await self.clients.score_with_max(hint_block + branch.token_ids)
                    conditioned = conditioned[len(hint_block) :]
                    maxima = maxima[len(hint_block) :]
                    selected = []
                    for allowed, conditioned_logprob, rollout_logprob, best_logprob in zip(
                        branch_mask, conditioned, branch.logprobs, maxima, strict=True
                    ):
                        selected.append(
                            allowed
                            and conditioned_logprob - rollout_logprob > 0
                            and conditioned_logprob < best_logprob + math.log(self.config.flip_threshold)
                        )
                    _add_branch_bonus(branch, branch_mask, selected, value * self.config.flip_tau)

            await asyncio.gather(*(score_candidate(trace, value) for trace, value in candidates))
