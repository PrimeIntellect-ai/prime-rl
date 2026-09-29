from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.configs.algorithm import CriPOSAlgoConfig
from prime_rl.orchestrator.algo.base import iter_trainable_traces
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.algo.routing import scalar_advantage
from prime_rl.orchestrator.trajectories import iter_trainable_branches

if TYPE_CHECKING:
    from renderers.base import Renderer, Tokenizer

    from prime_rl.orchestrator.clients import InferenceClient


def criterion_texts(trace: vf.Trace, key: str) -> dict[str, str]:
    criteria = trace.info.get(key)
    if criteria is None:
        criteria = getattr(trace.task.data, key, None)
    if (
        not isinstance(criteria, dict)
        or not criteria
        or any(
            not isinstance(name, str) or not name.strip() or not isinstance(text, str) or not text.strip()
            for name, text in criteria.items()
        )
    ):
        raise ValueError(f"cripo_s requires a nonempty {{reward_name: criterion_text}} mapping at '{key}'")
    return criteria


def suppressed_criteria(traces: list[vf.Trace], advantages: list[float], criteria: dict[str, str]) -> list[str]:
    """Equation (4), using unweighted binary scores for satisfaction."""
    selected: list[str] = []
    weights: dict[str, float] = {}
    for name in criteria:
        satisfying: list[int] = []
        for i, trace in enumerate(traces):
            reward = trace.rewards.get(name)
            if reward is None or reward.score not in (0.0, 1.0):
                raise ValueError(f"cripo_s criterion {name!r} requires a binary reward on every trace")
            if not math.isfinite(reward.weight) or reward.weight <= 0:
                raise ValueError(f"cripo_s criterion {name!r} requires a finite positive weight")
            if name in weights and weights[name] != reward.weight:
                raise ValueError(f"cripo_s criterion {name!r} has inconsistent weights within the cohort")
            weights[name] = reward.weight
            if reward.score == 1:
                satisfying.append(i)
        if not satisfying:
            continue
        credit = math.fsum(advantages[i] for i in satisfying)
        if credit < 0 or (credit == 0 and 2 * len(satisfying) < len(traces)):
            selected.append(name)
    return sorted(selected, key=lambda name: (-weights[name], name))


def flip_branch_advantages(
    branch: vf.Branch,
    mask: list[bool],
    student: list[float],
    teacher: list[float],
    maxima: list[float],
    *,
    threshold: float,
    advantage: float,
) -> None:
    """Equations (8–9), projected back to compact graph-node annotations."""
    size = len(branch.token_ids)
    if any(len(stream) != size for stream in (mask, student, teacher, maxima)):
        raise ValueError("cripo_s score streams and mask must align with branch token_ids")
    log_threshold = math.log(threshold)
    offset = 0
    for node in branch.nodes:
        end = offset + len(node.token_ids)
        if node.advantages is not None:
            sampled_index = 0
            for i, sampled in enumerate(node.mask, start=offset):
                if sampled:
                    if mask[i] and student[i] > teacher[i] and teacher[i] < maxima[i] + log_threshold:
                        node.advantages[sampled_index] = advantage
                    sampled_index += 1
        offset = end


class CriPOSAlgorithm(GRPOAlgorithm):
    """Preserve suppressed rubric behaviors through localized GRPO credit.

    The cohort supplies the suppression test; each eligible trace supplies one
    combined removal instruction. Fresh student and counterfactual prefills
    use untempered policy probabilities instead of stale sampling logprobs.
    """

    def __init__(self, config: CriPOSAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.config = config
        self.renderer: Renderer | None = None
        self.tokenizer: Tokenizer | None = None

    async def setup(self) -> None:
        from renderers.base import create_renderer, load_tokenizer

        self.tokenizer = load_tokenizer(self.clients.model_name)
        self.renderer = create_renderer(self.tokenizer, self.config.renderer)

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        entries = list(iter_trainable_traces(episodes))
        if not entries:
            return
        if len({id(episode) for episode, _ in entries}) != len(entries):
            raise ValueError("cripo_s requires one trainable trace per episode")
        traces = [trace for _, trace in entries]
        criteria = criterion_texts(traces[0], self.config.criteria_key)
        for trace in traces:
            if criterion_texts(trace, self.config.criteria_key) != criteria:
                raise ValueError("cripo_s requires the same criterion texts throughout a cohort")
            if not math.isfinite(trace.reward):
                raise ValueError("cripo_s requires finite aggregate rewards")

        await super().score_group(episodes)
        advantages = [scalar_advantage(trace) for trace in traces]
        if any(value is None or not math.isfinite(value) for value in advantages):
            raise ValueError("cripo_s requires finite GRPO advantages on every trainable trace")
        suppressed = suppressed_criteria(traces, advantages, criteria)

        for trace, advantage in zip(traces, advantages, strict=True):
            if advantage > 0 or (advantage == 0 and not self.config.flip_zero_advantage):
                continue
            names = [name for name in suppressed if trace.rewards[name].score == 1][: self.config.max_criteria]
            if not names:
                continue
            assert self.renderer is not None and self.tokenizer is not None, "call Algorithm.setup() first"
            for branch, mask in iter_trainable_branches(trace):
                if branch.multi_modal_data is not None:
                    raise ValueError("cripo_s counterfactual scoring currently supports text-only branches")
                previous_response = "\n".join(
                    self.tokenizer.decode(node.token_ids, skip_special_tokens=True)
                    for node in branch.nodes
                    if node.sampled
                )
                criteria_text = "\n".join(f"- {criteria[name]}" for name in names)
                hint = (
                    f"Given the previous response:\n<response>\n{previous_response}\n</response>\n"
                    "Revise it with the minimum necessary changes by modifying or deleting only the parts "
                    f"that satisfy the following criteria:\n{criteria_text}\n"
                    "Keep the rest of the response unchanged. Output only the revised response."
                )
                hint_ids = self.renderer.render_ids([{"role": "system", "content": hint}], add_generation_prompt=False)
                student, (teacher, maxima) = await asyncio.gather(
                    self.clients.score(branch.token_ids),
                    self.clients.score_with_max(hint_ids + branch.token_ids),
                )
                flip_branch_advantages(
                    branch,
                    mask,
                    student,
                    teacher[len(hint_ids) :],
                    maxima[len(hint_ids) :],
                    threshold=self.config.flip_threshold,
                    advantage=self.config.flip_advantage,
                )
