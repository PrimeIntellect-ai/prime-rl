"""Historical binary baseline with positive-anchored, zero-sum advantages."""

from __future__ import annotations

from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.configs.algorithm import LengthPenaltyConfig, NGUAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


def anchored_advantages(rewards: list[float], attempts: int, successes: int) -> list[float]:
    if not 0 <= successes <= attempts or attempts < len(rewards) or attempts == 0:
        raise ValueError("Invalid NGU historical reward counts")
    if any(reward not in (0, 1) for reward in rewards):
        raise ValueError("NGU requires unshaped binary rewards (0 or 1)")
    positives = sum(rewards)
    negatives = len(rewards) - positives
    if positives > successes or negatives > attempts - successes:
        raise ValueError("NGU retained rewards exceed historical counts")
    if not positives or not negatives:
        return [0.0] * len(rewards)
    positive = 1 - successes / attempts
    negative = -positive * positives / negatives
    return [positive if reward else negative for reward in rewards]


def length_penalized_advantages(
    traces: list[vf.Trace], attempts: int, successes: int, length_penalty: LengthPenaltyConfig | None
) -> list[float]:
    advantages = anchored_advantages([trace.reward for trace in traces], attempts, successes)
    if length_penalty is None:
        return advantages

    output = [trace.num_output_tokens for trace in traces]
    input_tokens = [trace.num_total_tokens - trace.num_output_tokens for trace in traces]
    turns = [trace.num_turns for trace in traces]
    max_output = max(max(output), 1)
    max_input = max(max(input_tokens), 1)
    max_turns = max(max(turns), 1)
    pass_rate = successes / attempts
    penalties = [
        pass_rate
        * (
            length_penalty.num_output_tokens_weight * output_tokens / max_output
            + length_penalty.num_input_tokens_weight * input_tokens / max_input
            + length_penalty.num_turns_weight * num_turns / max_turns
        )
        for output_tokens, input_tokens, num_turns in zip(output, input_tokens, turns, strict=True)
    ]
    mean_penalty = sum(penalties) / len(penalties)
    return [advantage - penalty + mean_penalty for advantage, penalty in zip(advantages, penalties, strict=True)]


class NGUAlgorithm(Algorithm):
    def __init__(self, config: NGUAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.length_penalty = config.length_penalty

    def score_history(self, episodes: list[vf.Episode], attempts: int, successes: int) -> None:
        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        advantages = length_penalized_advantages(traces, attempts, successes, self.length_penalty)
        for trace, advantage in zip(traces, advantages, strict=True):
            assign_advantages(trace, advantage)
            trace.info["ngu_attempts"] = attempts
            trace.info["ngu_successes"] = successes

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        self.score_history(episodes, len(traces), int(sum(trace.reward for trace in traces)))
