"""Historical binary baseline with positive-anchored, zero-sum advantages."""

import verifiers.v1 as vf

from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages


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


class NGUAlgorithm(Algorithm):
    def score_history(self, episodes: list[vf.Episode], attempts: int, successes: int) -> None:
        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        advantages = anchored_advantages([trace.reward for trace in traces], attempts, successes)
        for trace, advantage in zip(traces, advantages, strict=True):
            assign_advantages(trace, advantage)
            trace.info["ngu_attempts"] = attempts
            trace.info["ngu_successes"] = successes

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        self.score_history(episodes, len(traces), int(sum(trace.reward for trace in traces)))
