"""Turn-credit shaping: per-turn state scores turned into per-token credit."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.configs.algorithm import TurnCreditAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


def deltas(phis: list[float | None]) -> list[float]:
    """Per-turn progress from per-turn state scores: each scored turn's score minus
    the previous scored one. The first scored turn gets 0 — it sets the starting
    point, since the world's initial quality is not the model's doing. Unscored
    (``None``) turns get 0; their progress lands on the next scored turn."""
    out = [0.0] * len(phis)
    last: float | None = None
    for i, phi in enumerate(phis):
        if phi is None:
            continue
        if last is not None:
            out[i] = phi - last
        last = phi
    return out


def smear(progress: list[float], gamma: float) -> list[float]:
    """Split each turn's progress over that turn and the turns before it, weighted
    ``gamma^d`` at distance ``d`` and normalized so each turn's progress distributes
    exactly once. Returns each turn's collected credit; the total equals
    ``sum(progress)``. One backward pass: ``c_t = v_t + gamma * c_{t+1}``."""
    n = len(progress)
    values = []
    for t, r in enumerate(progress, start=1):
        z = t if gamma == 1.0 else (1.0 - gamma**t) / (1.0 - gamma)
        values.append(r / z)
    credits = [0.0] * n
    acc = 0.0
    for t in range(n - 1, -1, -1):
        acc = values[t] + gamma * acc
        credits[t] = acc
    return credits


class TurnCreditAlgorithm(Algorithm):
    """GRPO whose within-rollout credit follows per-turn state scores.

    The env scores the state of the world after each turn
    (``trace.info["turn_rewards"]``). Per trace, those scores become per-turn
    *progress* (deltas — only changing the world earns credit, staying in a good
    state does not), each turn's progress is smeared backward over the turns that
    led to it (``gamma``), and every token of a turn is shifted by ``beta`` times
    the turn's centered credit:

        a = A + beta * (c_turn - c_mean)

    ``A`` is the group-relative level, GRPO's baseline over the shaped return
    (final reward + net progress). ``c_mean`` is the token-weighted mean of the
    turn credits, so the shift is zero-sum over the trace's sampled tokens: the
    trace's total advantage stays exactly ``A``, and shaping only moves credit
    between its turns. A group of one has ``A = 0`` and trains on the shaping
    alone.

    Traces without ``turn_rewards`` fall back to plain GRPO (warned once per
    env). Traces that fork (compaction, subagents) are handled per branch: a
    turn's progress and credit are computed along its own chain of ancestors."""

    def __init__(self, config: TurnCreditAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.gamma = config.gamma
        self.beta = config.beta
        self.length_penalty = config.length_penalty
        self._warned_missing = False

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        scored: list[tuple[vf.Trace, dict[int, float] | None]] = []
        for episode, trace in iter_trainable_traces(episodes):
            scores = self._turn_scores(episode, trace)
            progress = self._chain_progress(trace, scores) if scores is not None else None
            scored.append((trace, progress))
        rewards = self._shaped_rewards([trace for trace, _ in scored])
        returns = [
            reward + (sum(progress.values()) if progress is not None else 0.0)
            for (_, progress), reward in zip(scored, rewards)
        ]
        baseline = sum(returns) / len(returns) if returns else 0.0
        for (trace, progress), ret in zip(scored, returns):
            level = ret - baseline
            if progress is None or self.beta == 0.0:
                assign_advantages(trace, level)
            else:
                assign_advantages(trace, self._token_advantages(trace, progress, level))

    def _shaped_rewards(self, traces: list[vf.Trace]) -> list[float]:
        """Each trace's final reward, less the GRPO length penalty when configured."""
        rewards = [trace.reward for trace in traces]
        penalty = self.length_penalty
        if penalty is None:
            return rewards
        pass_rate = sum(rewards) / len(rewards) if rewards else 0.0
        max_output = max((t.num_output_tokens for t in traces), default=0) or 1
        max_input = max((t.num_total_tokens - t.num_output_tokens for t in traces), default=0) or 1
        max_turns = max((t.num_turns for t in traces), default=0) or 1
        shaped = []
        for trace, reward in zip(traces, rewards):
            frac = (
                penalty.num_output_tokens_weight * (trace.num_output_tokens / max_output)
                + penalty.num_input_tokens_weight * ((trace.num_total_tokens - trace.num_output_tokens) / max_input)
                + penalty.num_turns_weight * (trace.num_turns / max_turns)
            )
            shaped.append(reward - pass_rate * frac)
        return shaped

    def _turn_scores(self, episode: vf.Episode, trace: vf.Trace) -> list[float | None] | None:
        """The env's per-turn state scores, validated: one ``float | None`` per
        sampled turn, in turn order. Missing entirely means shaping is off for
        this trace; malformed data raises."""
        raw = trace.info.get("turn_rewards")
        if raw is None:
            if not self._warned_missing:
                self._warned_missing = True
                get_logger().warning(
                    f"env '{episode.env.id}': traces carry no info['turn_rewards'] — "
                    "turn-credit shaping is inactive, training as plain GRPO"
                )
            return None
        num_turns = trace.num_turns
        if not isinstance(raw, list) or len(raw) != num_turns:
            got = len(raw) if isinstance(raw, list) else type(raw).__name__
            raise ValueError(
                f"info['turn_rewards'] must hold one entry per sampled turn: got {got}, "
                f"expected {num_turns} (trace {trace.id!r})"
            )
        scores: list[float | None] = []
        for i, entry in enumerate(raw):
            if entry is None:
                scores.append(None)
                continue
            if not isinstance(entry, (int, float)) or not math.isfinite(entry):
                raise ValueError(
                    f"info['turn_rewards'][{i}] must be a finite number or None: got {entry!r} (trace {trace.id!r})"
                )
            scores.append(float(entry))
        return scores

    def _chain_progress(self, trace: vf.Trace, scores: list[float | None]) -> dict[int, float]:
        """Each sampled turn's progress (``id(node) -> delta``), computed along its
        own branch's chain of sampled ancestors — so on a forked trace a turn's
        progress is measured against its actual predecessor, not whatever turn
        precedes it in node order. A turn on several branches keeps its first
        branch's value; net progress is the sum over turns."""
        turn_index = {id(node): i for i, node in enumerate(node for node in trace.nodes if node.sampled)}
        progress: dict[int, float] = {}
        for branch in trace.branches:
            chain = [node for node in branch.nodes if node.sampled]
            chain_progress = deltas([scores[turn_index[id(node)]] for node in chain])
            for node, value in zip(chain, chain_progress):
                progress.setdefault(id(node), value)
        return progress

    def _token_advantages(self, trace: vf.Trace, progress: dict[int, float], level: float) -> list[float]:
        """The trace's per-sampled-token advantages, in compact node order (what
        ``assign_advantages`` takes): ``level`` plus the turn's centered credit on
        every sampled token. Each branch smears its own chain's progress; a turn
        on several branches keeps its first branch's credit."""
        credits: dict[int, float] = {}
        for branch in trace.branches:
            chain = [node for node in branch.nodes if node.sampled]
            chain_credits = smear([progress[id(node)] for node in chain], self.gamma)
            for node, credit in zip(chain, chain_credits):
                credits.setdefault(id(node), credit)

        nodes = [node for node in trace.nodes if any(node.mask)]
        weights = {id(node): sum(node.mask) for node in nodes}
        total_weight = sum(weights.values())
        if total_weight == 0:
            return []
        center = sum(credits[key] * weight for key, weight in weights.items()) / total_weight

        stream: list[float] = []
        for node in nodes:
            value = level + self.beta * (credits[id(node)] - center)
            stream.extend([value] * sum(node.mask))
        return stream
