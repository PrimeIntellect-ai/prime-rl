"""Turn-credit shaping: per-turn state scores turned into per-token credit."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import TurnCreditAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.trajectories import iter_trainable_branches
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


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
    (``trace.info["turn_rewards"]``). Per rollout, those scores become per-turn
    *progress* (deltas — only changing the world earns credit, staying in a good
    state does not), each turn's progress is smeared backward over the turns that
    led to it (``gamma``), and every token of a turn is shifted by ``beta`` times
    the turn's centered credit:

        a = A + beta * (c_turn - c_mean)

    ``A`` is the group-relative level, GRPO's baseline over the shaped return
    (final reward + net progress). ``c_mean`` is the token-weighted mean of the
    turn credits, so the shift is zero-sum over the rollout's trainable tokens:
    the rollout's total advantage stays exactly ``A``, and shaping only moves
    credit between its turns. A group of one has ``A = 0`` and trains on the
    shaping alone.

    Rollouts without ``turn_rewards`` fall back to plain GRPO (warned once per
    env). Traces that fork (compaction, subagents) are handled per branch: a
    turn's progress and credit are computed along its own chain of ancestors."""

    def __init__(self, config: TurnCreditAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.gamma = config.gamma
        self.beta = config.beta
        self.length_penalty = config.length_penalty
        self._warned_missing = False

    async def score_group(self, group: list[Rollout]) -> None:
        turn_scores = [self._turn_scores(rollout) for rollout in group]
        progress = [sum(deltas(scores)) if scores is not None else 0.0 for scores in turn_scores]
        returns = [reward + net for reward, net in zip(self._shaped_rewards(group), progress)]
        baseline = sum(returns) / len(returns)
        for rollout, scores, ret in zip(group, turn_scores, returns):
            level = ret - baseline
            if scores is None or self.beta == 0.0:
                rollout.assign_advantages(level)
            else:
                rollout.assign_advantages(self._token_advantages(rollout, scores, level))

    def _shaped_rewards(self, group: list[Rollout]) -> list[float]:
        """Each rollout's final reward, less the GRPO length penalty when configured."""
        rewards = [rollout.reward for rollout in group]
        penalty = self.length_penalty
        if penalty is None:
            return rewards
        pass_rate = sum(rewards) / len(rewards)
        max_output = max((r.num_output_tokens for r in group), default=0) or 1
        max_input = max((r.num_total_tokens - r.num_output_tokens for r in group), default=0) or 1
        max_turns = max((r.num_turns for r in group), default=0) or 1
        shaped = []
        for rollout, reward in zip(group, rewards):
            frac = (
                penalty.num_output_tokens_weight * (rollout.num_output_tokens / max_output)
                + penalty.num_input_tokens_weight * ((rollout.num_total_tokens - rollout.num_output_tokens) / max_input)
                + penalty.num_turns_weight * (rollout.num_turns / max_turns)
            )
            shaped.append(reward - pass_rate * frac)
        return shaped

    def _turn_scores(self, rollout: Rollout) -> list[float | None] | None:
        """The env's per-turn state scores, validated: one ``float | None`` per
        sampled turn, in turn order. Missing entirely means shaping is off for
        this rollout; malformed data raises."""
        raw = rollout.info.get("turn_rewards")
        if raw is None:
            if not self._warned_missing:
                self._warned_missing = True
                get_logger().warning(
                    f"env '{rollout.env_name}': rollouts carry no info['turn_rewards'] — "
                    "turn-credit shaping is inactive, training as plain GRPO"
                )
            return None
        num_turns = rollout.num_turns
        if not isinstance(raw, list) or len(raw) != num_turns:
            got = len(raw) if isinstance(raw, list) else type(raw).__name__
            raise ValueError(
                f"info['turn_rewards'] must hold one entry per sampled turn: got {got}, "
                f"expected {num_turns} (env '{rollout.env_name}')"
            )
        scores: list[float | None] = []
        for i, entry in enumerate(raw):
            if entry is None:
                scores.append(None)
                continue
            if not isinstance(entry, (int, float)) or not math.isfinite(entry):
                raise ValueError(
                    f"info['turn_rewards'][{i}] must be a finite number or None: got {entry!r} "
                    f"(env '{rollout.env_name}')"
                )
            scores.append(float(entry))
        return scores

    def _token_advantages(self, rollout: Rollout, scores: list[float | None], level: float) -> list[float]:
        """The rollout's full-length per-token advantage stream: ``level`` plus the
        turn's centered credit on trainable tokens, 0 elsewhere. Credit is computed
        per branch along its own turn chain; a turn shared by several branches is
        credited where it trains (its first branch)."""
        turn_index: dict[int, int] = {}
        for node in rollout.nodes:
            if node.sampled:
                turn_index[id(node)] = len(turn_index)

        branches = list(iter_trainable_branches(rollout))
        credits: dict[int, float] = {}
        weights: dict[int, int] = {}
        for branch, mask in branches:
            chain = [node for node in branch.nodes if node.sampled]
            chain_credits = smear(deltas([scores[turn_index[id(node)]] for node in chain]), self.gamma)
            position = {id(node): i for i, node in enumerate(chain)}
            offset = 0
            for node in branch.nodes:
                span = len(node.token_ids)
                if node.sampled and id(node) not in credits:
                    trainable = sum(mask[offset : offset + span])
                    if trainable:
                        credits[id(node)] = chain_credits[position[id(node)]]
                        weights[id(node)] = trainable
                offset += span

        total_weight = sum(weights.values())
        if total_weight == 0:
            return []
        center = sum(credits[key] * weights[key] for key in credits) / total_weight

        stream: list[float] = []
        for branch, mask in branches:
            offset = 0
            for node in branch.nodes:
                span = len(node.token_ids)
                credit = credits.get(id(node))
                for i in range(offset, offset + span):
                    if mask[i] and credit is not None:
                        stream.append(level + self.beta * (credit - center))
                    else:
                        stream.append(0.0)
                offset += span
        return stream
