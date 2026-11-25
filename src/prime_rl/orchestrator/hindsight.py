"""Relabel verified, fixed-horizon trajectories for goal-conditioned supervised learning."""

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass

import verifiers.v1 as vf

Goal = tuple[int, ...]
Prefix = tuple[int, ...]


@dataclass(frozen=True)
class FixedHorizonWorld:
    """Contract from trusted environment code. Changing a goal cannot change dynamics, horizon or legal actions.

    Supported goals are the goals that can be requested from this initial state. The renderer returns
    the exact token prefix before every action; all context tokens outside the actions must be
    deterministic given the goal, starting state and previous observed actions.
    """

    env_id: str
    initial: tuple[int, ...]
    requested: Goal
    supported_goals: Collection[Goal]
    action_ids: tuple[int, ...]
    horizon: int
    stop_condition: str
    transition: Callable[[tuple[int, ...], int], tuple[int, ...]]
    prefixes: Callable[[Goal, tuple[int, ...]], Sequence[Prefix]]


@dataclass(frozen=True)
class ReconstructedRollout:
    trace_id: str
    requested: Goal
    achieved: Goal
    original_reward: float
    actions: tuple[int, ...]
    factual_prefixes: tuple[Prefix, ...]


@dataclass(frozen=True)
class HindsightTarget:
    trace_id: str
    requested: Goal
    achieved: Goal
    original_reward: float
    actions: tuple[int, ...]
    action_ids: tuple[int, ...]
    prefixes: tuple[Prefix, ...]


def render_goal(world: FixedHorizonWorld, goal: Goal, actions: tuple[int, ...]) -> tuple[Prefix, ...]:
    prefixes = tuple(tuple(prefix) for prefix in world.prefixes(goal, actions))
    if len(prefixes) != len(actions) or not all(prefixes):
        raise ValueError("Each action needs its exact nonempty goal-dependent prefix")
    return prefixes


def _check_recorded_support(node: vf.MessageNode, action_ids: tuple[int, ...]) -> None:
    support = node.sampler_support
    if support is None:
        return
    steps = int(sum(node.mask))
    width = len(action_ids)
    if (
        len(support.counts) != steps
        or len(support.replayed) != steps
        or not all(support.replayed)
        or any(count != width for count in support.counts)
        or len(support.ids) != steps * width
    ):
        raise ValueError("Expected a full, fixed-size support for each sampled action")
    legal = set(action_ids)
    for start in range(0, len(support.ids), width):
        if set(support.ids[start : start + width]) != legal:
            raise ValueError("Recorded sampler support differs from the declared action grammar")


def _read_actions(trace: vf.Trace, world: FixedHorizonWorld) -> tuple[tuple[int, ...], tuple[Prefix, ...]]:
    call_nodes = [call.node for call in trace.calls]
    if not call_nodes or len(set(call_nodes)) != len(call_nodes):
        raise ValueError("Expected distinct committed model-call nodes")
    seen = []
    history = []
    actions = []
    observed_prefixes = []
    for index, node in enumerate(trace.nodes):
        if node.parent != (index - 1 if index else None) or node.semantic_parents:
            raise ValueError("Only a linear single-agent trace can be reconstructed")
        if len(node.mask) != len(node.token_ids):
            raise ValueError("Missing token mask")
        if index in call_nodes:
            if not node.sampled or node.message.role != "assistant" or not any(node.mask):
                raise ValueError("Each model call must contain an observed action")
            seen.append(index)
            _check_recorded_support(node, world.action_ids)
        for token, active in zip(node.token_ids, node.mask, strict=True):
            if active:
                if index not in call_nodes or token not in world.action_ids:
                    raise ValueError("Missing or unsupported sampled action")
                actions.append(int(token))
                observed_prefixes.append(tuple(history))
            history.append(int(token))
    if seen != call_nodes or len(actions) != world.horizon:
        raise ValueError("Expected ordered model calls covering the full declared horizon")
    return tuple(actions), tuple(observed_prefixes)


def reconstruct_rollout(episode: vf.Episode, world: FixedHorizonWorld) -> ReconstructedRollout:
    """Verify the original goal, actions and terminal equality reward without changing the episode."""
    if episode.env.id != world.env_id or not episode.ok or episode.is_timeout or len(episode.traces) != 1:
        raise ValueError("Expected one completed trace from the declared environment")
    trace = episode.traces[0]
    if (
        not trace.is_completed
        or not trace.ok
        or trace.dropped
        or trace.is_timeout
        or trace.stop_condition != world.stop_condition
        or not trace.agent.trainable
    ):
        raise ValueError("Expected a complete trainable fixed-horizon trace")
    if not world.action_ids or len(set(world.action_ids)) != len(world.action_ids) or world.horizon < 1:
        raise ValueError("Invalid fixed-horizon action space")
    if world.requested not in world.supported_goals:
        raise ValueError("The requested goal was outside the task's stated support")
    actions, factual_prefixes = _read_actions(trace, world)
    if render_goal(world, world.requested, actions) != factual_prefixes:
        raise ValueError("Factual token history differs from the environment renderer")
    achieved = world.initial
    for action in actions:
        achieved = world.transition(achieved, action)
    outcome = trace.rewards.get("reward")
    success = float(achieved == world.requested)
    if outcome is None or outcome.score != success or outcome.weight != 1 or trace.reward != success:
        raise ValueError("Expected the original verified terminal equality reward")
    return ReconstructedRollout(trace.id, world.requested, achieved, outcome.score, actions, factual_prefixes)


def make_hindsight_target(episode: vf.Episode, world: FixedHorizonWorld) -> HindsightTarget | None:
    """Keep the real actions and use the achieved goal when it lies in this start state's task support."""
    rollout = reconstruct_rollout(episode, world)
    if rollout.achieved not in world.supported_goals:
        return None
    prefixes = (
        rollout.factual_prefixes
        if rollout.requested == rollout.achieved
        else render_goal(world, rollout.achieved, rollout.actions)
    )
    return HindsightTarget(
        trace_id=rollout.trace_id,
        requested=rollout.requested,
        achieved=rollout.achieved,
        original_reward=rollout.original_reward,
        actions=rollout.actions,
        action_ids=world.action_ids,
        prefixes=prefixes,
    )


def make_hindsight_batch(
    episodes: Sequence[vf.Episode], worlds: Sequence[FixedHorizonWorld], *, metrics: dict[str, int] | None = None
) -> list[HindsightTarget | None]:
    """Return one slot per input episode. Omitted goals still count toward the learner's denominator."""
    targets = [make_hindsight_target(ep, w) for ep, w in zip(episodes, worlds, strict=True)]
    if metrics is not None:
        metrics.update(
            original_episodes=len(targets),
            unsupported_omitted=sum(t is None for t in targets),
            singleton_hits=sum(
                t is not None and len(set(w.supported_goals)) == 1 for t, w in zip(targets, worlds, strict=True)
            ),
            supported_distinct_goal_misses=sum(t is not None and t.original_reward == 0 for t in targets),
        )
    return targets
