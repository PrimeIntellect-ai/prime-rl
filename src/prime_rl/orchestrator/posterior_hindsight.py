"""Optional on-policy hindsight estimators with a frozen behavior-policy score on one alternative goal."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from random import Random

import torch
import verifiers.v1 as vf
from torch import Tensor

from prime_rl.orchestrator.hindsight import FixedHorizonWorld, Goal, Prefix, reconstruct_rollout, render_goal
from prime_rl.trainer.rl.posterior_hindsight import centered_pair_loss


@dataclass(frozen=True)
class GoalPath:
    prefixes: tuple[Prefix, ...]
    actions: tuple[int, ...]
    action_ids: tuple[int, ...]


@dataclass(frozen=True)
class PosteriorTarget:
    trace_id: str
    behavior_snapshot: str
    original_reward: float
    posterior: float
    achieved: GoalPath | None
    foil: GoalPath | None
    foil_posterior: float | None = None


def _factual_logp(episode: vf.Episode, behavior_snapshot: str) -> float:
    """The shared decoder has checked the grammar and trace structure. Require full sampler evidence."""
    trace = episode.traces[0]
    logs = []
    for call in trace.calls:
        node = trace.nodes[call.node]
        size = int(sum(node.mask))
        if node.sampler_support is None:
            raise ValueError("Expected captured and replayable sampler support for every action")
        if node.logprobs is None or len(node.logprobs) != size:
            raise ValueError("Expected one processed behavior-policy log probability per sampled action")
        end = 0
        for span in node.weight_versions or []:
            if span.start != end or span.end <= span.start or span.version != behavior_snapshot:
                raise ValueError("Expected one immutable behavior snapshot across all actions")
            end = span.end
        if end != size:
            raise ValueError("Missing behavior weight versions on sampled actions")
        logs.extend(map(float, node.logprobs))
    if any(not math.isfinite(x) or x > 0 for x in logs):
        raise ValueError("Invalid factual action log probability")
    return math.fsum(logs)


def make_posterior_targets(
    episodes: Sequence[vf.Episode],
    worlds: Sequence[FixedHorizonWorld],
    goal_priors: Sequence[Mapping[Goal, float]],
    behavior_snapshot: str,
    score_frozen: Callable[[str, list[GoalPath]], Sequence[Sequence[float]]],
    rng: Random,
) -> list[PosteriorTarget]:
    """Return one target per original episode, batching all alternative frozen-policy likelihoods.

    The scorer must use the same sampling temperature, grammar and probability convention as the actor.
    On a miss, the foil is the factual requested goal. On a hit, sample a foil from the prior
    conditional on excluding the achieved goal.
    """
    if not behavior_snapshot:
        raise ValueError("An immutable behavior snapshot is required")
    targets = []
    queries = []
    pending = []
    for episode, world, prior in zip(episodes, worlds, goal_priors, strict=True):
        positive = {g for g, p in prior.items() if p > 0}
        if (
            any(not math.isfinite(p) or not 0 <= p <= 1 for p in prior.values())
            or positive != set(world.supported_goals)
            or not math.isclose(math.fsum(prior.values()), 1.0, rel_tol=1e-9, abs_tol=1e-12)
            or (1.0 in prior.values() and len(positive) != 1)
        ):
            raise ValueError("Goal prior must be normalized and positive exactly on the declared support")
        record = reconstruct_rollout(episode, world)
        factual_logp = _factual_logp(episode, behavior_snapshot)
        factual = GoalPath(record.factual_prefixes, record.actions, world.action_ids)
        q = prior.get(record.achieved, 0.0)
        hit = record.requested == record.achieved
        if q == 0:
            achieved, foil, weight = None, factual, 0.0
        elif len(positive) == 1:
            achieved, foil, weight = factual, None, 1.0
        else:
            if hit:
                others = [(goal, p) for goal, p in prior.items() if p > 0 and goal != record.achieved]
                goals, masses = zip(*others, strict=True)
                alternative_goal = rng.choices(goals, weights=masses, k=1)[0]
            else:
                alternative_goal = record.achieved
            alternative = GoalPath(
                render_goal(world, alternative_goal, record.actions), record.actions, world.action_ids
            )
            achieved, foil = (factual, alternative) if hit else (alternative, factual)
            queries.append(alternative)
            other_mass = math.fsum(p for g, p in prior.items() if g != record.achieved)
            prior_log_odds = math.log(q) - math.log(other_mass)
            pending.append((len(targets), prior_log_odds, factual_logp, hit))
            weight = 0.0  # Filled from the batched alternative likelihood below.
        targets.append(
            PosteriorTarget(record.trace_id, behavior_snapshot, record.original_reward, weight, achieved, foil)
        )

    if queries:
        scores = score_frozen(behavior_snapshot, queries)
        for (i, prior_log_odds, factual_logp, hit), query, extra in zip(pending, queries, scores, strict=True):
            if len(extra) != len(query.actions) or any(not math.isfinite(x) or x > 0 for x in extra):
                raise ValueError("Expected one finite conditional log probability per alternative action")
            alt_logp = math.fsum(extra)
            log_ratio = factual_logp - alt_logp if hit else alt_logp - factual_logp
            odds = prior_log_odds + log_ratio
            small = math.exp(-abs(odds))
            dominant, tail = 1 / (1 + small), small / (1 + small)
            achieved_weight, foil_weight = (dominant, tail) if odds >= 0 else (tail, dominant)
            targets[i] = replace(targets[i], posterior=achieved_weight, foil_posterior=foil_weight)
    return targets


def centered_update_loss(
    targets: Sequence[PosteriorTarget],
    student_snapshot: str,
    score_student: Callable[[list[GoalPath]], Tensor],
    baseline: Tensor,
) -> Tensor:
    """Score all present roles on fresh activations from the exact collecting parameters."""
    if not targets or not student_snapshot or any(t.behavior_snapshot != student_snapshot for t in targets):
        raise ValueError("The centered update requires a nonempty batch from the collecting policy")
    queries = []
    positions = []
    n = len(targets)
    for i, target in enumerate(targets):
        for offset, path in ((0, target.achieved), (n, target.foil)):
            if path is not None:
                queries.append(path)
                positions.append(offset + i)
    scores = score_student(queries)
    if scores.shape != (len(queries),) or not scores.requires_grad:
        raise ValueError("Expected one fresh differentiable path log probability per scored context")
    pairs = scores.new_zeros(2 * n).index_copy(0, torch.tensor(positions, device=scores.device), scores)
    dtype = torch.promote_types(torch.float32, torch.promote_types(scores.dtype, baseline.dtype))
    posterior = torch.tensor([t.posterior for t in targets], dtype=dtype, device=scores.device)
    # Form a missing complement in the source precision before transferring externally built targets.
    foil = torch.tensor(
        [t.foil_posterior if t.foil_posterior is not None else 1 - t.posterior for t in targets],
        dtype=dtype,
        device=scores.device,
    )
    return centered_pair_loss(pairs[:n], pairs[n:], posterior, baseline, foil_posterior=foil)
