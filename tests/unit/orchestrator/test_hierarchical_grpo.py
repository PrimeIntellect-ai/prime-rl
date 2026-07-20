"""Hierarchical GRPO's partition rule: fanned roles baseline within their episode,
once-per-episode roles across the group's episodes."""

import asyncio

import verifiers.v1 as vf

from prime_rl.orchestrator.algo.hierarchical_grpo import HierarchicalGRPOAlgorithm
from prime_rl.orchestrator.types import Rollout
from prime_rl.transport import TrainingSample


def _rollout(*, role: str, episode_id: str, reward: float) -> Rollout:
    rollout = Rollout(
        task=vf.TraceTask(type="Task", data=vf.WireTaskData(idx=0)),
        role=role,
        rewards={"r": reward},
    )
    rollout.episode_id = episode_id
    rollout.samples = [TrainingSample(token_ids=[1], mask=[True], logprobs=[0.0], temperatures=[1.0], env_name="e")]
    return rollout


def _algo() -> HierarchicalGRPOAlgorithm:
    algo = HierarchicalGRPOAlgorithm.__new__(HierarchicalGRPOAlgorithm)
    return algo


def _advantage(rollout: Rollout) -> float:
    assert rollout.advantages is not None
    return rollout.advantages[0]


def test_fanned_role_baselines_within_its_episode():
    """Solvers of episode 1 (rewards 1, 0) baseline against each other, not
    against episode 2's solvers (0, 0) — different minted problems."""
    group = [
        _rollout(role="solver", episode_id="ep1", reward=1.0),
        _rollout(role="solver", episode_id="ep1", reward=0.0),
        _rollout(role="solver", episode_id="ep2", reward=0.0),
        _rollout(role="solver", episode_id="ep2", reward=0.0),
    ]
    asyncio.run(_algo().score_group(group))
    assert [_advantage(r) for r in group] == [0.5, -0.5, 0.0, 0.0]


def test_singleton_role_baselines_across_episodes():
    """Proposers (one per episode, same seed task) form one cross-episode
    partition; a solver fan in any episode stays per-episode."""
    group = [
        _rollout(role="proposer", episode_id="ep1", reward=1.0),
        _rollout(role="solver", episode_id="ep1", reward=1.0),
        _rollout(role="solver", episode_id="ep1", reward=0.0),
        _rollout(role="proposer", episode_id="ep2", reward=0.0),
        _rollout(role="solver", episode_id="ep2", reward=1.0),
        _rollout(role="solver", episode_id="ep2", reward=1.0),
    ]
    asyncio.run(_algo().score_group(group))
    proposers = [r for r in group if r.role == "proposer"]
    assert [_advantage(r) for r in proposers] == [0.5, -0.5]
    ep1_solvers = [_advantage(r) for r in group if r.role == "solver" and r.episode_id == "ep1"]
    assert ep1_solvers == [0.5, -0.5]
    ep2_solvers = [_advantage(r) for r in group if r.role == "solver" and r.episode_id == "ep2"]
    assert ep2_solvers == [0.0, 0.0]


def test_single_agent_group_degenerates_to_plain_grpo():
    group = [
        _rollout(role=None, episode_id="ep1", reward=1.0),
        _rollout(role=None, episode_id="ep2", reward=0.0),
    ]
    asyncio.run(_algo().score_group(group))
    assert [_advantage(r) for r in group] == [0.5, -0.5]
