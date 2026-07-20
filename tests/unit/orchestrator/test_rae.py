"""RAE's estimator: per-role EMA baselines across groups, roles independent."""

import asyncio

import verifiers.v1 as vf

from prime_rl.orchestrator.algo.rae import RAEAlgorithm
from prime_rl.orchestrator.types import Rollout
from prime_rl.transport import TrainingSample


def _rollout(*, role: str, reward: float) -> Rollout:
    rollout = Rollout(
        task=vf.TraceTask(type="Task", data=vf.WireTaskData(idx=0)),
        role=role,
        rewards={"payoff": reward},
    )
    rollout.samples = [TrainingSample(token_ids=[1], mask=[True], logprobs=[0.0], temperatures=[1.0], env_name="e")]
    return rollout


def _algo(alpha: float = 0.5) -> RAEAlgorithm:
    algo = RAEAlgorithm.__new__(RAEAlgorithm)
    algo.alpha = alpha
    algo.baselines = {}
    return algo


def _advantage(rollout: Rollout) -> float:
    assert rollout.advantages is not None
    return rollout.advantages[0]


def test_first_group_centers_each_role_on_its_own_mean():
    algo = _algo()
    group = [
        _rollout(role="player0", reward=1.0),
        _rollout(role="player0", reward=-1.0),
        _rollout(role="player1", reward=-1.0),
    ]
    asyncio.run(algo.score_group(group))
    assert [_advantage(r) for r in group] == [1.0, -1.0, 0.0]
    assert algo.baselines == {"player0": 0.0, "player1": -1.0}


def test_baseline_is_an_ema_across_groups():
    """Second group is judged against the pre-update baseline; the EMA then
    absorbs its mean (alpha=0.5 → halfway)."""
    algo = _algo(alpha=0.5)
    asyncio.run(algo.score_group([_rollout(role="p", reward=0.0), _rollout(role="p", reward=0.0)]))
    assert algo.baselines["p"] == 0.0
    second = [_rollout(role="p", reward=1.0), _rollout(role="p", reward=1.0)]
    asyncio.run(algo.score_group(second))
    assert [_advantage(r) for r in second] == [1.0, 1.0]
    assert algo.baselines["p"] == 0.5
