import asyncio

import pytest

from prime_rl.configs.algorithm import GRPOAlgoConfig, TurnCreditAlgoConfig
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.algo.turn_credit import TurnCreditAlgorithm, deltas, smear
from tests.unit.orchestrator.test_advantage import _make_group, _scalar


def _run(group, *, turn_rewards, gamma=0.9, beta=1.0):
    """Attach per-rollout ``turn_rewards`` and drive ``score_group``."""
    for rollout, scores in zip(group, turn_rewards):
        if scores is not None:
            rollout.info["turn_rewards"] = scores
    algo = TurnCreditAlgorithm(TurnCreditAlgoConfig(gamma=gamma, beta=beta), policy_pool=None)
    asyncio.run(algo.score_group(group))
    return group


def _per_turn(rollout) -> list[float]:
    """The advantage each sampled turn's first token carries, in turn order —
    read by walking the branch's nodes against the flat advantage stream."""
    out = []
    offset = 0
    for branch in rollout.branches:
        for node in branch.nodes:
            if node.sampled and node.token_ids:
                out.append(rollout.advantages[offset])
            offset += len(node.token_ids)
    return out


# --------------------------------------------------------------------------
# deltas / smear: the two pure pieces.
# --------------------------------------------------------------------------


def test_deltas_are_progress():
    # First scored turn is the starting point (0); later turns score their change.
    assert deltas([1.0, 3.0, 2.0]) == [0.0, 2.0, -1.0]


def test_deltas_bridge_unscored_turns():
    # An unscored turn contributes 0; its progress lands on the next scored turn.
    assert deltas([1.0, None, 4.0]) == [0.0, 0.0, 3.0]


def test_deltas_all_none_is_zero():
    assert deltas([None, None]) == [0.0, 0.0]


def test_smear_conserves_mass():
    progress = [0.0, 2.0, -1.0, 0.5]
    for gamma in (0.0, 0.5, 0.9, 1.0):
        credits = smear(progress, gamma)
        assert sum(credits) == pytest.approx(sum(progress))


def test_smear_gamma_zero_is_identity():
    assert smear([0.0, 2.0, -1.0], 0.0) == pytest.approx([0.0, 2.0, -1.0])


def test_smear_gamma_one_spreads_evenly():
    # Turn 3's progress spreads evenly over turns 1-3.
    assert smear([0.0, 0.0, 3.0], 1.0) == pytest.approx([1.0, 1.0, 1.0])


def test_smear_decays_backward():
    # Progress at the last turn: nearer turns collect more of it.
    credits = smear([0.0, 0.0, 1.0], 0.5)
    assert credits[2] > credits[1] > credits[0] > 0.0
    assert sum(credits) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# The algorithm: level + shaping.
# --------------------------------------------------------------------------


def test_beta_zero_is_grpo():
    group = _run(
        _make_group(rewards=[1.0, 0.0], num_turns=[3, 3]),
        turn_rewards=[[0.0, 0.5, 1.0], [0.0, 0.2, 0.0]],
        beta=0.0,
    )
    # beta=0 still shifts the level by net progress: returns are 1+1 and 0+0.
    assert [_scalar(r) for r in group] == pytest.approx([1.0, -1.0])


def test_missing_turn_rewards_is_grpo():
    group = _run(_make_group(rewards=[1.0, 0.0]), turn_rewards=[None, None])
    plain = _make_group(rewards=[1.0, 0.0])
    asyncio.run(GRPOAlgorithm(GRPOAlgoConfig(), policy_pool=None).score_group(plain))
    assert [_scalar(r) for r in group] == pytest.approx([_scalar(r) for r in plain])


def test_total_advantage_is_the_level():
    """The shaping shift is zero-sum over trainable tokens: each rollout's summed
    advantage equals its level times its trainable-token count."""
    group = _make_group(rewards=[1.0, 0.0], completion_lengths=[6, 6], num_turns=[3, 3])
    _run(group, turn_rewards=[[0.0, 1.0, 0.5], [0.5, 0.5, 0.5]])
    returns = [1.0 + 0.5, 0.0 + 0.0]
    baseline = sum(returns) / 2
    for rollout, ret in zip(group, returns):
        n = sum(m for s in rollout.samples for m in s.mask)
        assert sum(rollout.advantages) == pytest.approx((ret - baseline) * n, abs=1e-6)


def test_shaping_orders_turns_by_progress():
    """Within a rollout, the turn that made the progress carries more credit than
    the turns that made none (gamma < 1 keeps most credit on the making turn)."""
    (rollout,) = _run(
        _make_group(rewards=[0.0], num_turns=[3]),
        turn_rewards=[[0.0, 0.0, 1.0]],
        gamma=0.5,
    )
    turns = _per_turn(rollout)
    assert turns[2] > turns[1] > turns[0]


def test_group_of_one_trains_on_shaping():
    """gs=1: the level is 0 but progress still orders the turns — the rollout
    carries a nonzero advantage stream (trainable where GRPO would be all-zero)."""
    (rollout,) = _run(
        _make_group(rewards=[1.0], num_turns=[2]),
        turn_rewards=[[0.0, 1.0]],
    )
    turns = _per_turn(rollout)
    assert turns[1] > 0.0 > turns[0]
    assert sum(rollout.advantages) == pytest.approx(0.0, abs=1e-6)


def test_uniform_reward_group_still_trains():
    """A group whose final rewards are identical (zero GRPO signal) still gets
    within-rollout signal from turn progress when net progress differs."""
    group = _make_group(rewards=[1.0, 1.0], num_turns=[2, 2])
    _run(group, turn_rewards=[[0.0, 1.0], [0.0, 0.0]])
    assert any(a != 0.0 for a in group[0].advantages)


def test_loitering_earns_nothing():
    """A rollout that reaches a good state and sits in it earns the same net
    progress as one that reaches it at the end: deltas ignore time-in-state."""
    group = _make_group(rewards=[0.0, 0.0], num_turns=[3, 3])
    _run(group, turn_rewards=[[1.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    # Net progress is 0 vs 1 -> the mover gets the higher level.
    assert _level(group[1]) > _level(group[0])


def _level(rollout) -> float:
    """The rollout's level: total advantage over trainable tokens."""
    n = sum(m for s in rollout.samples for m in s.mask)
    return sum(rollout.advantages) / n


def test_transient_spike_nets_zero():
    """A state score that spikes and reverts contributes nothing to the level —
    only kept progress counts."""
    group = _make_group(rewards=[0.0, 0.0], num_turns=[3, 3])
    _run(group, turn_rewards=[[0.0, 5.0, 0.0], [0.0, 0.0, 0.0]])
    assert _level(group[0]) == pytest.approx(_level(group[1]), abs=1e-6)


def test_sparse_scores_match_dense_endpoints():
    """Scoring every turn vs only the endpoints gives the same net progress."""
    group = _make_group(rewards=[0.0, 0.0], num_turns=[4, 4])
    _run(group, turn_rewards=[[0.0, 0.3, 0.6, 1.0], [0.0, None, None, 1.0]])
    assert _level(group[0]) == pytest.approx(_level(group[1]), abs=1e-6)


# --------------------------------------------------------------------------
# Validation.
# --------------------------------------------------------------------------


def test_wrong_length_raises():
    group = _make_group(rewards=[1.0], num_turns=[3])
    with pytest.raises(ValueError, match="one entry per sampled turn"):
        _run(group, turn_rewards=[[0.0, 1.0]])


def test_non_finite_raises():
    group = _make_group(rewards=[1.0], num_turns=[2])
    with pytest.raises(ValueError, match="finite"):
        _run(group, turn_rewards=[[0.0, float("nan")]])


def test_non_list_raises():
    group = _make_group(rewards=[1.0], num_turns=[2])
    group[0].info["turn_rewards"] = {"0": 1.0}
    algo = TurnCreditAlgorithm(TurnCreditAlgoConfig(), policy_pool=None)
    with pytest.raises(ValueError, match="one entry per sampled turn"):
        asyncio.run(algo.score_group(group))
