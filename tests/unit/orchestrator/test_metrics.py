"""Tests for multi-environment metric aggregation in the orchestrator.

Regression tests for the example_id collision bug:
When multiple environments each auto-assign integer example_ids starting from 0,
grouping by example_id alone silently merges rollouts from different problems
across environments, corrupting all /all/ metrics.
"""

import pandas as pd
import pytest

from prime_rl.orchestrator.metrics import compute_solve_rates


def _make_df(env_rewards: dict[str, list[list[float]]]) -> pd.DataFrame:
    """Build a results_df from {env_name: [[rollout_rewards per problem], ...]}.

    Example:
        _make_df({"math": [[1., 1.], [0., 0.]], "code": [[0., 0.], [1., 1.]]})
    produces a DataFrame with 8 rows (2 envs × 2 problems × 2 rollouts each),
    with example_ids 0 and 1 for both environments (simulating auto-assignment).
    """
    rows = []
    for env_name, problems in env_rewards.items():
        for example_id, rollout_rewards in enumerate(problems):
            for reward in rollout_rewards:
                rows.append({"env_name": env_name, "example_id": example_id, "reward": reward})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Single-environment baseline
# ---------------------------------------------------------------------------


def test_single_env_all_solved():
    """Single env: all rollouts correct → solve_all = 1.0, solve_none = 0.0."""
    df = _make_df({"math": [[1., 1., 1., 1.], [1., 1., 1., 1.]]})
    solve_none, solve_all, effective = compute_solve_rates(df, rollouts_per_example=4)
    assert solve_all == pytest.approx(1.0)
    assert solve_none == pytest.approx(0.0)
    assert effective == pytest.approx(0.0)


def test_single_env_none_solved():
    """Single env: no rollouts correct → solve_none = 1.0, solve_all = 0.0."""
    df = _make_df({"math": [[0., 0., 0., 0.], [0., 0., 0., 0.]]})
    solve_none, solve_all, effective = compute_solve_rates(df, rollouts_per_example=4)
    assert solve_none == pytest.approx(1.0)
    assert solve_all == pytest.approx(0.0)
    assert effective == pytest.approx(0.0)


def test_single_env_half_solved():
    """Single env: one problem fully solved, one fully failed → each metric = 0.5."""
    df = _make_df({"math": [[1., 1., 1., 1.], [0., 0., 0., 0.]]})
    solve_none, solve_all, effective = compute_solve_rates(df, rollouts_per_example=4)
    assert solve_all == pytest.approx(0.5)
    assert solve_none == pytest.approx(0.5)
    assert effective == pytest.approx(0.0)


def test_single_env_partial_credit():
    """Single env: one problem partially solved → shows up in effective_batch_size."""
    # problem 0: fully solved, problem 1: partial (2/4 correct)
    df = _make_df({"math": [[1., 1., 1., 1.], [1., 0., 1., 0.]]})
    solve_none, solve_all, effective = compute_solve_rates(df, rollouts_per_example=4)
    assert solve_all == pytest.approx(0.5)
    assert solve_none == pytest.approx(0.0)
    assert effective == pytest.approx(0.5)


def test_solve_rates_sum_to_one():
    """solve_none + solve_all + effective_batch_size must always equal 1.0."""
    df = _make_df({"math": [[1., 0., 1., 0.], [0., 0., 0., 0.], [1., 1., 1., 1.]]})
    solve_none, solve_all, effective = compute_solve_rates(df, rollouts_per_example=4)
    assert solve_none + solve_all + effective == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Multi-environment metrics should not collide
# ---------------------------------------------------------------------------


def test_multi_env_solve_all_not_corrupted():
    """solve_all must not be corrupted when two envs share example_id integers.

    Setup: math fully solves all problems, code fails all problems.
    Both envs auto-assign example_ids 0 and 1.

    With the buggy groupby("example_id"), merged groups have reward_sum = 4+0 = 4,
    which accidentally equals rollouts_per_example=4, making solve_all look correct
    when it isn't — it's hiding that code failed everything.

    The correct answer: 2/4 groups fully solved (math/0, math/1); solve_all = 0.5.
    """
    df = _make_df({
        "math": [[1., 1., 1., 1.], [1., 1., 1., 1.]],  # fully solved
        "code": [[0., 0., 0., 0.], [0., 0., 0., 0.]],  # fully failed
    })
    solve_none, solve_all, effective = compute_solve_rates(df, rollouts_per_example=4)
    assert solve_all == pytest.approx(0.5), (
        "solve_all should be 0.5 (math solved, code failed), "
        f"got {solve_all} — likely groupby('example_id') collision"
    )
    assert solve_none == pytest.approx(0.5)
    assert effective == pytest.approx(0.0)


def test_multi_env_solve_none_not_corrupted():
    """solve_none must not be corrupted by cross-env example_id collision.

    Setup: math fails all, code fails all → solve_none should be 1.0.
    But if rewards from both envs are summed into merged groups,
    solve_none only triggers when the sum is exactly 0, which it is here,
    so this test catches a different failure mode (false negatives).
    """
    df = _make_df({
        "math": [[0., 0., 0., 0.], [0., 0., 0., 0.]],
        "code": [[0., 0., 0., 0.], [0., 0., 0., 0.]],
    })
    solve_none, solve_all, effective = compute_solve_rates(df, rollouts_per_example=4)
    assert solve_none == pytest.approx(1.0)
    assert solve_all == pytest.approx(0.0)


def test_multi_env_collision_mixed():
    """Three envs, three problems each; verifies solve rates are computed per (env, problem).

    math: problem 0 fully solved, problem 1 partial, problem 2 failed  → 1 solved, 1 failed
    code: all problems failed                                           → 0 solved, 3 failed
    logic: all problems fully solved                                    → 3 solved, 0 failed

    Total: 4/9 fully solved, 4/9 fully failed, 1/9 partial.
    """
    df = _make_df({
        "math":  [[1., 1., 1., 1.], [0., 1., 0., 1.], [0., 0., 0., 0.]],
        "code":  [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]],
        "logic": [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
    })
    solve_none, solve_all, effective = compute_solve_rates(df, rollouts_per_example=4)
    assert solve_all == pytest.approx(4 / 9)
    assert solve_none == pytest.approx(4 / 9)
    assert effective == pytest.approx(1 / 9)
    assert solve_none + solve_all + effective == pytest.approx(1.0)
