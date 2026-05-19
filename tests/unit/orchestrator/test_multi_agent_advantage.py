"""Tests for RAEState + compute_rae_advantages.

This is the per-(env_name, example_id, member_id) baseline path that the
multi-agent training loop will use as the bridge consumer. Algorithm:
A_i = R_i - b[(env_name_i, example_id_i, member_id_i)], EMA baseline update.
"""

import pytest
from verifiers.types import MemberRollout

from prime_rl.orchestrator.multi_agent_advantage import (
    RAEState,
    compute_rae_advantages,
    fan_out_for_multi_agent,
)

ENV_NAME = "debate_v1"
TEMPERATURE = 0.7


def _make_rollout(
    *,
    example_id: int | str = 1,
    member_id: str = "prover",
    reward: float = 1.0,
    episode_id: str = "ep-0",
    task: object = ENV_NAME,
    env_name: str | None = None,
) -> MemberRollout:
    rollout = MemberRollout(
        example_id=example_id,
        task=task,
        trajectory=[],
        sampling_args={"temperature": TEMPERATURE},
        error=None,
        reward=reward,
        episode_id=episode_id,
        member_id=member_id,
    )
    if env_name is not None:
        rollout["env_name"] = env_name
    return rollout


def _make_episode_rollout(
    *,
    env_name: str,
    task: object,
    example_id: int | str = 1,
    member_id: str = "prover",
    reward: float = 1.0,
) -> dict[str, object]:
    return {
        "env_name": env_name,
        "example_id": example_id,
        "task": task,
        "trajectory": [{"extras": {"member_id": member_id}}],
        "sampling_args": {"temperature": TEMPERATURE},
        "error": None,
        "trajectory_id": "ep-0",
        "mar_score": {
            "members": [{"member_id": member_id, "reward": reward}],
            "episode_scalar": 1.0,
        },
    }


# ---------------------------------------------------------------------------
# RAE invariants
# ---------------------------------------------------------------------------


def test_cold_start_advantage_is_reward_minus_post_update_baseline():
    """Per SPIRAL Alg.1: b ← α·b + (1-α)·R BEFORE A = R - b. With cold-start
    b=0 and momentum=0.9, the baseline after the first update equals
    (1-α)·R; the advantage becomes R - (1-α)·R = α·R."""
    state = RAEState(baselines={}, momentum=0.9)
    rollouts = [
        _make_rollout(reward=1.0, member_id="prover"),
        _make_rollout(reward=0.0, member_id="verifier"),
    ]
    advs = compute_rae_advantages(rollouts, state)
    assert advs == [pytest.approx(0.9), pytest.approx(0.0)]


def test_baselines_update_after_batch():
    """After one batch, baselines should reflect EMA update."""
    state = RAEState(baselines={}, momentum=0.9)
    rollouts = [_make_rollout(reward=1.0, member_id="prover")]
    compute_rae_advantages(rollouts, state)
    assert state.baselines[(ENV_NAME, 1, "prover")] == pytest.approx(0.1)


def test_second_batch_uses_updated_baseline():
    """Two sequential batches with R=1, momentum=0.9 (cold start):
    b_1 = 0.1; A_1 = 1 - 0.1 = 0.9
    b_2 = 0.9·0.1 + 0.1·1 = 0.19; A_2 = 1 - 0.19 = 0.81"""
    state = RAEState(baselines={}, momentum=0.9)
    advs1 = compute_rae_advantages([_make_rollout(reward=1.0)], state)
    assert advs1 == [pytest.approx(0.9)]
    advs2 = compute_rae_advantages([_make_rollout(reward=1.0)], state)
    assert advs2 == [pytest.approx(0.81)]
    assert state.baselines[(ENV_NAME, 1, "prover")] == pytest.approx(0.19)


def test_degenerate_group_always_positive():
    """Member always wins (reward=1.0). EMA baseline < 1.0 for finite t,
    so advantage stays positive — the agent keeps learning."""
    state = RAEState(baselines={}, momentum=0.9)
    for _ in range(20):
        advs = compute_rae_advantages([_make_rollout(reward=1.0)], state)
    assert advs[0] > 0
    assert state.baselines[(ENV_NAME, 1, "prover")] < 1.0


def test_per_member_baselines_independent():
    state = RAEState(baselines={}, momentum=0.5)
    rollouts = [
        _make_rollout(example_id=1, member_id="prover", reward=1.0),
        _make_rollout(example_id=1, member_id="verifier", reward=0.0),
    ]
    compute_rae_advantages(rollouts, state)
    assert state.baselines[(ENV_NAME, 1, "prover")] == pytest.approx(0.5)
    assert state.baselines[(ENV_NAME, 1, "verifier")] == pytest.approx(0.0)


def test_per_example_baselines_independent():
    state = RAEState(baselines={}, momentum=0.5)
    rollouts = [
        _make_rollout(example_id=1, member_id="prover", reward=1.0),
        _make_rollout(example_id=2, member_id="prover", reward=0.0, episode_id="ep-1"),
    ]
    compute_rae_advantages(rollouts, state)
    assert state.baselines[(ENV_NAME, 1, "prover")] == pytest.approx(0.5)
    assert state.baselines[(ENV_NAME, 2, "prover")] == pytest.approx(0.0)


def test_string_task_fallback_baselines_independent():
    """String task identities still partition direct MemberRollout inputs."""
    state = RAEState(baselines={}, momentum=0.5)
    r_a = _make_rollout(example_id=1, member_id="prover", reward=1.0, task="env_a")
    r_b = _make_rollout(example_id=1, member_id="prover", reward=0.0, task="env_b")
    compute_rae_advantages([r_a, r_b], state)
    assert state.baselines[("env_a", 1, "prover")] == pytest.approx(0.5)
    assert state.baselines[("env_b", 1, "prover")] == pytest.approx(0.0)


def test_unhashable_task_uses_env_name_identity():
    state = RAEState(baselines={}, momentum=0.5)
    task_row = {"question": "Which option?", "answer": "A"}
    rollout = _make_rollout(task=task_row, env_name="gpqa_debate", reward=1.0)
    advs = compute_rae_advantages([rollout], state)
    assert advs == [pytest.approx(0.5)]
    assert state.baselines[("gpqa_debate", 1, "prover")] == pytest.approx(0.5)


def test_env_name_baselines_independent_when_task_rows_match():
    state = RAEState(baselines={}, momentum=0.5)
    task_row = {"question": "shared row", "answer": "A"}
    r_a = _make_rollout(task=task_row, env_name="env_a", reward=1.0)
    r_b = _make_rollout(task=task_row, env_name="env_b", reward=0.0)
    compute_rae_advantages([r_a, r_b], state)
    assert state.baselines[("env_a", 1, "prover")] == pytest.approx(0.5)
    assert state.baselines[("env_b", 1, "prover")] == pytest.approx(0.0)


def test_fan_out_carries_env_name_for_rae_identity():
    state = RAEState(baselines={}, momentum=0.5)
    units, mapping = fan_out_for_multi_agent([_make_episode_rollout(env_name="gpqa_debate", task={"question": "q"})])
    assert mapping == [[0]]
    compute_rae_advantages(units, state)
    assert state.baselines[("gpqa_debate", 1, "prover")] == pytest.approx(0.5)


def test_within_batch_ordering_compounds_per_trajectory():
    """Per SPIRAL Alg.1, the EMA recursion runs once per τ. When two
    rollouts share a key, swapping their order yields different advantages
    AND different end baselines — the recursion is the point.

    Forward [r=1.0, r=0.0], b₀=0.5, momentum=0.9:
        b₁ = 0.9·0.5 + 0.1·1 = 0.55; A₁ = 1 - 0.55 = 0.45
        b₂ = 0.9·0.55 + 0.1·0 = 0.495; A₂ = 0 - 0.495 = -0.495

    Reverse [r=0.0, r=1.0]:
        b₁ = 0.9·0.5 + 0.1·0 = 0.45; A₁ = 0 - 0.45 = -0.45
        b₂ = 0.9·0.45 + 0.1·1 = 0.505; A₂ = 1 - 0.505 = 0.495"""
    state_fwd = RAEState(baselines={(ENV_NAME, 1, "prover"): 0.5}, momentum=0.9)
    state_rev = RAEState(baselines={(ENV_NAME, 1, "prover"): 0.5}, momentum=0.9)
    r1 = _make_rollout(reward=1.0, episode_id="ep-0")
    r2 = _make_rollout(reward=0.0, episode_id="ep-1")
    advs_fwd = compute_rae_advantages([r1, r2], state_fwd)
    advs_rev = compute_rae_advantages([r2, r1], state_rev)
    assert advs_fwd == [pytest.approx(0.45), pytest.approx(-0.495)]
    assert advs_rev == [pytest.approx(-0.45), pytest.approx(0.495)]
    # End baselines DIFFER — order matters.
    assert state_fwd.baselines[(ENV_NAME, 1, "prover")] == pytest.approx(0.495)
    assert state_rev.baselines[(ENV_NAME, 1, "prover")] == pytest.approx(0.505)


def test_empty_batch_preserves_baselines():
    state = RAEState(baselines={(ENV_NAME, 1, "prover"): 0.5})
    advs = compute_rae_advantages([], state)
    assert advs == []
    assert state.baselines[(ENV_NAME, 1, "prover")] == 0.5


def test_str_example_id_keys_baseline_correctly():
    """HuggingFace UID-style example_ids (e.g. 'mmlu_0001') flow through
    the RAE key as-is; baseline lookup matches by tuple equality."""
    state = RAEState(baselines={}, momentum=0.5)
    rollouts = [_make_rollout(example_id="mmlu_0001", reward=1.0)]
    compute_rae_advantages(rollouts, state)
    assert state.baselines[(ENV_NAME, "mmlu_0001", "prover")] == pytest.approx(0.5)


def test_repeated_key_in_batch_compounds_per_trajectory():
    """Two rollouts sharing (env_name, example_id, member_id) trigger N
    sequential per-trajectory EMA updates (SPIRAL Alg.1, not a single
    mean-aggregated update). With momentum=0.5, b₀=0, rewards [1.0, 0.0]:
        b₁ = 0.5·0 + 0.5·1 = 0.5; A₁ = 1 - 0.5 = 0.5
        b₂ = 0.5·0.5 + 0.5·0 = 0.25; A₂ = 0 - 0.25 = -0.25"""
    state = RAEState(baselines={}, momentum=0.5)
    rs = [
        _make_rollout(reward=1.0, episode_id="ep-0"),
        _make_rollout(reward=0.0, episode_id="ep-1"),
    ]
    advs = compute_rae_advantages(rs, state)
    assert advs == [pytest.approx(0.5), pytest.approx(-0.25)]
    assert state.baselines[(ENV_NAME, 1, "prover")] == pytest.approx(0.25)


def test_zero_reward_from_errored_rollout_keys_correctly():
    """Errored mar_score produces MemberRollouts with reward=0.0 — RAE
    should treat them like any other zero-reward sample (no special case).
    Per Alg.1: b = 0.5·0.7 + 0.5·0 = 0.35; A = 0 - 0.35 = -0.35."""
    state = RAEState(baselines={(ENV_NAME, 1, "prover"): 0.7}, momentum=0.5)
    advs = compute_rae_advantages([_make_rollout(reward=0.0)], state)
    assert advs == [pytest.approx(-0.35)]
