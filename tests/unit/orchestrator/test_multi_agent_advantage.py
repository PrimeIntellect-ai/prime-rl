import pytest
from verifiers.types import MemberRollout

from prime_rl.orchestrator.multi_agent_advantage import (
    RAEState,
    compute_rae_advantages,
    fan_out_for_multi_agent,
)

ENV_NAME = "debate_v1"


def _make_member_rollout(
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
        sampling_args={"temperature": 0.7},
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
    env_name: str = ENV_NAME,
    task: object = ENV_NAME,
    example_id: int | str = 1,
    members: list[tuple[str, float]] | None = None,
) -> dict[str, object]:
    if members is None:
        members = [("prover", 1.0), ("verifier", 0.0)]
    return {
        "env_name": env_name,
        "example_id": example_id,
        "task": task,
        "trajectory": [{"extras": {"member_id": member_id}} for member_id, _reward in members],
        "sampling_args": {"temperature": 0.7},
        "error": None,
        "trajectory_id": "ep-0",
        "mar_score": {
            "members": [{"member_id": member_id, "reward": reward} for member_id, reward in members],
            "episode_scalar": 1.0,
        },
    }


def test_cold_start_advantage_uses_post_update_baseline():
    state = RAEState(momentum=0.9)

    advantages = compute_rae_advantages([_make_member_rollout(reward=1.0)], state)

    assert advantages == [pytest.approx(0.9)]
    assert state.baselines[(ENV_NAME, 1, "prover")] == pytest.approx(0.1)


def test_second_batch_uses_persisted_baseline():
    state = RAEState(momentum=0.9)

    compute_rae_advantages([_make_member_rollout(reward=1.0)], state)
    advantages = compute_rae_advantages([_make_member_rollout(reward=1.0)], state)

    assert advantages == [pytest.approx(0.81)]
    assert state.baselines[(ENV_NAME, 1, "prover")] == pytest.approx(0.19)


def test_per_member_and_per_env_baselines_are_independent():
    state = RAEState(momentum=0.5)
    rollouts = [
        _make_member_rollout(env_name="env_a", member_id="prover", reward=1.0),
        _make_member_rollout(env_name="env_a", member_id="verifier", reward=0.0),
        _make_member_rollout(env_name="env_b", member_id="prover", reward=0.0),
    ]

    compute_rae_advantages(rollouts, state)

    assert state.baselines[("env_a", 1, "prover")] == pytest.approx(0.5)
    assert state.baselines[("env_a", 1, "verifier")] == pytest.approx(0.0)
    assert state.baselines[("env_b", 1, "prover")] == pytest.approx(0.0)


def test_within_batch_order_compounds_per_trajectory():
    state = RAEState(baselines={(ENV_NAME, 1, "prover"): 0.5}, momentum=0.9)
    rollouts = [
        _make_member_rollout(reward=1.0, episode_id="ep-0"),
        _make_member_rollout(reward=0.0, episode_id="ep-1"),
    ]

    advantages = compute_rae_advantages(rollouts, state)

    assert advantages == [pytest.approx(0.45), pytest.approx(-0.495)]
    assert state.baselines[(ENV_NAME, 1, "prover")] == pytest.approx(0.495)


def test_fan_out_carries_env_name_for_rae_identity():
    state = RAEState(momentum=0.5)
    units, mapping = fan_out_for_multi_agent([_make_episode_rollout(env_name="gpqa_debate", task={"question": "q"})])

    compute_rae_advantages(units, state)

    assert mapping == [[0, 1]]
    assert state.baselines[("gpqa_debate", 1, "prover")] == pytest.approx(0.5)
    assert state.baselines[("gpqa_debate", 1, "verifier")] == pytest.approx(0.0)


def test_fan_out_respects_trainable_member_predicate():
    rollouts = [_make_episode_rollout(members=[("debater_a", 0.8), ("debater_b", 0.2), ("judge", 1.0)])]

    units, mapping = fan_out_for_multi_agent(
        rollouts,
        is_trainable_member=lambda _rollout, member_id: member_id == "debater_a",
    )

    assert mapping == [[0]]
    assert [unit["member_id"] for unit in units] == ["debater_a"]


def test_rae_requires_string_identity_when_task_is_the_fallback():
    state = RAEState(momentum=0.5)
    rollout = _make_member_rollout(task={"question": "q"}, env_name=None)

    with pytest.raises(TypeError, match="string env_name"):
        compute_rae_advantages([rollout], state)
