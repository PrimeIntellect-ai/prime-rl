"""Unit tests for the multi-agent episode → member training-unit bridge."""

from __future__ import annotations

import json
from typing import Any

import verifiers as vf
from verifiers.types import MARScore, MemberScore, RolloutTiming, State, TrajectoryStep
from verifiers.utils.save_utils import state_to_output

from prime_rl.configs.multi_agent import (
    FixedMemberTargetConfig,
    MultiAgentConfig,
    TrainOneConfig,
)
from prime_rl.orchestrator.member_generation import (
    DISPATCH_ID_FIELD,
    compile_member_generation_plan,
    is_trainable_member,
)
from prime_rl.orchestrator.multi_agent_advantage import (
    RAEState,
    compute_rae_advantages,
    fan_out_for_multi_agent,
)

REQUIRED_COLUMNS = ["trajectory", "sampling_args", "trajectory_id"]


def _step(member_id: str, content: str = "x") -> TrajectoryStep:
    return TrajectoryStep(
        prompt=[],
        completion=[],
        response={
            "id": "r",
            "created": 0,
            "model": "m",
            "message": {"role": "assistant", "content": content, "finish_reason": "stop", "is_truncated": False},
        },
        tokens=None,
        reward=None,
        advantage=None,
        is_truncated=False,
        trajectory_id="ep-0",
        extras={"member_id": member_id, "phase": "p"},
    )


def _build_rollout(
    *,
    example_id: int = 1,
    trajectory_id: str = "ep-0",
    members: list[tuple[str, float]] | None = None,
    include_judge: bool = True,
) -> dict[str, Any]:
    members = members or [("debater_a", 1.0), ("debater_b", -1.0)]
    steps = [_step(mid) for mid, _ in members]
    if include_judge:
        steps.append(_step("judge", content="judge verdict"))
    state = State()
    state["example_id"] = example_id
    state["task"] = "debate_v1"
    state["trajectory"] = steps
    state["trajectory_id"] = trajectory_id
    state["sampling_args"] = {"temperature": 0.7}
    state["timing"] = RolloutTiming()
    member_scores = [MemberScore(member_id=mid, reward=r) for mid, r in members]
    if include_judge:
        member_scores.append(MemberScore(member_id="judge", reward=0.0))
    state["mar_score"] = MARScore(members=member_scores, episode_scalar=1.0)
    output = state_to_output(state, state_columns=REQUIRED_COLUMNS)
    # Round-trip through JSON to mirror the wire boundary the orchestrator
    # actually consumes — env_server returns serialized dicts.
    return json.loads(json.dumps(output, default=lambda o: o.model_dump(exclude_none=True)))


def test_fan_out_keeps_all_members_by_default():
    rollouts = [_build_rollout(include_judge=True)]
    units, mapping = fan_out_for_multi_agent(rollouts)
    assert len(units) == 3
    assert {u["member_id"] for u in units} == {"debater_a", "debater_b", "judge"}
    assert mapping == [[0, 1, 2]]


def test_fan_out_filters_with_trainability_predicate():
    rollouts = [_build_rollout(include_judge=True)]
    units, mapping = fan_out_for_multi_agent(
        rollouts,
        is_trainable_member=lambda _rollout, member_id: member_id != "judge",
    )
    assert {u["member_id"] for u in units} == {"debater_a", "debater_b"}
    assert mapping == [[0, 1]]


def test_fan_out_index_mapping_for_multiple_rollouts():
    """rollout_to_unit_idxs[i] must yield exactly the unit indices for
    rollout i. results_df aggregation in the orchestrator reads this to
    fold per-unit token counts back to per-episode metric rows."""
    rollouts = [
        _build_rollout(example_id=1, trajectory_id="ep-1"),
        _build_rollout(example_id=2, trajectory_id="ep-2"),
        _build_rollout(example_id=3, trajectory_id="ep-3"),
    ]
    units, mapping = fan_out_for_multi_agent(rollouts)
    # 3 rollouts × 3 members each = 9 units
    assert len(units) == 9
    assert mapping == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # Verify each mapping[i] points to units derived from rollout i
    for rollout_idx, unit_idxs in enumerate(mapping):
        unit_episodes = {units[ui]["episode_id"] for ui in unit_idxs}
        assert unit_episodes == {f"ep-{rollout_idx + 1}"}


def test_fan_out_pipeline_into_compute_rae_advantages():
    """End-to-end: fan_out → compute_rae_advantages produces one
    advantage per unit, and the unit/advantage zip behaves correctly
    for the orchestrator's per-unit sample.advantage assignment."""
    rollouts = [
        _build_rollout(example_id=1, members=[("debater_a", 1.0), ("debater_b", -1.0)]),
        _build_rollout(example_id=1, members=[("debater_a", -1.0), ("debater_b", 1.0)]),
    ]
    units, _mapping = fan_out_for_multi_agent(
        rollouts,
        is_trainable_member=lambda _rollout, member_id: member_id != "judge",
    )
    state = RAEState(momentum=0.9)
    advantages = compute_rae_advantages(units, state)
    assert len(advantages) == len(units) == 4
    # Both debater_a units share key (task, example_id=1, "debater_a") →
    # the EMA recursion compounds. With cold start b=0, momentum=0.9:
    #   debater_a unit 1: R=1.0 → b=0.1, A = 0.9
    #   debater_b unit 1: R=-1.0 → b=-0.1, A = -0.9
    #   debater_a unit 2: R=-1.0 → b=0.9*0.1 + 0.1*(-1) = -0.01, A = -0.99
    #   debater_b unit 2: R=1.0 → b=0.9*(-0.1) + 0.1*1 = 0.01, A = 0.99
    assert advantages[0] == 0.9
    assert advantages[1] == -0.9
    assert abs(advantages[2] - (-0.99)) < 1e-9
    assert abs(advantages[3] - 0.99) < 1e-9


def test_fan_out_handles_empty_rollouts_list():
    units, mapping = fan_out_for_multi_agent([])
    assert units == []
    assert mapping == []


def test_fan_out_trainability_predicate_keeps_only_matching_member():
    rollouts = [_build_rollout(example_id=1, trajectory_id="ep-1")]
    units, mapping = fan_out_for_multi_agent(
        rollouts,
        is_trainable_member=lambda _rollout, member_id: member_id == "debater_a",
    )
    assert [u["member_id"] for u in units] == ["debater_a"]
    assert mapping == [[0]]


def test_train_one_selects_one_trainable_debater_and_excludes_fixed_judge():
    config = MultiAgentConfig(
        train_one=TrainOneConfig(
            members=["debater_a", "debater_b"],
            seed=42,
            unselected="opponent",
        ),
        fixed={
            "opponent": FixedMemberTargetConfig(
                members=[],
                model="gpt-4.1-mini",
                base_url=["https://api.openai.com/v1"],
                api_key_var="OPENAI_API_KEY",
            ),
            "judge": FixedMemberTargetConfig(
                members=["judge"],
                model="judge-model",
                base_url=["http://judge:8000/v1"],
            ),
        },
    )
    rollout = _build_rollout(example_id=1, trajectory_id="ep-1")
    rollout[DISPATCH_ID_FIELD] = "dispatch-1"

    trainable_members = [
        member_id
        for member_id in ("debater_a", "debater_b", "judge")
        if is_trainable_member(config, rollout, member_id)
    ]

    assert len(trainable_members) == 1
    assert trainable_members[0] in {"debater_a", "debater_b"}
    assert not is_trainable_member(config, rollout, "judge")


def test_compile_member_generation_plan_routes_selected_fixed_and_judge_members():
    config = MultiAgentConfig(
        train_one=TrainOneConfig(
            members=["debater_a", "debater_b"],
            seed=0,
            unselected="opponent",
        ),
        fixed={
            "opponent": FixedMemberTargetConfig(
                members=[],
                model="gpt-4.1-mini",
                base_url=["https://api.openai.com/v1"],
                api_key_var="OPENAI_API_KEY",
                request_mode="chat",
                sampling={"temperature": 0.0},
            ),
            "judge": FixedMemberTargetConfig(
                members=["judge"],
                model="judge-model",
                base_url=["http://judge-1:8000/v1", "http://judge-2:8000/v1"],
                request_mode="chat",
                sampling={"temperature": 0.0, "max_completion_tokens": 256},
            ),
        },
    )
    dispatch_id = "dispatch-compile"
    plan = compile_member_generation_plan(
        config,
        member_ids=["debater_a", "debater_b", "judge"],
        default_client=vf.ClientConfig(
            client_type="renderer",
            api_base_url="http://learner:8000/v1",
            api_key_var="VLLM_API_KEY",
        ),
        default_model="learner-lora",
        learner_sampling_args={"temperature": 1.0, "extra_body": {"cache_salt": "7"}},
        fixed_sampling_args={"temperature": 1.0, "max_completion_tokens": 1024},
        dispatch_id=dispatch_id,
    )
    assert plan is not None

    trainable = [
        member
        for member in ("debater_a", "debater_b")
        if is_trainable_member(config, {DISPATCH_ID_FIELD: dispatch_id}, member)
    ]
    assert len(trainable) == 1
    selected = trainable[0]
    frozen = ({"debater_a", "debater_b"} - {selected}).pop()

    assert plan.members[selected].model == "learner-lora"
    assert plan.members[selected].client.client_type == "renderer"
    assert plan.members[selected].sampling_args["extra_body"] == {"cache_salt": "7"}
    assert plan.members[frozen].model == "gpt-4.1-mini"
    assert plan.members[frozen].client.client_type == "openai_chat_completions"
    assert plan.members[frozen].sampling_args == {
        "temperature": 0.0,
        "max_completion_tokens": 1024,
    }
    assert plan.members["judge"].model == "judge-model"
    assert plan.members["judge"].sampling_args == {
        "temperature": 0.0,
        "max_completion_tokens": 256,
    }
