"""Unit tests for ``fan_out_for_multi_agent`` — the orchestrator's
per-step bridge wrapper that turns episode-level RolloutOutputs into
per-member training units.

The orchestrator inlines this against ``rollout_to_member_rollouts`` from
verifiers, then feeds the units to ``compute_rae_advantages``. These
tests exercise the routing + judge filter + index mapping that the
production code depends on for results_df aggregation.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from verifiers.types import MARScore, MemberScore, State, TrajectoryStep
from verifiers.utils.save_utils import state_to_output

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
    member_scores = [MemberScore(member_id=mid, reward=r) for mid, r in members]
    if include_judge:
        member_scores.append(MemberScore(member_id="judge", reward=0.0))
    state["mar_score"] = MARScore(members=member_scores, episode_scalar=1.0)
    output = state_to_output(state, state_columns=REQUIRED_COLUMNS)
    # Round-trip through JSON to mirror the wire boundary the orchestrator
    # actually consumes — env_server returns serialized dicts.
    return json.loads(json.dumps(output, default=lambda o: o.model_dump(exclude_none=True)))


def test_fan_out_drops_judge_by_default():
    """drop_judge=True (default) excludes member_id="judge" from training
    units. Judge rewards are 0 by zero_sum_reward construction → wasted
    gradient compute."""
    rollouts = [_build_rollout(include_judge=True)]
    units, mapping = fan_out_for_multi_agent(rollouts)
    assert len(units) == 2
    assert {u["member_id"] for u in units} == {"debater_a", "debater_b"}
    assert mapping == [[0, 1]]


def test_fan_out_keeps_judge_when_requested():
    """drop_judge=False threads the judge through (diagnostic-only paths
    like SFT-on-judge-transcripts)."""
    rollouts = [_build_rollout(include_judge=True)]
    units, mapping = fan_out_for_multi_agent(rollouts, drop_judge=False)
    assert {u["member_id"] for u in units} == {"debater_a", "debater_b", "judge"}
    assert mapping == [[0, 1, 2]]


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
    # 3 rollouts × 2 members each (judge dropped) = 6 units
    assert len(units) == 6
    assert mapping == [[0, 1], [2, 3], [4, 5]]
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
    units, _mapping = fan_out_for_multi_agent(rollouts)
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


def test_fan_out_filter_by_learner_seat_keeps_only_matching_member():
    """filter_by_learner_seat=True reads rollout.info['learner_seat'] and
    keeps only that member's unit. External-opponent training: the frozen
    opposite seat's trajectory never reaches the trainer."""
    rollouts = [_build_rollout(example_id=1, trajectory_id="ep-1")]
    rollouts[0]["info"] = {"learner_seat": "debater_a"}
    units, mapping = fan_out_for_multi_agent(rollouts, filter_by_learner_seat=True)
    assert [u["member_id"] for u in units] == ["debater_a"]
    assert mapping == [[0]]


def test_fan_out_filter_by_learner_seat_missing_info_raises():
    """filter_by_learner_seat=True on a rollout without info.learner_seat
    is a config mismatch (filter enabled on a self-play env), not a silent
    no-op."""
    rollouts = [_build_rollout(example_id=1)]
    rollouts[0]["info"] = {}
    with pytest.raises(ValueError, match="info\\['learner_seat'\\] is missing"):
        fan_out_for_multi_agent(rollouts, filter_by_learner_seat=True)
