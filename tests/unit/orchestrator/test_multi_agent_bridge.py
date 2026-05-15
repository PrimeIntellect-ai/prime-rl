"""Bridge tests construct RolloutOutput by going through state_to_output.

This is the structural test fix that closes the test-fabrication gap that
hid the original P0 (state["mar_score"] silently dropped at serialization).
Every fixture goes through the real serialization boundary.
"""

import json
from typing import Any

import pytest
from verifiers import rollout_to_member_rollouts
from verifiers.types import MARScore, MemberRollout, MemberScore, State, TrajectoryStep
from verifiers.utils.save_utils import state_to_output

ENV_NAME = "debate_v1"
TEMPERATURE = 0.7


# ---------------------------------------------------------------------------
# Fixtures (all routed through state_to_output)
# ---------------------------------------------------------------------------


def _make_tagged_step(
    member_id: str,
    phase: str = "opening",
    trajectory_id: str = "traj-0",
    parse_error: str | None = None,
) -> TrajectoryStep:
    extras: dict[str, Any] = {
        "member_id": member_id,
        "phase": phase,
    }
    if parse_error is not None:
        extras["parse_error"] = parse_error
    return TrajectoryStep(
        prompt=[],
        completion=[],
        response={
            "id": "r0",
            "created": 0,
            "model": "test",
            "message": {
                "role": "assistant",
                "content": f"{member_id} says hi",
                "finish_reason": "stop",
                "is_truncated": False,
            },
        },
        tokens=None,
        reward=None,
        advantage=None,
        is_truncated=False,
        trajectory_id=trajectory_id,
        extras=extras,
    )


def _build_state(
    *,
    members: list[tuple[str, float]] | None = None,
    steps: list[TrajectoryStep] | None = None,
    example_id: int | str = 42,
    trajectory_id: str = "debate-0",
    episode_scalar: float = 1.0,
    error: BaseException | None = None,
) -> State:
    """Build a State that the env would produce, with mar_score written by rubric."""
    if members is None:
        members = [("A", 1.0), ("B", 0.0)]
    if steps is None:
        steps = [
            _make_tagged_step("A", "opening"),
            _make_tagged_step("B", "opening"),
            _make_tagged_step("A", "rebuttal"),
            _make_tagged_step("B", "rebuttal"),
        ]
    state = State()
    state["example_id"] = example_id
    state["task"] = ENV_NAME
    state["trajectory"] = steps
    state["trajectory_id"] = trajectory_id
    state["sampling_args"] = {"temperature": TEMPERATURE}
    state["mar_score"] = MARScore(
        members=[MemberScore(member_id=mid, reward=r) for mid, r in members],
        episode_scalar=episode_scalar,
        episode_metrics={"agreement": 1.0},
    )
    if error is not None:
        state["error"] = error
    return state


REQUIRED_COLUMNS = ["trajectory", "sampling_args", "trajectory_id"]


def _output_via_state_to_output(state: State) -> dict[str, Any]:
    """Round-trip state → output → JSON → output (as the wire would).

    Mirrors prime-rl's REQUIRED_STATE_COLUMNS — the orchestrator passes these
    so trajectory + sampling_args + trajectory_id reach the bridge.
    """
    output = state_to_output(state, state_columns=REQUIRED_COLUMNS)
    # JSON round-trip exercises the actual serialization the env-server uses.
    serialized = json.dumps(output, default=lambda o: o.model_dump(exclude_none=True))
    return json.loads(serialized)


# ---------------------------------------------------------------------------
# Bridge — happy path
# ---------------------------------------------------------------------------


def test_bridge_splits_by_member():
    state = _build_state()
    output = _output_via_state_to_output(state)
    rollouts = rollout_to_member_rollouts(output)
    assert len(rollouts) == 2
    assert {r["member_id"] for r in rollouts} == {"A", "B"}


def test_bridge_correct_step_counts():
    state = _build_state()
    output = _output_via_state_to_output(state)
    rollouts = rollout_to_member_rollouts(output)
    by_member = {r["member_id"]: r for r in rollouts}
    assert len(by_member["A"]["trajectory"]) == 2
    assert len(by_member["B"]["trajectory"]) == 2


def test_bridge_training_fields():
    state = _build_state()
    output = _output_via_state_to_output(state)
    rollouts = rollout_to_member_rollouts(output)
    a = next(r for r in rollouts if r["member_id"] == "A")
    assert a["example_id"] == 42
    # task is what state["task"] held — bridge no longer overwrites with env_name
    assert a["task"] == ENV_NAME
    assert a["sampling_args"] == {"temperature": TEMPERATURE}
    assert a["error"] is None
    assert a["reward"] == 1.0
    assert a["episode_id"] == "debate-0"


def test_bridge_per_member_rewards_match_mar_score():
    state = _build_state(members=[("A", 0.7), ("B", 0.3)])
    output = _output_via_state_to_output(state)
    rollouts = rollout_to_member_rollouts(output)
    by_member = {r["member_id"]: r["reward"] for r in rollouts}
    assert by_member == {"A": 0.7, "B": 0.3}


def test_bridge_preserves_temporal_order():
    state = _build_state()
    output = _output_via_state_to_output(state)
    rollouts = rollout_to_member_rollouts(output)
    a = next(r for r in rollouts if r["member_id"] == "A")
    assert a["trajectory"][0]["extras"]["phase"] == "opening"
    assert a["trajectory"][1]["extras"]["phase"] == "rebuttal"


def test_bridge_str_example_id():
    state = _build_state(example_id="mmlu_0001")
    output = _output_via_state_to_output(state)
    rollouts = rollout_to_member_rollouts(output)
    assert all(r["example_id"] == "mmlu_0001" for r in rollouts)


# ---------------------------------------------------------------------------
# Bridge — failure modes
# ---------------------------------------------------------------------------


def test_bridge_missing_mar_score_raises_key_error():
    """Missing mar_score = env did not run a MultiAgentRubric. Fail loud."""
    state = State()
    state["example_id"] = 1
    state["task"] = ENV_NAME
    state["trajectory"] = [_make_tagged_step("A")]
    state["trajectory_id"] = "ep-0"
    state["sampling_args"] = {"temperature": TEMPERATURE}
    output = state_to_output(state)
    with pytest.raises(KeyError, match="mar_score"):
        rollout_to_member_rollouts(output)


def test_bridge_missing_member_id_in_step_raises():
    bad_step = _make_tagged_step("A")
    bad_step["extras"] = {}  # strip member_id
    state = _build_state(steps=[bad_step])
    output = _output_via_state_to_output(state)
    with pytest.raises(ValueError, match="member_id"):
        rollout_to_member_rollouts(output)


def test_bridge_empty_trajectory_still_emits_per_member_rollouts():
    """A member with no own-turns is still in mar_score → still gets a
    MemberRollout (with empty trajectory). Bridge does not silently drop."""
    state = _build_state(steps=[])
    output = _output_via_state_to_output(state)
    rollouts = rollout_to_member_rollouts(output)
    # mar_score has 2 members; both get rollouts even with no steps.
    assert len(rollouts) == 2
    assert all(r["trajectory"] == [] for r in rollouts)


# ---------------------------------------------------------------------------
# state_to_output projection — invariants
# ---------------------------------------------------------------------------


def test_state_to_output_projects_episode_scalar_to_reward():
    state = _build_state(episode_scalar=0.42)
    output = state_to_output(state)
    assert output["reward"] == 0.42


def test_state_to_output_projects_per_member_reward_to_flat_metrics():
    state = _build_state(members=[("A", 0.8), ("B", 0.2)])
    output = state_to_output(state)
    assert output["reward/A"] == 0.8
    assert output["reward/B"] == 0.2


def test_state_to_output_serializes_mar_score_as_dict():
    state = _build_state()
    output = state_to_output(state)
    # In-memory: mar_score is dict (model_dump'd).
    assert isinstance(output["mar_score"], dict)
    assert "members" in output["mar_score"]


def test_state_to_output_episode_metrics_to_top_level():
    state = _build_state()
    output = state_to_output(state)
    assert output["agreement"] == 1.0


def test_bridge_round_trips_through_json():
    """End-to-end: state → state_to_output → JSON → bridge → MemberRollouts."""
    state = _build_state()
    output_dict = _output_via_state_to_output(state)
    rollouts = rollout_to_member_rollouts(output_dict)
    assert {r["member_id"] for r in rollouts} == {"A", "B"}


# ---------------------------------------------------------------------------
# MARScore — schema enforcement
# ---------------------------------------------------------------------------


def test_mar_score_rejects_duplicate_member_ids():
    with pytest.raises(ValueError, match="Duplicate member_id"):
        MARScore(
            members=[
                MemberScore(member_id="A", reward=1.0),
                MemberScore(member_id="A", reward=0.0),
            ],
            episode_scalar=0.5,
        )


def test_mar_score_rejects_empty_members():
    with pytest.raises(ValueError, match="cannot be empty"):
        MARScore(members=[], episode_scalar=0.0)


def test_mar_score_to_metrics_flat_canonical_keys():
    mar = MARScore(
        members=[
            MemberScore(
                member_id="A",
                reward=1.0,
                metrics={"accuracy": 1.0},
            ),
        ],
        episode_scalar=1.0,
        episode_metrics={"agreement": 1.0, "winner": 0.0},
    )
    flat = mar.to_metrics_flat()
    assert flat["reward/A"] == 1.0
    assert flat["accuracy/A"] == 1.0
    assert flat["agreement"] == 1.0
    assert flat["winner"] == 0.0


def test_mar_score_to_metrics_flat_omits_zero_parse_errors():
    """Zero parse_error_count should NOT appear in flat metrics."""
    mar = MARScore(
        members=[MemberScore(member_id="A", reward=1.0)],
        episode_scalar=1.0,
    )
    flat = mar.to_metrics_flat()
    assert "parse_errors/A" not in flat


def test_mar_score_to_metrics_flat_includes_nonzero_parse_errors():
    mar = MARScore(
        members=[MemberScore(member_id="A", reward=1.0, parse_error_count=3)],
        episode_scalar=1.0,
    )
    flat = mar.to_metrics_flat()
    assert flat["parse_errors/A"] == 3


def test_mar_score_by_id_returns_lookup():
    mar = MARScore(
        members=[
            MemberScore(member_id="A", reward=1.0),
            MemberScore(member_id="B", reward=0.0),
        ],
        episode_scalar=1.0,
    )
    by_id = mar.by_id()
    assert by_id["A"].reward == 1.0
    assert by_id["B"].reward == 0.0


# ---------------------------------------------------------------------------
# MemberRollout TypedDict shape (regression)
# ---------------------------------------------------------------------------


def test_member_rollout_has_required_training_fields():
    state = _build_state()
    output = _output_via_state_to_output(state)
    rollouts = rollout_to_member_rollouts(output)
    r = rollouts[0]
    required = {
        "example_id",
        "task",
        "trajectory",
        "sampling_args",
        "error",
        "reward",
        "episode_id",
        "member_id",
    }
    assert required.issubset(set(r.keys()))


def test_bridge_member_rollout_compatible_with_typed_dict():
    """Sanity: bridge return values can be passed where MemberRollout is expected."""
    state = _build_state()
    output = _output_via_state_to_output(state)
    rollouts = rollout_to_member_rollouts(output)
    sample: MemberRollout = rollouts[0]
    assert sample["member_id"] in {"A", "B"}
