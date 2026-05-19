"""Adversarial stress tests for the MARScore architectural cleanup.

Each test targets a specific invariant the cleanup either established or
fixed. These are the regression suite for the design — if any of these
break, the cleanup has been undone or a fix has been re-introduced.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import verifiers as vf
from verifiers import rollout_to_member_rollouts
from verifiers.envs.multi_agent_env import MultiAgentEnv
from verifiers.envs.multi_agent_kernel import (
    ContentChannels,
    KernelState,
    StaticSchedule,
    TurnSlot,
    apply_action,
)
from verifiers.types import MARScore, MemberScore, State, TrajectoryStep
from verifiers.utils.save_utils import state_to_output
from verifiers.utils.usage_utils import StateUsageTracker

REQUIRED_COLUMNS = ["trajectory", "sampling_args", "trajectory_id"]


# ===========================================================================
# Section 1 — MARScore schema invariants (construction-time)
# ===========================================================================


def test_marscore_rejects_duplicate_member_ids():
    with pytest.raises(ValueError, match="Duplicate member_id"):
        MARScore(
            members=[
                MemberScore(member_id="A", reward=1.0),
                MemberScore(member_id="A", reward=0.0),
            ],
            episode_scalar=0.5,
        )


def test_marscore_rejects_empty_members():
    with pytest.raises(ValueError, match="cannot be empty"):
        MARScore(members=[], episode_scalar=0.0)


def test_marscore_validation_runs_on_model_validate_too():
    """Schema invariants must hold regardless of construction path."""
    payload = {
        "members": [
            {"member_id": "X", "reward": 1.0},
            {"member_id": "X", "reward": 0.0},
        ],
        "episode_scalar": 0.5,
    }
    with pytest.raises(Exception, match="Duplicate member_id"):
        MARScore.model_validate(payload)


# ===========================================================================
# Section 2 — Round-trip: state → state_to_output → JSON → bridge → MemberRollouts
# ===========================================================================


def _build_clean_state(*, episode_scalar=1.0, parse_errors=None) -> State:
    parse_errors = parse_errors or {}
    state = State()
    state["example_id"] = 7
    state["task"] = "default"
    state["trajectory_id"] = "ep-7"
    state["sampling_args"] = {"temperature": 0.5}
    state["timing"] = {}
    extras_a = {"member_id": "A", "phase": "p"}
    extras_b = {"member_id": "B", "phase": "p"}
    if parse_errors.get("A"):
        extras_a["parse_error"] = parse_errors["A"]
    if parse_errors.get("B"):
        extras_b["parse_error"] = parse_errors["B"]
    state["trajectory"] = [
        TrajectoryStep(
            prompt=[],
            completion=[],
            response={
                "id": "x",
                "created": 0,
                "model": "m",
                "message": {"role": "assistant", "content": "a-says", "finish_reason": "stop", "is_truncated": False},
            },
            tokens=None,
            reward=None,
            advantage=None,
            is_truncated=False,
            trajectory_id="ep-7",
            extras=extras_a,
        ),
        TrajectoryStep(
            prompt=[],
            completion=[],
            response={
                "id": "y",
                "created": 0,
                "model": "m",
                "message": {"role": "assistant", "content": "b-says", "finish_reason": "stop", "is_truncated": False},
            },
            tokens=None,
            reward=None,
            advantage=None,
            is_truncated=False,
            trajectory_id="ep-7",
            extras=extras_b,
        ),
    ]
    state["mar_score"] = MARScore(
        members=[
            MemberScore(
                member_id="A",
                reward=1.0,
                parse_error_count=1 if parse_errors.get("A") else 0,
                metrics={"accuracy": 1.0},
            ),
            MemberScore(
                member_id="B",
                reward=0.0,
                parse_error_count=1 if parse_errors.get("B") else 0,
                metrics={"accuracy": 0.0},
            ),
        ],
        episode_scalar=episode_scalar,
        episode_metrics={"agreement": 0.0, "winner": 0.0},
    )
    return state


def _wire_round_trip(state: State) -> dict:
    output = state_to_output(state, state_columns=REQUIRED_COLUMNS)
    serialized = json.dumps(output, default=lambda o: o.model_dump(exclude_none=True))
    return json.loads(serialized)


def test_round_trip_preserves_per_member_rewards():
    state = _build_clean_state()
    output = _wire_round_trip(state)
    rollouts = rollout_to_member_rollouts(output)
    rewards = {r["member_id"]: r["reward"] for r in rollouts}
    assert rewards == {"A": 1.0, "B": 0.0}


def test_round_trip_preserves_episode_scalar_via_legacy_reward_key():
    state = _build_clean_state(episode_scalar=0.7)
    output = _wire_round_trip(state)
    assert output["reward"] == 0.7  # legacy GRPO consumer path


def test_round_trip_preserves_per_member_metrics_via_flat_top_level_keys():
    """wandb consumer (orchestrator.py:679) reads top-level flat keys."""
    state = _build_clean_state()
    output = _wire_round_trip(state)
    assert output["reward/A"] == 1.0
    assert output["reward/B"] == 0.0
    assert output["accuracy/A"] == 1.0
    assert output["accuracy/B"] == 0.0
    assert output["agreement"] == 0.0
    assert output["winner"] == 0.0


def test_round_trip_preserves_member_id_assignment():
    """After the α-cut, member_id is the single label; round-trip MUST keep
    each step routed to the correct member's MemberRollout."""
    state = _build_clean_state()
    output = _wire_round_trip(state)
    rollouts = rollout_to_member_rollouts(output)
    by_member = {r["member_id"]: [s["extras"]["member_id"] for s in r["trajectory"]] for r in rollouts}
    assert by_member == {"A": ["A"], "B": ["B"]}


def test_round_trip_preserves_episode_id_from_trajectory_id():
    state = _build_clean_state()
    output = _wire_round_trip(state)
    rollouts = rollout_to_member_rollouts(output)
    assert all(r["episode_id"] == "ep-7" for r in rollouts)


def test_round_trip_preserves_temporal_step_order_per_member():
    state = _build_clean_state()
    # Add 2 more steps interleaved so each member has 2 steps.
    state["trajectory"].extend(
        [
            TrajectoryStep(
                prompt=[],
                completion=[],
                response={
                    "id": "x2",
                    "created": 0,
                    "model": "m",
                    "message": {
                        "role": "assistant",
                        "content": "a-rebut",
                        "finish_reason": "stop",
                        "is_truncated": False,
                    },
                },
                tokens=None,
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="ep-7",
                extras={"member_id": "A", "phase": "rebuttal"},
            ),
            TrajectoryStep(
                prompt=[],
                completion=[],
                response={
                    "id": "y2",
                    "created": 0,
                    "model": "m",
                    "message": {
                        "role": "assistant",
                        "content": "b-rebut",
                        "finish_reason": "stop",
                        "is_truncated": False,
                    },
                },
                tokens=None,
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="ep-7",
                extras={"member_id": "B", "phase": "rebuttal"},
            ),
        ]
    )
    output = _wire_round_trip(state)
    rollouts = rollout_to_member_rollouts(output)
    a = next(r for r in rollouts if r["member_id"] == "A")
    assert [s["extras"]["phase"] for s in a["trajectory"]] == ["p", "rebuttal"]


def test_str_example_id_round_trips_unmodified():
    state = _build_clean_state()
    state["example_id"] = "huggingface_uid_42"
    output = _wire_round_trip(state)
    rollouts = rollout_to_member_rollouts(output)
    assert all(r["example_id"] == "huggingface_uid_42" for r in rollouts)


# ===========================================================================
# Section 3 — Single-agent envs unaffected (no mar_score => legacy path)
# ===========================================================================


def test_single_agent_env_legacy_metrics_path_intact():
    """Rubric writes state["metrics"] directly (no mar_score). Output flatten
    behavior preserved for backwards compat with single-agent envs."""
    state = State()
    state["example_id"] = 1
    state["task"] = "default"
    state["reward"] = 0.42
    state["metrics"] = {"my_metric": 1.0, "other": 2.0}
    state["timing"] = {}
    output = state_to_output(state)
    assert output["reward"] == 0.42
    assert output["my_metric"] == 1.0
    assert output["other"] == 2.0
    # mar_score absent → bridge correctly raises (single-agent envs don't bridge).
    assert "mar_score" not in output


def test_single_agent_bridge_call_raises_keyerror():
    state = State()
    state["example_id"] = 1
    state["task"] = "default"
    state["reward"] = 0.42
    state["metrics"] = {}
    state["timing"] = {}
    output = state_to_output(state, state_columns=["sampling_args"])
    state["sampling_args"] = {"temperature": 0.5}
    output = state_to_output(state, state_columns=["sampling_args"])
    with pytest.raises(KeyError, match="mar_score"):
        rollout_to_member_rollouts(output)


# ===========================================================================
# Section 4 — Bridge accepts BOTH MARScore object and dict (post-JSON)
# ===========================================================================


def test_bridge_accepts_marscore_object_in_memory():
    """In-memory: state_to_output puts dict, but pre-serialization MARScore stays."""
    state = _build_clean_state()
    output = state_to_output(state, state_columns=REQUIRED_COLUMNS)
    # state_to_output dumps to dict via model_dump.
    assert isinstance(output["mar_score"], dict)
    rollouts = rollout_to_member_rollouts(output)
    assert len(rollouts) == 2


def test_bridge_accepts_dict_after_json_round_trip():
    state = _build_clean_state()
    output = _wire_round_trip(state)
    assert isinstance(output["mar_score"], dict)
    rollouts = rollout_to_member_rollouts(output)
    assert len(rollouts) == 2


def test_bridge_accepts_mar_score_object_directly():
    """If a caller passes an output with MARScore object directly (e.g. tests),
    bridge handles it."""
    state = _build_clean_state()
    output = state_to_output(state, state_columns=REQUIRED_COLUMNS)
    output["mar_score"] = state["mar_score"]  # replace dict with object
    rollouts = rollout_to_member_rollouts(output)
    assert len(rollouts) == 2


# ===========================================================================
# Section 5 — P0-1: ExceptionGroup mixed-error handling
# ===========================================================================


def test_simultaneous_slot_mixed_vf_errors_raises_single_concrete_exception():
    """P0-1 regression: when agents raise different vf.Error subclasses,
    rollout loop must catch a SINGLE exception (not ExceptionGroup)."""

    class _MixedFailEnv(MultiAgentEnv):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.dataset = lambda: None

        async def get_model_response(
            self,
            state,
            prompt,
            *,
            client=None,
            model=None,
            request_context=None,
        ):
            mid = request_context.member_id if request_context else None
            if mid == "A":
                raise vf.OverlongPromptError("A: too long")
            if mid == "B":
                raise vf.InvalidModelResponseError("B: bad response")
            raise vf.Error("unknown agent")

        async def build_prompt(self, state, member_id, slot):
            return [{"role": "user", "content": "hi"}]

        async def render_completion(self, state):
            state["completion"] = []

    env = _MixedFailEnv(
        schedule=StaticSchedule((TurnSlot(slot_id=0, agents=("A", "B"), phase="p"),)),
        members=["A", "B"],
        dataset=lambda: None,
    )
    state = State()
    state["_kernel"] = KernelState(slot_index=0)
    state["trajectory"] = []
    state["trajectory_id"] = "ep-0"
    state["sampling_args"] = {"temperature": 0.5}

    slot = TurnSlot(slot_id=0, agents=("A", "B"), phase="p")
    # Calling _run_simultaneous_slot directly to exercise the except path.
    # Single, concrete exception expected (NOT ExceptionGroup).
    with pytest.raises(vf.Error) as ei:
        asyncio.run(env._run_simultaneous_slot(state, slot))
    assert not isinstance(ei.value, BaseExceptionGroup), (
        f"P0-1 REGRESSION: ExceptionGroup escaped, got {type(ei.value).__name__}"
    )


def test_simultaneous_slot_overlong_takes_priority_over_other_vf_errors():
    """When mixed errors include OverlongPromptError, that's chosen so the
    rollout loop's prompt_too_long path runs."""

    class _MixedFailEnv(MultiAgentEnv):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.dataset = lambda: None

        async def get_model_response(
            self,
            state,
            prompt,
            *,
            client=None,
            model=None,
            request_context=None,
        ):
            mid = request_context.member_id if request_context else None
            if mid == "A":
                raise vf.InvalidModelResponseError("A first")
            if mid == "B":
                raise vf.OverlongPromptError("B overlong")
            raise vf.Error("unknown")

        async def build_prompt(self, state, member_id, slot):
            return [{"role": "user", "content": "hi"}]

        async def render_completion(self, state):
            state["completion"] = []

    env = _MixedFailEnv(
        schedule=StaticSchedule((TurnSlot(slot_id=0, agents=("A", "B"), phase="p"),)),
        members=["A", "B"],
        dataset=lambda: None,
    )
    state = State()
    state["_kernel"] = KernelState(slot_index=0)
    state["trajectory"] = []
    state["trajectory_id"] = "ep-0"
    state["sampling_args"] = {"temperature": 0.5}

    slot = TurnSlot(slot_id=0, agents=("A", "B"), phase="p")
    with pytest.raises(vf.OverlongPromptError):
        asyncio.run(env._run_simultaneous_slot(state, slot))


# ===========================================================================
# Section 6 — P0-2: Quarantined steps must be propagated for masking
# ===========================================================================


def test_quarantined_step_carries_parse_error_in_extras():
    """Kernel propagates protocol parse errors so the trainer can mask
    completion tokens."""
    schedule = StaticSchedule((TurnSlot(slot_id=0, agents=("A",), phase="p"),))
    state = KernelState(slot_index=0)
    result = apply_action(
        state,
        schedule,
        "A",
        "<think>unclosed",
        token_count=5,
        channels=ContentChannels(public="", parse_error="unclosed think tag"),
    )
    utt = result.committed[0]
    assert utt.parse_error is not None
    assert utt.public_channel == ""


def test_member_snapshot_tallies_parse_errors():
    """Parse-error counting now lives in ``member_snapshot``: per-member step
    walk that records ``extras['parse_error']`` occurrences. The trainer
    consumes the count through ``MemberScore.parse_error_count``."""
    from verifiers.protocols.debate.prompts import resolve_prompts
    from verifiers.protocols.debate.rubric import member_snapshot

    prompts = resolve_prompts("default")

    def _step(member_id: str, parse_error: str | None = None) -> TrajectoryStep:
        extras: dict[str, Any] = {"member_id": member_id, "phase": "propose"}
        if parse_error is not None:
            extras["parse_error"] = parse_error
        return TrajectoryStep(
            prompt=[],
            completion=[],
            response={
                "id": "x",
                "created": 0,
                "model": "m",
                "message": {"role": "assistant", "content": "", "finish_reason": "stop", "is_truncated": False},
            },
            tokens=None,
            reward=None,
            advantage=None,
            is_truncated=False,
            trajectory_id="t",
            extras=extras,
        )

    a_steps = [_step("A", "unbalanced"), _step("A")]
    b_steps = [_step("B", "stray closer")]
    snap_a = member_snapshot("debater_a", a_steps, prompts)
    snap_b = member_snapshot("debater_b", b_steps, prompts)
    assert snap_a["parse_errors"] == 1
    assert snap_b["parse_errors"] == 1


def test_marscore_projects_parse_error_count_to_wandb_when_nonzero():
    state = _build_clean_state(parse_errors={"A": "unbalanced"})
    output = _wire_round_trip(state)
    assert output["parse_errors/A"] == 1
    assert "parse_errors/B" not in output  # B had no parse errors


# ===========================================================================
# Section 7 — P0-4: UsageTracker fork/merge isolation
# ===========================================================================


def test_usage_tracker_fork_returns_zero_initialized_child():
    parent = StateUsageTracker()
    parent.increment(input_tokens=100, output_tokens=50)
    child = parent.fork()
    assert child.usage["input_tokens"] == 0.0
    assert child.usage["output_tokens"] == 0.0
    assert child.snapshot() is None  # _usage_seen is False


def test_usage_tracker_merge_accumulates():
    parent = StateUsageTracker()
    parent.increment(input_tokens=10, output_tokens=5)
    child1 = parent.fork()
    child1.increment(input_tokens=20, output_tokens=10)
    child2 = parent.fork()
    child2.increment(input_tokens=30, output_tokens=15)
    parent.merge(child1)
    parent.merge(child2)
    assert parent.usage["input_tokens"] == 60.0
    assert parent.usage["output_tokens"] == 30.0


def test_usage_tracker_failed_branch_drop_preserves_parent_snapshot():
    """The whole point of fork/merge: a failed branch's usage doesn't leak.
    If a slot fails, its forks are dropped (no merge), parent stays put."""
    parent = StateUsageTracker()
    parent.increment(input_tokens=100, output_tokens=50)
    snapshot_before = dict(parent.usage)

    # Simulated failed slot: fork + accumulate, then DROP without merging.
    failed_child = parent.fork()
    failed_child.increment(input_tokens=999, output_tokens=999)
    del failed_child  # representing a dropped branch on slot failure

    assert dict(parent.usage) == snapshot_before


def test_usage_tracker_fork_merge_preserves_seen_flag():
    parent = StateUsageTracker()  # _usage_seen=False initially
    child = parent.fork()
    child.increment(input_tokens=5, output_tokens=2, mark_seen=True)
    parent.merge(child)
    snap = parent.snapshot()
    assert snap is not None
    assert snap["input_tokens"] == 5.0
    assert snap["output_tokens"] == 2.0


# ===========================================================================
# Section 8 — Errored rollouts: contract preserved end-to-end
# ===========================================================================


def test_errored_rollout_marscore_carries_zero_rewards_for_all_members():
    """The base MultiAgentRubric writes a zero-reward MARScore on vf.Error so
    bridge can still produce per-member rollouts (no KeyError)."""
    from verifiers.rubrics.multi_agent_rubric import MultiAgentRubric

    class _AlwaysFail(MultiAgentRubric):
        members = ["A", "B"]

        async def build_marscore(self, state):
            raise vf.InfraError("boom")

    rubric = _AlwaysFail()
    state = State()
    asyncio.run(rubric.score_group([state]))
    assert isinstance(state.get("error"), vf.InfraError)
    mar = state["mar_score"]
    assert {m.member_id: m.reward for m in mar.members} == {"A": 0.0, "B": 0.0}
    assert mar.episode_metrics["errored_rollout"] == 1.0


def test_errored_rollout_round_trips_correctly():
    state = _build_clean_state()
    state["mar_score"] = MARScore(
        members=[
            MemberScore(member_id="A", reward=0.0),
            MemberScore(member_id="B", reward=0.0),
        ],
        episode_scalar=0.0,
        episode_metrics={"errored_rollout": 1.0},
        episode_error={
            "error_type": "InvalidModelResponseError",
            "error_phase": "scoring",
        },
    )
    output = _wire_round_trip(state)
    # Averageable scalars project to top-level (legacy GRPO/wandb shape).
    assert output["reward"] == 0.0
    assert output["errored_rollout"] == 1.0
    # error metadata stays nested under mar_score (categorical, not averageable).
    assert output["mar_score"]["episode_error"] == {
        "error_type": "InvalidModelResponseError",
        "error_phase": "scoring",
    }
    # Bridge still produces per-member rollouts.
    rollouts = rollout_to_member_rollouts(output)
    assert len(rollouts) == 2
    assert all(r["reward"] == 0.0 for r in rollouts)


# ===========================================================================
# Section 9 — Schema tightening: bridge fails loud on malformed inputs
# ===========================================================================


def test_bridge_raises_on_step_missing_member_id():
    state = _build_clean_state()
    state["trajectory"][0]["extras"] = {}  # strip member_id
    output = _wire_round_trip(state)
    with pytest.raises(ValueError, match="member_id"):
        rollout_to_member_rollouts(output)


def test_bridge_defaults_temperature_when_sampling_args_omits_it():
    """``state_to_output`` always projects ``sampling_args`` (defaulting to
    ``{}``); the bridge in turn defaults ``temperature`` to OpenAI's
    canonical ``1.0`` so vf-eval flows that don't pin temperature still
    produce trainable MemberRollouts."""
    state = _build_clean_state()
    state["sampling_args"] = {}  # explicit empty — no temperature
    output = state_to_output(state, state_columns=REQUIRED_COLUMNS)
    rollouts = rollout_to_member_rollouts(output)
    assert all(r["sampling_args"]["temperature"] == 1.0 for r in rollouts)


def test_bridge_raises_on_missing_mar_score():
    state = _build_clean_state()
    del state["mar_score"]
    output = state_to_output(state, state_columns=REQUIRED_COLUMNS)
    with pytest.raises(KeyError, match="mar_score"):
        rollout_to_member_rollouts(output)


# ===========================================================================
# Section 10 — to_metrics_flat projection invariants
# ===========================================================================


def test_to_metrics_flat_idempotent_on_repeated_calls():
    mar = MARScore(
        members=[MemberScore(member_id="A", reward=1.0, metrics={"x": 1.0})],
        episode_scalar=1.0,
        episode_metrics={"agreement": 1.0},
    )
    a = mar.to_metrics_flat()
    b = mar.to_metrics_flat()
    assert a == b
    a["mutated"] = True
    assert "mutated" not in mar.to_metrics_flat()  # fresh dict each time


def test_to_metrics_flat_episode_metrics_not_clobbered_by_member_metrics():
    """Episode-level keys MUST NOT be overwritten by per-member projections."""
    mar = MARScore(
        members=[MemberScore(member_id="A", reward=1.0, metrics={"agreement": 999.0})],  # collides
        episode_scalar=1.0,
        episode_metrics={"agreement": 0.5},
    )
    flat = mar.to_metrics_flat()
    # Per-member projects to "agreement/A", episode keeps "agreement"
    assert flat["agreement"] == 0.5
    assert flat["agreement/A"] == 999.0


def test_member_rewards_are_canonical_keys():
    """Bridge readers that prefer the structured payload should not need to
    parse "reward/X" from the flat projection — by_id() is the canonical
    accessor."""
    mar = MARScore(
        members=[
            MemberScore(member_id="A", reward=0.7),
            MemberScore(member_id="B", reward=0.3),
        ],
        episode_scalar=0.5,
    )
    by = mar.by_id()
    assert by["A"].reward == 0.7
    assert by["B"].reward == 0.3
