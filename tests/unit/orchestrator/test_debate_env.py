"""DebateEnv integration tests through the orchestrator's lens.

Everything inside the boundary is real: StaticSchedule, the multi-agent
kernel, the selfplay prompt pack, DebateRubric, state_to_output and the
member-rollout bridge. Only the Client is faked — it is the one legitimate
system boundary.

Scope discipline: verifiers tests its own kernel/rubric/renderer internals
in-repo (tests/test_multi_agent_kernel.py, test_debate_rubric_*.py,
test_debate_renderer_first.py, test_debate_env_prompts_coverage.py). This
file pins only the contracts prime-rl depends on:

  * trajectory tagging + token survival (the bridge's split_by_member input)
  * prompt construction: stitcher-friendly monotonic prefixes, opponent
    attribution, reasoning air-gap under the default pack
  * field extraction into ``extras['fields']`` (rubric + metrics input)
  * init-time cross-checks for the schedules prime-rl configures
  * flat metric keys consumed by ``prime_rl/metrics/debate.py``
  * the full rollout → score → state_to_output → member-rollout pipeline
  * the ``load_environment`` factory surface research-environments call
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest
import verifiers as vf
from verifiers import rollout_to_member_rollouts
from verifiers.clients import Client as _VFClient
from verifiers.envs.multi_agent_kernel import KernelState, StaticSchedule, TurnSlot
from verifiers.protocols.debate.env import DebateEnv, load_environment
from verifiers.protocols.debate.prompts import resolve_prompts
from verifiers.protocols.debate.rubric import DebateRubric
from verifiers.rubrics.judge_rubric import JudgeRubric
from verifiers.types import (
    Response,
    ResponseMessage,
    ResponseTokens,
    RolloutInput,
    State,
    Usage,
)
from verifiers.utils.save_utils import state_to_output

# ---------------------------------------------------------------------------
# Fake client — system boundary stub
# ---------------------------------------------------------------------------


class FakeClient(_VFClient):
    """Concrete vf.Client returning canned responses, recording every call."""

    def __init__(self, responses: list[Response]) -> None:
        self.logger = logging.getLogger(f"{__name__}.FakeClient")
        self._config = None
        self._client = None
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def get_response(self, prompt, model, sampling_args=None, tools=None, **kwargs) -> Response:
        self.calls.append({"prompt": prompt, "model": model})
        if not self._responses:
            raise RuntimeError("FakeClient exhausted — more calls than expected")
        return self._responses.pop(0)

    def setup_client(self, config):
        raise NotImplementedError

    async def to_native_tool(self, tool):
        raise NotImplementedError

    async def to_native_prompt(self, messages):
        raise NotImplementedError

    async def get_native_response(self, *args, **kwargs):
        raise NotImplementedError

    async def raise_from_native_response(self, response):
        raise NotImplementedError

    async def from_native_response(self, response):
        raise NotImplementedError

    async def close(self) -> None:
        return None


def _make_response(
    content: str,
    reasoning: str | None = None,
    token_ids: list[int] | None = None,
) -> Response:
    """Canned Response as the renderer client hands it to the env: reasoning
    already split into the structured ``reasoning_content`` field."""
    tokens = None
    if token_ids is not None:
        tokens = ResponseTokens(
            prompt_ids=[1, 2, 3],
            prompt_mask=[0, 0, 0],
            completion_ids=token_ids,
            completion_mask=[1] * len(token_ids),
            completion_logprobs=[-0.5] * len(token_ids),
            routed_experts=None,
        )
    return Response(
        id="fake-resp",
        created=0,
        model="test-model",
        usage=Usage(
            prompt_tokens=3,
            reasoning_tokens=0,
            completion_tokens=len(content),
            total_tokens=3 + len(content),
        ),
        message=ResponseMessage(
            content=content,
            reasoning_content=reasoning,
            finish_reason="stop",
            is_truncated=False,
            tokens=tokens,
        ),
    )


# ---------------------------------------------------------------------------
# Shared fixtures — real selfplay pack, sequential + simultaneous schedules
# ---------------------------------------------------------------------------

MEMBERS = ["debater_a", "debater_b", "judge"]
SELFPLAY = resolve_prompts("selfplay")

SEQUENTIAL_SCHEDULE = StaticSchedule(
    (
        TurnSlot(slot_id=0, agents=("debater_a",), phase="propose"),
        TurnSlot(slot_id=1, agents=("debater_b",), phase="propose"),
        TurnSlot(slot_id=2, agents=("debater_a",), phase="critique"),
        TurnSlot(slot_id=3, agents=("debater_b",), phase="critique"),
        TurnSlot(slot_id=4, agents=("judge",), phase="final"),
    )
)

SIMULTANEOUS_SCHEDULE = StaticSchedule(
    (
        TurnSlot(slot_id=0, agents=("debater_a", "debater_b"), phase="propose"),
        TurnSlot(slot_id=1, agents=("judge",), phase="final"),
    )
)

TASK_PROMPT = "What is 2+2?\n\nA) 1\nB) 3\nC) 4\nD) 5"


def _make_env(
    truth_member: str | None = None,
    schedule: StaticSchedule = SEQUENTIAL_SCHEDULE,
) -> DebateEnv:
    rubric = DebateRubric(truth_member=truth_member, members=MEMBERS, prompts=SELFPLAY)
    return DebateEnv(
        schedule=schedule,
        prompts=SELFPLAY,
        members=MEMBERS,
        rubric=rubric,
        dataset=lambda: None,
    )


def _rollout_input(example_id: int = 42) -> RolloutInput:
    return RolloutInput(
        prompt=[{"role": "user", "content": TASK_PROMPT}],
        example_id=example_id,
        task_prompt=TASK_PROMPT,
        answer="C",
    )


def _run(coro):
    return asyncio.run(coro)


def _canonical_responses() -> list[Response]:
    """Two-round debate where debater_a flips B→C (earned flip) and
    debater_b holds D; the judge picks debater_a."""
    return [
        _make_response("I propose <answer>B</answer>", reasoning="a-think-1", token_ids=[10, 11]),
        _make_response("I propose <answer>D</answer>", reasoning="b-think-1", token_ids=[20, 21]),
        _make_response("On reflection <answer>C</answer>", reasoning="a-think-2", token_ids=[12, 13, 14]),
        _make_response("I hold <answer>D</answer>", reasoning="b-think-2", token_ids=[22, 23]),
        _make_response(
            "<reasoning>A fixed the flaw</reasoning><decision>debater_a</decision>",
            token_ids=[30],
        ),
    ]


@pytest.fixture
def canonical_rollout() -> tuple[State, FakeClient]:
    """One full sequential rollout on the canonical responses (unscored)."""
    env = _make_env(truth_member="debater_a")
    client = FakeClient(_canonical_responses())
    state = _run(env.rollout(_rollout_input(), client, "test-model"))
    return state, client


@pytest.fixture
def scored_state(canonical_rollout) -> State:
    """The canonical rollout after real DebateRubric scoring (MCQ enum path,
    no LLM judge client needed)."""
    state, _ = canonical_rollout
    rubric = DebateRubric(truth_member="debater_a", members=MEMBERS, prompts=SELFPLAY)
    _run(rubric.score_rollout(state))
    return state


def _prompt_text(messages) -> str:
    return "\n".join(str(getattr(m, "content", m.get("content", "") if isinstance(m, dict) else "")) for m in messages)


# ---------------------------------------------------------------------------
# Rollout mechanics — the bridge's split_by_member input
# ---------------------------------------------------------------------------


def test_rollout_tags_every_step_with_member_id_and_phase(canonical_rollout):
    state, _ = canonical_rollout
    tags = [(s["extras"]["member_id"], s["extras"]["phase"]) for s in state["trajectory"]]
    assert tags == [
        ("debater_a", "propose"),
        ("debater_b", "propose"),
        ("debater_a", "critique"),
        ("debater_b", "critique"),
        ("judge", "final"),
    ]


def test_rollout_steps_carry_prompt_completion_and_tokens(canonical_rollout):
    state, _ = canonical_rollout
    for step in state["trajectory"]:
        assert len(step["prompt"]) > 0
        assert len(step["completion"]) > 0
        assert step["tokens"] is not None, "token data lost — trainer alignment depends on it"


def test_rollout_completes_with_schedule_exhausted(canonical_rollout):
    state, _ = canonical_rollout
    assert state["error"] is None
    assert state["is_completed"] is True
    assert state["stop_condition"] == "schedule_exhausted"


def test_kernel_transcript_carries_channel_split_content(canonical_rollout):
    """Response content flows verbatim into the public channel; the renderer's
    structured reasoning_content into the private channel."""
    state, _ = canonical_rollout
    kernel: KernelState = state["_kernel"]
    assert kernel.slot_index == 5
    assert [(u.member_id, u.phase) for u in kernel.transcript] == [
        ("debater_a", "propose"),
        ("debater_b", "propose"),
        ("debater_a", "critique"),
        ("debater_b", "critique"),
        ("judge", "final"),
    ]
    first = kernel.transcript[0]
    assert first.public_channel == "I propose <answer>B</answer>"
    assert first.private_channel == "a-think-1"
    assert first.parse_error is None


def test_render_completion_is_assistant_steps_only(canonical_rollout):
    """state['completion'] is the per-step completions flattened — one
    assistant message per slot, no per-turn prompts leaking in."""
    state, _ = canonical_rollout
    completion = state["completion"]
    assert len(completion) == 5
    assert all(getattr(m, "role", None) == "assistant" for m in completion)


def test_simultaneous_slot_commits_all_agents():
    """Selfplay-canonical schedule: one simultaneous propose slot + judge."""
    env = _make_env(truth_member="debater_a", schedule=SIMULTANEOUS_SCHEDULE)
    client = FakeClient(
        [
            _make_response("<answer>C</answer>", token_ids=[1, 2]),
            _make_response("<answer>B</answer>", token_ids=[3, 4]),
            _make_response("<decision>debater_a</decision>", token_ids=[5]),
        ]
    )
    state = _run(env.rollout(_rollout_input(), client, "test-model"))

    assert state["error"] is None
    tags = [(s["extras"]["member_id"], s["extras"]["phase"]) for s in state["trajectory"]]
    assert tags == [
        ("debater_a", "propose"),
        ("debater_b", "propose"),
        ("judge", "final"),
    ]

    _run(env.rubric.score_rollout(state))
    mar = state["mar_score"]
    assert mar.episode_scalar == 1.0
    assert {m.member_id: m.reward for m in mar.members} == {
        "debater_a": 1.0,
        "debater_b": -1.0,
        "judge": 0.0,
    }


# ---------------------------------------------------------------------------
# Prompt construction — stitching/replay through the orchestrator's lens
# ---------------------------------------------------------------------------


def test_first_prompt_contains_system_and_question(canonical_rollout):
    _, client = canonical_rollout
    text = _prompt_text(client.calls[0]["prompt"])
    assert "interactive debate" in text  # selfplay system template
    assert TASK_PROMPT in text  # question template = {{ task_prompt }}


def test_critique_prompt_attributes_opponent_and_airgaps_reasoning(canonical_rollout):
    """debater_a's critique sees debater_b's public propose, framed by the
    pack's opponent_wrap attribution — and never debater_b's reasoning
    (default think_visibility => public_only air-gap). The judge's view is
    air-gapped the same way under the default pack."""
    _, client = canonical_rollout

    a_critique = _prompt_text(client.calls[2]["prompt"])
    assert "I propose <answer>D</answer>" in a_critique
    assert "written by your opponent, not you" in a_critique  # opponent_wrap frame
    assert "debater_b" in a_critique  # speaker attribution
    assert "b-think-1" not in a_critique  # air-gap

    judge_view = _prompt_text(client.calls[4]["prompt"])
    assert "I propose <answer>B</answer>" in judge_view
    assert "I hold <answer>D</answer>" in judge_view
    for secret in ("a-think-1", "a-think-2", "b-think-1", "b-think-2"):
        assert secret not in judge_view


def test_prompts_extend_monotonically_per_member(canonical_rollout):
    """For each member, the slot-N+1 prompt must byte-extend the slot-N
    prompt: older messages identical, new tail appended. This is the prefix
    contract the stitcher and the vLLM prefix cache depend on."""
    state, _ = canonical_rollout
    per_member: dict[str, list[list[dict]]] = {}
    for step in state["trajectory"]:
        mid = step["extras"]["member_id"]
        per_member.setdefault(mid, []).append([m.model_dump(exclude_none=True) for m in step["prompt"]])
    assert set(per_member) == set(MEMBERS)
    for mid, seq in per_member.items():
        for prev, curr in zip(seq, seq[1:]):
            assert len(curr) >= len(prev), f"{mid}: prompt shrank between slots"
            for i, (p, c) in enumerate(zip(prev, curr)):
                assert p == c, f"{mid}: message {i} diverged between slots\n  prev: {p!r}\n  curr: {c!r}"


# ---------------------------------------------------------------------------
# Field extraction — extras['fields'] feeds the rubric and debate metrics
# ---------------------------------------------------------------------------


def test_extract_fields_populates_extras(canonical_rollout):
    state, _ = canonical_rollout
    fields = [s["extras"].get("fields") for s in state["trajectory"]]
    assert fields[0] == {"answer": "B"}
    assert fields[1] == {"answer": "D"}
    assert fields[2] == {"answer": "C"}
    assert fields[3] == {"answer": "D"}
    assert fields[4] == {"reasoning": "A fixed the flaw", "decision": "debater_a"}


def test_ambiguous_duplicate_field_is_per_step_failure_not_rollout_error():
    """Conflicting duplicate <answer> tags are a per-step parse failure: the
    step still lands in the trajectory with no 'fields' key, the schedule
    advances, and the rollout terminates cleanly. Terminating the rollout
    would discard the other members' valid commits."""
    env = _make_env(truth_member="debater_a")
    client = FakeClient(
        [
            _make_response("<answer>A</answer> wait no <answer>B</answer>"),
            _make_response("<answer>D</answer>"),
            _make_response("<answer>A</answer>"),
            _make_response("<answer>D</answer>"),
            _make_response("<decision>debater_b</decision>"),
        ]
    )
    state = _run(env.rollout(_rollout_input(), client, "test-model"))

    assert state["error"] is None
    assert len(state["trajectory"]) == 5
    first = state["trajectory"][0]["extras"]
    assert first["member_id"] == "debater_a"
    assert "fields" not in first  # extraction failed → no fields key
    # Peers' commits are unaffected.
    assert state["trajectory"][1]["extras"]["fields"] == {"answer": "D"}


# ---------------------------------------------------------------------------
# Init-time cross-checks — the schedule/member wiring prime-rl configures
# ---------------------------------------------------------------------------


def test_members_must_match_rubric_members_in_order():
    """env.members and rubric.members must be identical including order —
    silent drift desyncs round_index from reward attribution."""
    rubric = DebateRubric(members=MEMBERS, prompts=SELFPLAY)
    with pytest.raises(ValueError, match="members != rubric.members"):
        DebateEnv(
            schedule=SEQUENTIAL_SCHEDULE,
            prompts=SELFPLAY,
            members=["debater_b", "debater_a", "judge"],
            rubric=rubric,
            dataset=lambda: None,
        )


def test_members_must_match_static_schedule_agents():
    """For a StaticSchedule, the unique slot agents must equal the declared
    member set; dynamic SlotPrograms are exempt (data-dependent agent sets)."""
    members = ["debater_a", "debater_b"]
    rubric = DebateRubric(members=members, prompts=SELFPLAY)
    schedule_missing_b = StaticSchedule((TurnSlot(slot_id=0, agents=("debater_a",), phase="propose"),))
    with pytest.raises(ValueError, match="unique agents in StaticSchedule"):
        DebateEnv(
            schedule=schedule_missing_b,
            prompts=SELFPLAY,
            members=members,
            rubric=rubric,
            dataset=lambda: None,
        )

    class DynamicProgram:
        def current_slot(self, state):
            return None

    env = DebateEnv(
        schedule=DynamicProgram(),
        prompts=SELFPLAY,
        members=members,
        rubric=rubric,
        dataset=lambda: None,
    )
    assert env.members == members


# ---------------------------------------------------------------------------
# Scoring — the flat metric keys prime_rl/metrics/debate.py consumes
# ---------------------------------------------------------------------------


def test_scored_rollout_flat_metrics_match_train_metrics_contract(scored_state):
    """Golden contract for the canonical flipped rollout: every key consumed
    by prime_rl/metrics/debate.py (winner categorical, final/initial_correct,
    flipped, turns, agreement) plus the per-member zero-sum rewards."""
    mar = scored_state["mar_score"]
    assert mar.episode_scalar == 1.0  # truth_member declared and won
    assert mar.episode_categorical == {
        "winner": "debater_a",
        "first_answer/debater_a": "B",
        "final_answer/debater_a": "C",
        "first_answer/debater_b": "D",
        "final_answer/debater_b": "D",
    }
    assert mar.to_metrics_flat() == {
        "reward/debater_a": 1.0,
        "reward/debater_b": -1.0,
        "reward/judge": 0.0,
        "turns/debater_a": 2.0,
        "turns/debater_b": 2.0,
        "turns/judge": 1.0,
        "num_commits/debater_a": 2.0,
        "num_commits/debater_b": 2.0,
        "num_unique_commits/debater_a": 2.0,
        "num_unique_commits/debater_b": 1.0,
        "flipped/debater_a": 1.0,  # B → C: earned flip
        "flipped/debater_b": 0.0,  # held D
        "accuracy/debater_a": 1.0,
        "accuracy/debater_b": 0.0,
        "final_correct/debater_a": 1.0,
        "final_correct/debater_b": 0.0,
        "initial_correct/debater_a": 0.0,
        "initial_correct/debater_b": 0.0,
        "extraction_failed/debater_a": 0.0,
        "extraction_failed/debater_b": 0.0,
        "agreement": 0.0,  # C vs D
        "any_answer_member_correct": 1.0,
        "all_answer_members_correct": 0.0,
        "any_debater_correct": 1.0,
        "all_debaters_correct": 0.0,
        "judge_selected_correct": 1.0,
        "judge_selected_correct_given_any_correct": 1.0,
        "truth_member_correct": 1.0,
        "truth_member_won": 1.0,
    }


def test_tie_decision_zero_rewards_and_inert_scalar():
    """'tie' is a first-class judge outcome: both debaters get 0 (zero-sum
    holds), the episode scalar stays 0 even with a declared truth side, and
    the winner categorical carries 'tie' for the metrics tie_rate path."""
    env = _make_env(truth_member="debater_a")
    client = FakeClient(
        [
            _make_response("<answer>C</answer> t0"),
            _make_response("<answer>C</answer> t1"),
            _make_response("<answer>C</answer> t2"),
            _make_response("<answer>C</answer> t3"),
            _make_response("<decision>tie</decision>"),
        ]
    )
    state = _run(env.rollout(_rollout_input(), client, "test-model"))
    _run(env.rubric.score_rollout(state))

    mar = state["mar_score"]
    assert mar.episode_scalar == 0.0
    assert mar.episode_categorical["winner"] == "tie"
    assert {m.member_id: m.reward for m in mar.members} == {
        "debater_a": 0.0,
        "debater_b": 0.0,
        "judge": 0.0,
    }
    assert mar.episode_metrics["agreement"] == 1.0  # both answered C


def test_judge_declared_but_no_decision_is_errored_rollout():
    """Selfplay declares a judge; a rollout whose judge produced no parseable
    <decision> must score as an errored rollout (vf.Error on state['error'],
    zero rewards, errored_rollout=1.0) — never fall back to answer grading,
    which would install fake training signal."""
    env = _make_env(truth_member="debater_a")
    client = FakeClient(
        [
            _make_response("<answer>C</answer>"),
            _make_response("<answer>D</answer>"),
            _make_response("<answer>C</answer>"),
            _make_response("<answer>D</answer>"),
            _make_response("no decision tag here"),
        ]
    )
    state = _run(env.rollout(_rollout_input(), client, "test-model"))
    _run(env.rubric.score_rollout(state))

    err = state["error"]
    assert isinstance(err, vf.Error)
    assert "judge" in str(err).lower()
    mar = state["mar_score"]
    assert mar.episode_scalar == 0.0
    assert mar.episode_metrics == {"errored_rollout": 1.0}
    assert mar.episode_error == {"error_type": "Error", "error_phase": "scoring"}
    assert all(m.reward == 0.0 for m in mar.members)


def test_missing_answer_is_dataset_schema_violation():
    """state['answer'] (ground truth) is a hard dataset-schema requirement
    for debate scoring; its absence raises KeyError loud through score_group
    (non-vf.Error exceptions are programming/schema bugs, never swallowed)."""
    rubric = DebateRubric(members=["debater_a", "debater_b"], prompts=SELFPLAY)
    state = State()
    state["prompt"] = [{"role": "user", "content": "Q?"}]
    state["trajectory"] = []
    with pytest.raises(KeyError, match="state missing 'answer'"):
        _run(rubric.score_group([state]))


# ---------------------------------------------------------------------------
# Full pipeline: rollout → score → state_to_output → member rollouts
# ---------------------------------------------------------------------------


def test_full_pipeline_rollout_score_bridge(scored_state):
    """The critical integration path the orchestrator runs every step:
    real rollout, real scoring, real serialization boundary, real bridge."""
    output = state_to_output(
        scored_state,
        # Mirrors prime-rl's REQUIRED_STATE_COLUMNS (+ identity columns the
        # bridge requires).
        state_columns=["trajectory", "sampling_args", "trajectory_id", "task_prompt"],
    )
    assert output["example_id"] == 42
    assert isinstance(output["task"], dict)  # task payloads are JSON objects
    assert output["task"]["task_prompt"] == TASK_PROMPT
    # mar_score serialized as a plain dict for the wire.
    assert json.dumps(output["mar_score"])  # JSON-serializable
    assert output["reward"] == 1.0  # episode scalar projected to legacy key

    rollouts = rollout_to_member_rollouts(output)
    by_member = {r["member_id"]: r for r in rollouts}
    assert set(by_member) == set(MEMBERS)

    a, b, judge = by_member["debater_a"], by_member["debater_b"], by_member["judge"]
    assert (a["reward"], b["reward"], judge["reward"]) == (1.0, -1.0, 0.0)
    assert (len(a["trajectory"]), len(b["trajectory"]), len(judge["trajectory"])) == (2, 2, 1)

    for rollout in rollouts:
        assert rollout["example_id"] == 42
        assert rollout["episode_id"] == scored_state["trajectory_id"]
        assert rollout["task"] == output["task"]
        for step in rollout["trajectory"]:
            assert step["tokens"] is not None, "token data lost in the bridge"

    # Per-member steps preserve temporal order and stay member-pure.
    assert [s["extras"]["phase"] for s in a["trajectory"]] == ["propose", "critique"]
    assert all(s["extras"]["member_id"] == "debater_b" for s in b["trajectory"])


# ---------------------------------------------------------------------------
# load_environment — the factory surface research-environments call
# ---------------------------------------------------------------------------

_SCHEDULE_SLOTS = [
    {"slot_id": 0, "agents": ["debater_a", "debater_b"], "phase": "propose"},
    {"slot_id": 1, "agents": ["judge"], "phase": "final"},
]


def test_load_environment_constructs_without_judge_client():
    """Selfplay scores answer fields via classify_enum, so no LLM grader is
    required: grader/matcher stay None and the env is fully functional for
    the MCQ path. truth_member defaults to None (symmetric zero-sum)."""
    env = load_environment(
        schedule_slots=_SCHEDULE_SLOTS,
        members=MEMBERS,
        prompts_ref="selfplay",
        eval_dataset=lambda: None,
    )
    assert isinstance(env, DebateEnv)
    assert env.members == MEMBERS
    assert env.rubric.grader is None
    assert env.rubric.matcher is None
    assert env.rubric.truth_member is None


def test_load_environment_wires_judge_client_into_grader_and_matcher():
    """With a judge client, both pack-declared judge templates are compiled
    into JudgeRubric children sharing the SAME client instance."""
    client = FakeClient([])
    env = load_environment(
        schedule_slots=_SCHEDULE_SLOTS,
        members=MEMBERS,
        prompts_ref="selfplay",
        truth_member="debater_a",
        judge_client=client,
        judge_model="fake-model",
        eval_dataset=lambda: None,
    )
    assert isinstance(env.rubric.grader, JudgeRubric)
    assert isinstance(env.rubric.matcher, JudgeRubric)
    assert env.rubric.grader.judge_client is client
    assert env.rubric.matcher.judge_client is client
    assert env.rubric.truth_member == "debater_a"
