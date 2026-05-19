"""Regression tests for three MAEnv findings on Phase 5 round-2.

F1. Consecutive user-role messages are folded at the MAEnv rollout
    boundary so the token-stitch tail is _is_valid_env_tail-accepted and
    the chat template sees alternating role boundaries.
F2. DebateEnv.build_prompt derives per-utterance round_index from
    positional ordering in the transcript, not from slot_id arithmetic.
    Sparse/semantic slot_ids no longer produce nonsensical labels.
F3. DebatePrompts._validate rejects per-turn variables in system and
    question templates at load time (they'd break the monotonic prefix
    invariant silently if rendered per slot).
"""

from __future__ import annotations

import asyncio
from types import MappingProxyType

import pytest
from verifiers.clients.openai_chat_completions_token_client import _is_valid_env_tail
from verifiers.envs.multi_agent_kernel import (
    KernelState,
    StaticSchedule,
    TurnSlot,
    Utterance,
)
from verifiers.protocols.debate.env import DebateEnv
from verifiers.protocols.debate.prompts import _validate, resolve_prompts
from verifiers.protocols.debate.rubric import DebateRubric
from verifiers.types import AssistantMessage, SystemMessage, ToolMessage, UserMessage
from verifiers.utils.message_utils import fold_consecutive_user_messages

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utt(mid: str, turn_index: int, sid: int, phase: str, raw: str, pub: str) -> Utterance:
    return Utterance(
        member_id=mid,
        turn_index=turn_index,
        slot_id=sid,
        phase=phase,
        raw_content=raw,
        public_channel=pub,
        private_channel=None,
        token_count=len(raw),
    )


def _make_env(schedule: StaticSchedule) -> DebateEnv:
    prompts = resolve_prompts("selfplay")
    members = ["debater_a", "debater_b"]
    rubric = DebateRubric(truth_member="debater_a", members=members, prompts=prompts)
    return DebateEnv(
        schedule=schedule,
        prompts=prompts,
        members=members,
        rubric=rubric,
        dataset=lambda: None,
    )


def _state(commits: list[Utterance]) -> dict:
    return {
        "_kernel": KernelState(
            slot_index=len(commits),
            transcript=tuple(commits),
            pending=MappingProxyType({}),
        ),
        "prompt": [{"role": "user", "content": "Is 2+2 = 4?"}],
        "answer": "A",
        "trajectory": [],
    }


# ---------------------------------------------------------------------------
# F1 — fold_consecutive_user_messages contract
# ---------------------------------------------------------------------------


def test_fold_idempotent_on_simple_run():
    msgs = [
        UserMessage(content="a"),
        UserMessage(content="b"),
        UserMessage(content="c"),
    ]
    once = fold_consecutive_user_messages(msgs)
    twice = fold_consecutive_user_messages(once)
    assert once == twice
    assert len(once) == 1
    assert once[0].content == "a\n\nb\n\nc"


def test_fold_noop_on_sa_tool_trajectory():
    msgs = [
        SystemMessage(content="sys"),
        UserMessage(content="q"),
        AssistantMessage(content=None, tool_calls=[{"id": "1", "name": "t", "arguments": "{}"}]),
        ToolMessage(tool_call_id="1", content="r"),
        AssistantMessage(content="done"),
    ]
    assert fold_consecutive_user_messages(msgs) == msgs


def test_fold_leaves_tool_messages_untouched():
    msgs = [
        ToolMessage(tool_call_id="a", content="r1"),
        ToolMessage(tool_call_id="b", content="r2"),
    ]
    # Adjacent tool messages MUST NOT be folded — tool_call_id uniqueness
    # is what correlates results to specific calls.
    assert fold_consecutive_user_messages(msgs) == msgs


def test_fold_skips_multimodal_content_lists():
    # Don't blindly concatenate when content is a list of parts — we'd
    # lose the image part structure. The downstream stitcher will see
    # [user, user], fall back to MITO, and the model still gets valid
    # input via the chat template. Slow-path, but correct.
    msgs = [
        UserMessage(content=[{"type": "image_url", "image_url": {"url": "x"}}]),
        UserMessage(content="describe it"),
    ]
    folded = fold_consecutive_user_messages(msgs)
    assert folded == msgs  # exact pass-through
    assert folded[0].content == [{"type": "image_url", "image_url": {"url": "x"}}]
    assert folded[1].content == "describe it"


def test_fold_preserves_other_fields_on_merged_user():
    # CustomBaseModel has extra="allow", so OpenAI-style `name` rides along.
    msgs = [
        UserMessage(content="hi", name="alice"),
        UserMessage(content="again"),
    ]
    folded = fold_consecutive_user_messages(msgs)
    assert len(folded) == 1
    # `name` from the first message carries over via model_copy; we don't
    # merge conflicting metadata.
    assert getattr(folded[0], "name", None) == "alice"
    assert folded[0].content == "hi\n\nagain"


# ---------------------------------------------------------------------------
# F1 — end-to-end: MAEnv rollout prompts are stitcher-friendly
# ---------------------------------------------------------------------------


async def _run_roundtrip():
    """Render at slot 0, then at slot 2 after commits — verify stitcher tail."""
    schedule = StaticSchedule(
        slots=(
            TurnSlot(slot_id=0, agents=("debater_a",), phase="propose"),
            TurnSlot(slot_id=1, agents=("debater_b",), phase="propose"),
            TurnSlot(slot_id=2, agents=("debater_a",), phase="critique"),
            TurnSlot(slot_id=3, agents=("debater_b",), phase="critique"),
        )
    )
    env = _make_env(schedule)
    slots = env.schedule._slots

    msgs0 = await env.build_prompt(_state([]), "debater_a", slots[0])
    msgs2 = await env.build_prompt(
        _state(
            [
                _utt("debater_a", 0, 0, "propose", "A1 raw", "A1 pub"),
                _utt("debater_b", 1, 1, "propose", "B1 raw", "B1 pub"),
            ]
        ),
        "debater_a",
        slots[2],
    )

    f0 = fold_consecutive_user_messages(msgs0)
    f2 = fold_consecutive_user_messages(msgs2)

    # Post-commit form of slot 0 = folded msgs0 + A1 as assistant msg.
    cache_after_0 = list(f0) + [AssistantMessage(content="A1 raw")]
    tail = f2[len(cache_after_0) :]

    # Prefix stability
    assert f2[: len(cache_after_0)] == cache_after_0
    # Tail is exactly one user message
    assert [m.role for m in tail] == ["user"]
    # Stitcher accepts it (_get_role reads attr or key, so typed is fine)
    assert _is_valid_env_tail(tail)


def test_folded_rollout_produces_stitcher_friendly_tail():
    asyncio.run(_run_roundtrip())


# ---------------------------------------------------------------------------
# F2 — positional round_index, not slot_id arithmetic
# ---------------------------------------------------------------------------


async def _past_instruction_positional(slot_ids: tuple[int, int, int, int]):
    """Return the past-instruction text rendered into msgs2 for member A."""
    schedule = StaticSchedule(
        slots=(
            TurnSlot(slot_id=slot_ids[0], agents=("debater_a",), phase="propose"),
            TurnSlot(slot_id=slot_ids[1], agents=("debater_b",), phase="propose"),
            TurnSlot(slot_id=slot_ids[2], agents=("debater_a",), phase="critique"),
            TurnSlot(slot_id=slot_ids[3], agents=("debater_b",), phase="critique"),
        )
    )
    env = _make_env(schedule)
    commits = [
        _utt("debater_a", 0, slot_ids[0], "propose", "A1 raw", "A1 pub"),
        _utt("debater_b", 1, slot_ids[1], "propose", "B1 raw", "B1 pub"),
    ]
    msgs = await env.build_prompt(_state(commits), "debater_a", env.schedule._slots[2])
    # Past-own-turn instruction sits before the assistant msg with A1's raw content.
    past_user_texts = []
    for i, m in enumerate(msgs):
        if m["role"] == "assistant" and m.get("content") == "A1 raw" and i > 0 and msgs[i - 1]["role"] == "user":
            past_user_texts.append(msgs[i - 1]["content"])
    return past_user_texts


async def _render_own_ctx(schedule: StaticSchedule, commits: list[Utterance], current_slot_idx: int):
    env = _make_env(schedule)
    msgs = await env.build_prompt(_state(commits), "debater_a", env.schedule._slots[current_slot_idx])
    return msgs, env._member_round_count("debater_a")


def test_num_rounds_is_per_member_under_simultaneous_schedule():
    # [AB propose, AB critique]: 2 slots, each with both agents.
    # Global arithmetic: num_slots=2, num_agents=2, num_rounds = 2//2 = 1  → WRONG.
    # Per-member: A participates in 2 slots → num_rounds = 2. Correct.
    schedule = StaticSchedule(
        slots=(
            TurnSlot(slot_id=0, agents=("debater_a", "debater_b"), phase="propose"),
            TurnSlot(slot_id=1, agents=("debater_a", "debater_b"), phase="critique"),
        )
    )
    _, num_rounds = asyncio.run(_render_own_ctx(schedule, [], 0))
    assert num_rounds == 2, f"simultaneous schedule [AB, AB]: expected 2 rounds per member, got {num_rounds}"


def test_num_rounds_is_per_member_under_asymmetric_schedule():
    # A participates in 3 slots, B in 2. Global arithmetic would give
    # 5//2 = 2 for both. Per-member: A=3, B=2.
    # All phases must be present in the selfplay pack (DebateEnv's
    # init-time coverage check) — propose / critique are; "closing"
    # would not be, so slot 4 reuses "critique" as a third A turn.
    schedule = StaticSchedule(
        slots=(
            TurnSlot(slot_id=0, agents=("debater_a",), phase="propose"),
            TurnSlot(slot_id=1, agents=("debater_b",), phase="propose"),
            TurnSlot(slot_id=2, agents=("debater_a",), phase="critique"),
            TurnSlot(slot_id=3, agents=("debater_b",), phase="critique"),
            TurnSlot(slot_id=4, agents=("debater_a",), phase="critique"),
        )
    )
    env = _make_env(schedule)
    assert env._member_round_count("debater_a") == 3
    assert env._member_round_count("debater_b") == 2


def test_sparse_slot_ids_do_not_corrupt_past_round_label():
    # Contiguous slot_ids (0..3): baseline
    contiguous = asyncio.run(_past_instruction_positional((0, 1, 2, 3)))
    # Sparse slot_ids (10, 20, 30, 40): positional logic should give SAME
    # past-instruction text as contiguous (both = round 0, phase=propose).
    sparse = asyncio.run(_past_instruction_positional((10, 20, 30, 40)))
    assert contiguous == sparse, (
        f"slot_id arithmetic leaked into instruction rendering:\n  contiguous: {contiguous}\n  sparse    : {sparse}"
    )


# ---------------------------------------------------------------------------
# F3 — pack validator rejects per-turn vars in system/question
# ---------------------------------------------------------------------------


def test_validate_rejects_round_index_in_system():
    pack = {
        "version": 2,
        "system": {
            "debater_a": "Round {{ round_index }}: you are the pro debater.",
            "debater_b": "You are debater_b.",
            "judge": "You are the judge.",
        },
        "question": {
            "debater_a": "{{ task_prompt }}",
            "debater_b": "{{ task_prompt }}",
        },
        "user": {"debater_a": {"propose": "go"}, "debater_b": {"propose": "go"}},
        "fields": {},
    }
    with pytest.raises(ValueError, match="system.debater_a.*per-turn variable"):
        _validate(pack)


def test_validate_rejects_phase_in_question():
    pack = {
        "version": 2,
        "system": {
            "debater_a": "You are debater_a.",
            "debater_b": "You are debater_b.",
            "judge": "You are the judge.",
        },
        "question": {
            "debater_a": "[{{ phase }}] {{ task_prompt }}",
            "debater_b": "{{ task_prompt }}",
        },
        "user": {"debater_a": {"propose": "go"}, "debater_b": {"propose": "go"}},
        "fields": {},
    }
    with pytest.raises(ValueError, match="question.debater_a.*per-turn variable"):
        _validate(pack)


def test_validate_rejects_is_first_round_in_system():
    # Bypass case gatekeeper flagged: is_first_round is in build_context
    # (prompts.py:245) but was missing from the original _PER_TURN_VARS list.
    pack = {
        "version": 2,
        "system": {
            "debater_a": "{% if is_first_round %}First turn.{% endif %} You are debater_a.",
            "debater_b": "You are debater_b.",
            "judge": "You are the judge.",
        },
        "question": {
            "debater_a": "{{ task_prompt }}",
            "debater_b": "{{ task_prompt }}",
        },
        "user": {"debater_a": {"propose": "go"}, "debater_b": {"propose": "go"}},
        "fields": {},
    }
    with pytest.raises(ValueError, match="system.debater_a.*per-turn variable"):
        _validate(pack)


def test_validate_rejects_statement_tag_bypass():
    # {% if round_index > 0 %}X{% endif %} evades the old regex which only
    # matched {{ ... }} expression tags. AST-based check catches it.
    pack = {
        "version": 2,
        "system": {
            "debater_a": "{% if round_index > 0 %}Round N.{% endif %} You are debater_a.",
            "debater_b": "You are debater_b.",
            "judge": "You are the judge.",
        },
        "question": {
            "debater_a": "{{ task_prompt }}",
            "debater_b": "{{ task_prompt }}",
        },
        "user": {"debater_a": {"propose": "go"}, "debater_b": {"propose": "go"}},
        "fields": {},
    }
    with pytest.raises(ValueError, match="system.debater_a.*per-turn variable"):
        _validate(pack)


def test_validate_rejects_index_access_bypass():
    # {{ data[round_index] }} — indirect index access evaded old regex.
    pack = {
        "version": 2,
        "system": {
            "debater_a": "Hint: {{ hints[round_index] }}. You are debater_a.",
            "debater_b": "You are debater_b.",
            "judge": "You are the judge.",
        },
        "question": {
            "debater_a": "{{ task_prompt }}",
            "debater_b": "{{ task_prompt }}",
        },
        "user": {"debater_a": {"propose": "go"}, "debater_b": {"propose": "go"}},
        "fields": {},
    }
    with pytest.raises(ValueError, match="system.debater_a.*per-turn variable"):
        _validate(pack)


def test_validate_rejects_set_directive_bypass():
    # {% set r = round_index %} — evaded regex, AST catches it.
    pack = {
        "version": 2,
        "system": {
            "debater_a": "{% set r = round_index %}Welcome.",
            "debater_b": "You are debater_b.",
            "judge": "You are the judge.",
        },
        "question": {
            "debater_a": "{{ task_prompt }}",
            "debater_b": "{{ task_prompt }}",
        },
        "user": {"debater_a": {"propose": "go"}, "debater_b": {"propose": "go"}},
        "fields": {},
    }
    with pytest.raises(ValueError, match="system.debater_a.*per-turn variable"):
        _validate(pack)


def test_validate_accepts_turn_invariant_templates():
    pack = {
        "version": 2,
        "system": {
            "debater_a": "You are {{ viewer_role }}.",
            "debater_b": "You are {{ viewer_role }}.",
            "judge": "You are the judge.",
        },
        "question": {
            "debater_a": "{{ task_prompt }}",
            "debater_b": "{{ task_prompt }}",
        },
        "user": {
            "debater_a": {"propose": "Round {{ round_index }}: go"},
            "debater_b": {"propose": "Round {{ round_index }}: go"},
        },
        "fields": {},
    }
    # Should not raise — user block is allowed to reference per-turn vars.
    _validate(pack)
