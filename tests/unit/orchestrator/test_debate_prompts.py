"""Unit tests for debate prompts: YAML loading, rendering, validation.

NO mocks. Real DebatePrompts, real Jinja rendering, real YAML files.

Run from the fork venv — see test_debate_env.py docstring for setup.
"""

from __future__ import annotations

import importlib.resources
import tempfile
import textwrap
from pathlib import Path

import pytest
from verifiers.protocols.debate.prompts import (
    DebatePrompts,
    JudgeTemplate,
    _normalize_think,
    _validate,
    build_context,
    resolve_prompts,
)

# importlib.resources is namespace-package-safe and survives non-editable
# (wheel) installs — stdlib-idiomatic way to locate package-shipped data.
_PROMPTS_DIR = Path(str(importlib.resources.files("verifiers.protocols.debate") / "prompts"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_prompts():
    resolve_prompts.cache_clear()
    return resolve_prompts(str(_PROMPTS_DIR / "default.yaml"))


@pytest.fixture
def selfplay_prompts():
    resolve_prompts.cache_clear()
    return resolve_prompts(str(_PROMPTS_DIR / "selfplay.yaml"))


def _ctx(**overrides):
    defaults = dict(
        task_prompt="What is 2+2?",
        viewer_id="debater_a",
        phase="propose",
        round_index=0,
        num_rounds=2,
    )
    defaults.update(overrides)
    return build_context(**defaults)


def _load_yaml(yaml_str: str) -> DebatePrompts:
    """Write YAML to temp file and load it."""
    resolve_prompts.cache_clear()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(textwrap.dedent(yaml_str))
        f.flush()
    return resolve_prompts(f.name)


# ===================================================================
# build_context
# ===================================================================


def test_build_context_basic():
    ctx = _ctx()
    assert ctx["task_prompt"] == "What is 2+2?"
    assert ctx["viewer_id"] == "debater_a"
    assert ctx["phase"] == "propose"
    assert ctx["is_first_round"] is True
    assert ctx["is_last_round"] is False


def test_build_context_last_round():
    ctx = _ctx(round_index=1, num_rounds=2)
    assert ctx["is_last_round"] is True


def test_build_context_has_assigned_answer():
    ctx = _ctx(answer="4")
    assert ctx["has_assigned_answer"] is True
    ctx_no = _ctx()
    assert ctx_no["has_assigned_answer"] is False


# ===================================================================
# resolve_prompts — YAML loading
# ===================================================================


def test_resolve_default_yaml(default_prompts):
    assert default_prompts.source_ref.endswith("default.yaml")
    assert "debater_a" in default_prompts.system
    assert "debater_b" in default_prompts.system
    assert "judge" in default_prompts.system


def test_resolve_selfplay_yaml(selfplay_prompts):
    assert selfplay_prompts.source_ref.endswith("selfplay.yaml")
    assert "propose" in selfplay_prompts.user["debater_a"]
    assert "critique" in selfplay_prompts.user["debater_a"]


def test_resolve_by_name():
    resolve_prompts.cache_clear()
    dp = resolve_prompts("default")
    assert dp.source_ref.endswith("default.yaml")


def test_resolve_lru_cache():
    resolve_prompts.cache_clear()
    p1 = resolve_prompts(str(_PROMPTS_DIR / "default.yaml"))
    p2 = resolve_prompts(str(_PROMPTS_DIR / "default.yaml"))
    assert p1 is p2


def test_system_is_flat(default_prompts):
    """system.<role> is a compiled template, not a nested phase dict."""
    tmpl = default_prompts.system["debater_a"]
    assert hasattr(tmpl, "render"), "expected a Jinja Template, got a dict"


# ===================================================================
# Rendering
# ===================================================================


def test_render_system(default_prompts):
    ctx = _ctx(answer="4")
    result = default_prompts.render_system("debater_a", ctx)
    assert "debater_a" in result
    assert "defend" in result


def test_render_system_no_answer(default_prompts):
    ctx = _ctx()
    result = default_prompts.render_system("debater_a", ctx)
    assert "defend" not in result


def test_render_system_missing_role(default_prompts):
    """Raises KeyError when the role isn't declared in system."""
    ctx = _ctx()
    with pytest.raises(KeyError, match="nonexistent"):
        default_prompts.render_system("nonexistent", ctx)


def test_render_system_phase_independent(default_prompts):
    """render_system signature is role-only — the same role always renders
    the same template regardless of what phase we're in (prefix cache invariant)."""
    ctx = _ctx()
    result = default_prompts.render_system("debater_a", ctx)
    # Call it again with a different ctx phase-ish value, result must be identical
    # (build_context would normally include "phase", but render_system doesn't read it).
    ctx2 = _ctx()
    ctx2["phase"] = "critique"
    assert default_prompts.render_system("debater_a", ctx2) == result


def test_render_question(default_prompts):
    """default.yaml's question template is {{ task_prompt }} only — it must
    NOT render the ground-truth answer. Leaking GT into the debater's
    question block collapses debate to echo-the-answer rhetoric."""
    ctx = _ctx(answer="4")
    result = default_prompts.render_question("debater_a", ctx)
    assert result is not None
    assert result == "What is 2+2?"
    assert "assigned position" not in result
    assert "4" not in result


def test_render_question_missing_role(default_prompts):
    """render_question returns None for a role not in question section."""
    # default.yaml has question for all three roles, so use a custom one
    dp = _load_yaml("""\
        version: 2
        system:
          debater_a: "A"
          debater_b: "B"
        question:
          debater_a: "Q"
          debater_b: "Q"
    """)
    ctx = _ctx()
    assert dp.render_question("judge", ctx) is None


def test_render_instruction_with_phase_template(selfplay_prompts):
    ctx = _ctx()
    result = selfplay_prompts.render_instruction("debater_a", "propose", ctx)
    assert result is not None
    assert "compelling" in result


def test_render_instruction_includes_field_instructions(selfplay_prompts):
    ctx = _ctx()
    result = selfplay_prompts.render_instruction("debater_a", "propose", ctx)
    assert "<answer>" in result


def test_render_instruction_none_when_empty():
    dp = _load_yaml("""\
        version: 2
        system:
          debater_a: "A"
          debater_b: "B"
        question:
          debater_a: "Q"
          debater_b: "Q"
    """)
    ctx = _ctx()
    assert dp.render_instruction("debater_a", "propose", ctx) is None


def test_render_prefill_none_when_absent(default_prompts):
    ctx = _ctx()
    assert default_prompts.render_prefill("debater_a", "propose", ctx) is None


def test_render_prefill_returns_value():
    dp = _load_yaml("""\
        version: 2
        system:
          debater_a: "A"
          debater_b: "B"
        question:
          debater_a: "Q"
          debater_b: "Q"
        prefill:
          debater_a:
            propose: "I think the answer is"
    """)
    ctx = _ctx()
    result = dp.render_prefill("debater_a", "propose", ctx)
    assert result == "I think the answer is"


# ===================================================================
# Think visibility
# ===================================================================


def test_think_visibility_default(default_prompts):
    assert default_prompts.think_visibility == {}
    assert default_prompts.think_tag == "thinking"


def test_think_visibility_from_yaml():
    dp = _load_yaml("""\
        version: 2
        system:
          debater_a: "A"
          debater_b: "B"
        question:
          debater_a: "Q"
          debater_b: "Q"
        think:
          debater_a: private
          debater_b: private
          judge: disabled
    """)
    assert dp.think_visibility["debater_a"] == "private"
    assert dp.think_visibility["debater_b"] == "private"
    assert dp.think_visibility["judge"] == "disabled"


def test_think_custom_tag():
    dp = _load_yaml("""\
        version: 2
        system:
          debater_a: "A"
          debater_b: "B"
        question:
          debater_a: "Q"
          debater_b: "Q"
        think:
          debater_a: {tag: reason, visibility: private}
    """)
    assert dp.think_tag == "reason"
    assert dp.think_visibility["debater_a"] == "private"


def test_think_instruction_in_render():
    dp = _load_yaml("""\
        version: 2
        system:
          debater_a: "A"
          debater_b: "B"
        question:
          debater_a: "Q"
          debater_b: "Q"
        user:
          debater_a: {default: "Argue."}
        think:
          debater_a: private
    """)
    ctx = _ctx()
    result = dp.render_instruction("debater_a", "propose", ctx)
    assert "<thinking>" in result
    assert "private" in result.lower()


def test_think_disabled_no_instruction():
    dp = _load_yaml("""\
        version: 2
        system:
          debater_a: "A"
          debater_b: "B"
        question:
          debater_a: "Q"
          debater_b: "Q"
        user:
          debater_a: {default: "Argue."}
        think:
          debater_a: disabled
    """)
    ctx = _ctx()
    result = dp.render_instruction("debater_a", "propose", ctx)
    assert "<thinking>" not in result


# ===================================================================
# Field specs
# ===================================================================


def test_get_field_specs(default_prompts):
    specs = default_prompts.get_field_specs("judge", "final")
    assert specs is not None
    assert "decision" in specs
    assert "reason" in specs


def test_get_field_specs_none_for_missing(default_prompts):
    assert default_prompts.get_field_specs("debater_a", "propose") is None


def test_selfplay_debater_fields(selfplay_prompts):
    propose = selfplay_prompts.get_field_specs("debater_a", "propose")
    assert propose is not None
    assert "answer" in propose

    critique = selfplay_prompts.get_field_specs("debater_a", "critique")
    assert critique is not None
    assert set(critique.keys()) == {"opponent_error", "rebuttal", "answer"}


# ===================================================================
# JudgeTemplate
# ===================================================================


def test_judge_templates_default(default_prompts):
    assert "grader" in default_prompts.judges
    assert "matcher" in default_prompts.judges

    grader = default_prompts.judges["grader"]
    assert isinstance(grader, JudgeTemplate)
    assert grader.positive == "CORRECT"
    assert grader.negative == "INCORRECT"

    matcher = default_prompts.judges["matcher"]
    assert matcher.positive == "SAME"
    assert matcher.negative == "DIFFERENT"


def test_judge_template_carries_user_template_verbatim():
    """JudgeTemplate.user is the raw Python format-string that JudgeRubric
    will call .format() on with {question}/{answer}/{response} keys."""
    dp = _load_yaml("""\
        version: 2
        system:
          debater_a: "A"
          debater_b: "B"
        question:
          debater_a: "Q"
          debater_b: "Q"
        _grader:
          system: "Grade it."
          user: "Target: {answer} Response: {response}"
          positive: CORRECT
          negative: INCORRECT
    """)
    jt = dp.judges["grader"]
    assert jt.user == "Target: {answer} Response: {response}"
    rendered = jt.user.format(answer="42", response="forty-two")
    assert "42" in rendered
    assert "forty-two" in rendered


# ===================================================================
# Opponent wrap
# ===================================================================


def test_wrap_opponent_fallback_attributes_speaker(default_prompts):
    """No opponent_wrap template defined → fall back to [member_id] prefix.
    Bare passthrough would leave attribution implicit (order-dependent) —
    see F1 judge attribution fix."""
    result = default_prompts.wrap_opponent(
        "propose",
        "hello",
        member_id="debater_a",
        viewer_id="debater_b",
    )
    assert result == "[debater_a] hello"


def test_wrap_opponent_with_template_injects_member_label():
    """A template that references {{ member_id }} must render the speaker's
    member id in the wrapped output, so the judge can attribute unambiguously."""
    dp = _load_yaml("""\
        version: 2
        system:
          debater_a: "A"
          debater_b: "B"
        question:
          debater_a: "Q"
          debater_b: "Q"
        opponent_wrap:
          debater: "[{{ member_id }} @ {{ phase }}]: {{ text }}"
    """)
    result_a = dp.wrap_opponent(
        "propose",
        "I argue X",
        member_id="debater_a",
        viewer_id="debater_b",
    )
    result_b = dp.wrap_opponent(
        "propose",
        "I argue Y",
        member_id="debater_b",
        viewer_id="debater_a",
    )
    assert result_a == "[debater_a @ propose]: I argue X"
    assert result_b == "[debater_b @ propose]: I argue Y"
    # Critical: the two must render DIFFERENTLY. This is the whole point of
    # F1 — pre-fix both rendered as identical "OPPONENT" blocks.
    assert result_a != result_b


# ===================================================================
# Validation
# ===================================================================


def test_validate_rejects_missing_version():
    with pytest.raises(ValueError, match="version"):
        _validate({})


def test_validate_rejects_wrong_version():
    with pytest.raises(ValueError, match="version"):
        _validate({"version": 1})


def test_validate_rejects_unknown_role():
    with pytest.raises(ValueError, match="Unknown role"):
        _validate(
            {
                "version": 2,
                "system": {"bad_role": "x"},
                "question": {"debater_a": "q", "debater_b": "q"},
            }
        )


def test_validate_rejects_missing_question_role():
    with pytest.raises(ValueError, match="debater_b"):
        _validate(
            {
                "version": 2,
                "system": {},
                "question": {"debater_a": "q"},
            }
        )


def test_validate_rejects_bad_think_visibility():
    with pytest.raises(ValueError, match="unknown visibility"):
        _validate(
            {
                "version": 2,
                "system": {},
                "question": {"debater_a": "q", "debater_b": "q"},
                "think": {"debater_a": "invalid_vis"},
            }
        )


def test_validate_rejects_bare_true_think():
    with pytest.raises(ValueError, match="bare"):
        _validate(
            {
                "version": 2,
                "system": {},
                "question": {"debater_a": "q", "debater_b": "q"},
                "think": {"debater_a": True},
            }
        )


def test_validate_accepts_minimal():
    _validate(
        {
            "version": 2,
            "system": {"debater_a": "x", "debater_b": "y"},
            "question": {"debater_a": "q", "debater_b": "q"},
        }
    )


def test_validate_rejects_nested_system():
    """System must be role -> string. Phase-indexed system is forbidden
    because it would break prefix caching."""
    with pytest.raises(ValueError, match="expected a string"):
        _validate(
            {
                "version": 2,
                "system": {"debater_a": {"default": "x"}, "debater_b": "y"},
                "question": {"debater_a": "q", "debater_b": "q"},
            }
        )


def test_validate_rejects_bad_judge_block():
    with pytest.raises(ValueError, match="missing"):
        _validate(
            {
                "version": 2,
                "system": {},
                "question": {"debater_a": "q", "debater_b": "q"},
                "_grader": {"system": "x", "user": "y", "positive": "YES"},
            }
        )


def test_validate_rejects_colliding_verdicts():
    with pytest.raises(ValueError, match="distinct"):
        _validate(
            {
                "version": 2,
                "system": {},
                "question": {"debater_a": "q", "debater_b": "q"},
                "_grader": {"system": "x", "user": "y", "positive": "YES", "negative": "yes"},
            }
        )


def test_validate_rejects_bad_opponent_wrap():
    with pytest.raises(ValueError, match="unknown keys"):
        _validate(
            {
                "version": 2,
                "system": {},
                "question": {"debater_a": "q", "debater_b": "q"},
                "opponent_wrap": {"bad_key": "x"},
            }
        )


# ===================================================================
# _normalize_think
# ===================================================================


def test_normalize_think_empty():
    vis, tag = _normalize_think({})
    assert vis == {}
    assert tag == "thinking"


def test_normalize_think_strings():
    vis, tag = _normalize_think({"debater_a": "private", "debater_b": "open"})
    assert vis == {"debater_a": "private", "debater_b": "open"}


def test_normalize_think_false():
    vis, tag = _normalize_think({"debater_a": False})
    assert vis == {"debater_a": "disabled"}


def test_normalize_think_dict_with_tag():
    vis, tag = _normalize_think({"debater_a": {"tag": "reason", "visibility": "private"}})
    assert vis == {"debater_a": "private"}
    assert tag == "reason"
