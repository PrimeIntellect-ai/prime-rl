import pytest

from glyphbench.core.action import ActionSpec


def test_action_spec_n_equals_number_of_names():
    spec = ActionSpec(
        names=("NORTH", "SOUTH"),
        descriptions=("move up", "move down"),
    )
    assert spec.n == 2


def test_action_spec_index_of_valid_name():
    spec = ActionSpec(
        names=("NORTH", "SOUTH", "WAIT"),
        descriptions=("", "", ""),
    )
    assert spec.index_of("NORTH") == 0
    assert spec.index_of("WAIT") == 2


def test_action_spec_index_of_invalid_name_raises_key_error():
    spec = ActionSpec(names=("NORTH",), descriptions=("",))
    with pytest.raises(KeyError, match="FLY"):
        spec.index_of("FLY")


def test_action_spec_render_for_prompt_lists_every_action_with_description():
    spec = ActionSpec(
        names=("NORTH", "SOUTH", "WAIT"),
        descriptions=("move up", "move down", "do nothing"),
    )
    text = spec.render_for_prompt()
    assert "NORTH" in text
    assert "move up" in text
    assert "SOUTH" in text
    assert "WAIT" in text
    assert "do nothing" in text
    assert text.count("\n") >= 3  # header + 3 action lines


def test_action_spec_render_for_prompt_uses_bracketed_actions_section():
    spec = ActionSpec(
        names=("NORTH",),
        descriptions=("move up",),
    )
    text = spec.render_for_prompt()
    assert text.startswith("[Actions]\nPick exactly one per turn:")


def test_action_spec_names_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        ActionSpec(names=("NORTH", "NORTH"), descriptions=("a", "b"))


def test_action_spec_names_and_descriptions_must_have_same_length():
    with pytest.raises(ValueError, match="same length"):
        ActionSpec(names=("NORTH", "SOUTH"), descriptions=("a",))


def test_action_spec_is_frozen():
    spec = ActionSpec(names=("NORTH",), descriptions=("a",))
    try:
        spec.names = ("FLY",)  # type: ignore[misc]
    except Exception as e:
        assert "frozen" in str(e).lower() or "can't set attribute" in str(e).lower() or "cannot assign" in str(e).lower()
    else:
        raise AssertionError("expected frozen dataclass")


def test_index_of_exact_match_preserved():
    spec = ActionSpec(
        names=("NORTH", "SOUTH", "NOOP"),
        descriptions=("", "", ""),
    )
    assert spec.index_of("NORTH") == 0
    assert spec.index_of("SOUTH") == 1
    assert spec.index_of("NOOP") == 2


def test_index_of_case_insensitive_fallback():
    spec = ActionSpec(
        names=("UP", "DOWN", "NOOP"),
        descriptions=("", "", ""),
    )
    assert spec.index_of("up") == 0
    assert spec.index_of("Down") == 1
    assert spec.index_of("nOoP") == 2


def test_index_of_ambiguous_case_raises():
    spec = ActionSpec(
        names=("UP", "up"),
        descriptions=("shouty", "quiet"),
    )
    # Exact matches must still work.
    assert spec.index_of("UP") == 0
    assert spec.index_of("up") == 1
    # But an only-case-matching lookup that is ambiguous must fail.
    with pytest.raises(KeyError, match="Up"):
        spec.index_of("Up")


def test_index_of_whitespace_stripped():
    spec = ActionSpec(
        names=("NORTH", "SOUTH", "NOOP"),
        descriptions=("", "", ""),
    )
    assert spec.index_of("  NORTH  ") == 0
    assert spec.index_of("\tsouth\n") == 1
    with pytest.raises(KeyError):
        spec.index_of("   ")
    with pytest.raises(KeyError):
        spec.index_of("")
