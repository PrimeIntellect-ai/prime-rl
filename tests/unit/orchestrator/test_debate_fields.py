"""Unit tests for debate field scoring, classifiers, normalizers, and MCQ extraction.

NO mocks. All objects are real FieldSpec, real normalizers, real extract_fields.

Run from the fork venv — see test_debate_env.py docstring for setup.
"""

from __future__ import annotations

import pytest
from verifiers.protocols.debate.fields import (
    BinaryScoring,
    EnumScoring,
    NumericScoring,
    ScoringMode,
    _resolve_fields,
    binary_normalizer,
    classify_binary,
    classify_enum,
    classify_numeric,
    enum_normalizer,
    normalizer_for_scoring,
    numeric_normalizer,
    resolve_scoring,
    validate_type_scoring,
)
from verifiers.protocols.debate.parsing import extract_fields, generate_format_instructions
from verifiers.utils.mcq import normalize_mcq

# ===================================================================
# _resolve_fields — YAML dict → FieldSpec
# ===================================================================


def test_resolve_fields_shorthand_str():
    specs = _resolve_fields({"answer": "str", "score": "float"})
    assert specs["answer"].type is str
    assert specs["score"].type is float
    assert specs["answer"].scoring is None
    assert specs["answer"].normalizer is None


def test_resolve_fields_full_dict():
    specs = _resolve_fields(
        {
            "verdict": {
                "type": "str",
                "description": "Judge verdict",
                "scoring": "binary",
            },
        }
    )
    assert specs["verdict"].type is str
    assert specs["verdict"].description == "Judge verdict"
    assert isinstance(specs["verdict"].scoring, BinaryScoring)
    assert specs["verdict"].normalizer is not None


def test_resolve_fields_with_configured_scoring():
    specs = _resolve_fields(
        {
            "winner": {
                "type": "str",
                "scoring": {"mode": "enum", "values": ["A", "B", "tie"]},
            },
        }
    )
    assert isinstance(specs["winner"].scoring, EnumScoring)
    assert specs["winner"].scoring.values == ("A", "B", "tie")


def test_resolve_fields_bad_type():
    with pytest.raises(ValueError, match="Unknown field type"):
        _resolve_fields({"x": "complex"})


def test_resolve_fields_bad_type_in_dict():
    with pytest.raises(ValueError, match="Unknown field type"):
        _resolve_fields({"x": {"type": "complex"}})


# ===================================================================
# resolve_scoring
# ===================================================================


def test_resolve_scoring_none():
    assert resolve_scoring(None) is None


def test_resolve_scoring_binary_str():
    s = resolve_scoring("binary")
    assert isinstance(s, BinaryScoring)
    assert s.true_value == "yes"
    assert s.false_value == "no"


def test_resolve_scoring_numeric_str():
    s = resolve_scoring("numeric")
    assert isinstance(s, NumericScoring)
    assert s.min_val == 0.0
    assert s.max_val == 1.0


def test_resolve_scoring_dict_configured():
    s = resolve_scoring({"mode": "binary", "true_value": "oui", "false_value": "non"})
    assert isinstance(s, BinaryScoring)
    assert s.true_value == "oui"
    assert s.false_value == "non"


def test_resolve_scoring_enum_dict():
    s = resolve_scoring({"mode": "enum", "values": ["A", "B", "C"]})
    assert isinstance(s, EnumScoring)
    assert s.values == ("A", "B", "C")


def test_resolve_scoring_bare_enum_raises():
    with pytest.raises(ValueError, match="Bare 'enum'"):
        resolve_scoring("enum")


def test_resolve_scoring_unknown_mode():
    with pytest.raises(ValueError, match="Unknown scoring mode"):
        resolve_scoring("quantum")


def test_resolve_scoring_bad_type():
    with pytest.raises(ValueError, match="Invalid scoring value"):
        resolve_scoring(42)


def test_resolve_scoring_dict_missing_mode():
    with pytest.raises(ValueError, match="must include 'mode'"):
        resolve_scoring({"values": ["a", "b"]})


def test_resolve_scoring_enum_values_as_string():
    with pytest.raises(ValueError, match="must be a list"):
        resolve_scoring({"mode": "enum", "values": "abc"})


def test_resolve_scoring_binary_collision():
    with pytest.raises(ValueError, match="collide"):
        resolve_scoring({"mode": "binary", "true_value": "Yes", "false_value": "yes"})


def test_resolve_scoring_binary_multiword():
    with pytest.raises(ValueError, match="single token"):
        resolve_scoring({"mode": "binary", "true_value": "yes indeed"})


def test_resolve_scoring_binary_empty():
    with pytest.raises(ValueError, match="non-empty"):
        resolve_scoring({"mode": "binary", "true_value": "  "})


def test_resolve_scoring_binary_bad_agg():
    with pytest.raises(ValueError, match="agg must be"):
        resolve_scoring({"mode": "binary", "agg": "sum"})


def test_resolve_scoring_enum_empty_values():
    with pytest.raises(ValueError, match="non-empty"):
        resolve_scoring({"mode": "enum", "values": []})


def test_resolve_scoring_enum_collision():
    with pytest.raises(ValueError, match="collide"):
        resolve_scoring({"mode": "enum", "values": ["Yes", "yes"]})


def test_resolve_scoring_numeric_inverted():
    with pytest.raises(ValueError, match="min_val.*<=.*max_val"):
        resolve_scoring({"mode": "numeric", "min_val": 10.0, "max_val": 1.0})


def test_resolve_scoring_numeric_infinite():
    with pytest.raises(ValueError, match="finite"):
        resolve_scoring({"mode": "numeric", "min_val": float("inf")})


# ===================================================================
# validate_type_scoring
# ===================================================================


def test_validate_type_scoring_compatible():
    validate_type_scoring("v", str, BinaryScoring())
    validate_type_scoring("v", str, EnumScoring(values=("a",)))
    validate_type_scoring("s", float, NumericScoring())
    validate_type_scoring("s", int, NumericScoring())


def test_validate_type_scoring_incompatible_numeric_str():
    with pytest.raises(ValueError, match="incompatible"):
        validate_type_scoring("score", str, NumericScoring())


def test_validate_type_scoring_incompatible_binary_float():
    with pytest.raises(ValueError, match="incompatible"):
        validate_type_scoring("verdict", float, BinaryScoring())


def test_validate_type_scoring_incompatible_enum_int():
    with pytest.raises(ValueError, match="incompatible"):
        validate_type_scoring("choice", int, EnumScoring(values=("a",)))


# ===================================================================
# classify_binary
# ===================================================================


def test_classify_binary_exact():
    c = classify_binary("yes", "yes", "no")
    assert c.is_true is True
    assert c.is_valid is True
    assert c.canonical == "yes"


def test_classify_binary_exact_false():
    c = classify_binary("no", "yes", "no")
    assert c.is_true is False
    assert c.is_valid is True


def test_classify_binary_case_insensitive():
    c = classify_binary("YES", "yes", "no")
    assert c.is_true is True
    assert c.is_valid is True


def test_classify_binary_first_token_fallback():
    c = classify_binary("yes, definitely", "yes", "no")
    assert c.is_true is True
    assert c.is_valid is True


def test_classify_binary_invalid():
    c = classify_binary("maybe", "yes", "no")
    assert c.is_true is None
    assert c.is_valid is False


def test_classify_binary_custom_values():
    c = classify_binary("Correct!", "correct", "incorrect")
    assert c.is_true is True
    assert c.is_valid is True
    assert c.canonical == "correct"


def test_classify_binary_strips_punctuation():
    c = classify_binary('"yes"', "yes", "no")
    assert c.is_true is True
    assert c.is_valid is True


# ===================================================================
# classify_enum
# ===================================================================


def test_classify_enum_exact():
    c = classify_enum("A", ("A", "B", "tie"))
    assert c.canonical == "A"
    assert c.is_valid is True


def test_classify_enum_case_insensitive():
    c = classify_enum("TIE", ("A", "B", "tie"))
    assert c.canonical == "tie"
    assert c.is_valid is True


def test_classify_enum_first_token_fallback():
    c = classify_enum("A wins clearly", ("A", "B", "tie"))
    assert c.canonical == "A"
    assert c.is_valid is True


def test_classify_enum_invalid():
    c = classify_enum("draw", ("A", "B", "tie"))
    assert c.canonical is None
    assert c.is_valid is False


# ===================================================================
# classify_numeric
# ===================================================================


def test_classify_numeric_in_range():
    c = classify_numeric(0.5, 0.0, 1.0)
    assert c.value == 0.5
    assert c.is_valid is True


def test_classify_numeric_boundaries():
    assert classify_numeric(0.0, 0.0, 1.0).is_valid is True
    assert classify_numeric(1.0, 0.0, 1.0).is_valid is True


def test_classify_numeric_out_of_range():
    c = classify_numeric(2.0, 0.0, 1.0)
    assert c.value is None
    assert c.is_valid is False


def test_classify_numeric_string_coercion():
    c = classify_numeric("0.7", 0.0, 1.0)
    assert c.value == 0.7
    assert c.is_valid is True


def test_classify_numeric_string_out_of_range():
    c = classify_numeric("5.0", 0.0, 1.0)
    assert c.is_valid is False


def test_classify_numeric_bool_rejected():
    c = classify_numeric(True, 0.0, 1.0)
    assert c.is_valid is False


def test_classify_numeric_int():
    c = classify_numeric(1, 0, 10)
    assert c.value == 1.0
    assert c.is_valid is True


def test_classify_numeric_nan():
    c = classify_numeric(float("nan"), 0.0, 1.0)
    assert c.is_valid is False


def test_classify_numeric_inf():
    c = classify_numeric(float("inf"), 0.0, 1.0)
    assert c.is_valid is False


def test_classify_numeric_bad_string():
    c = classify_numeric("abc", 0.0, 1.0)
    assert c.is_valid is False


# ===================================================================
# Normalizers
# ===================================================================


def test_binary_normalizer_valid():
    norm = binary_normalizer("yes", "no")
    assert norm("YES") == "yes"
    assert norm("no!") == "no"


def test_binary_normalizer_passthrough():
    norm = binary_normalizer("yes", "no")
    assert norm("dunno") == "dunno"


def test_enum_normalizer_exact():
    norm = enum_normalizer(("A", "B", "C"))
    assert norm("a") == "A"
    assert norm("B") == "B"


def test_enum_normalizer_mcq_fallback():
    norm = enum_normalizer(("A", "B", "C", "D", "E"))
    assert norm("the answer is B") == "B"
    assert norm("C) 10") == "C"


def test_enum_normalizer_non_mcq_no_fallback():
    norm = enum_normalizer(("high", "medium", "low"))
    assert norm("the answer is high") == "the answer is high"


def test_enum_normalizer_passthrough():
    norm = enum_normalizer(("A", "B"))
    assert norm("Z") == "Z"


def test_numeric_normalizer_valid():
    norm = numeric_normalizer(0.0, 1.0)
    assert norm(0.5) == 0.5
    assert norm("0.8") == 0.8


def test_numeric_normalizer_passthrough():
    norm = numeric_normalizer(0.0, 1.0)
    assert norm("abc") == "abc"
    assert norm(5.0) == 5.0


def test_normalizer_for_scoring_binary():
    norm = normalizer_for_scoring(BinaryScoring(true_value="oui", false_value="non"))
    assert norm("OUI") == "oui"


def test_normalizer_for_scoring_enum():
    norm = normalizer_for_scoring(EnumScoring(values=("x", "y")))
    assert norm("X") == "x"


def test_normalizer_for_scoring_numeric():
    norm = normalizer_for_scoring(NumericScoring(min_val=1.0, max_val=5.0))
    assert norm(3.0) == 3.0


def test_normalizer_for_scoring_base():
    assert normalizer_for_scoring(ScoringMode()) is None


# ===================================================================
# extract_fields — full XML pipeline
# ===================================================================


def test_extract_fields_basic():
    specs = _resolve_fields(
        {
            "verdict": {"type": "str", "scoring": "binary"},
            "score": {"type": "float", "scoring": {"mode": "numeric", "min_val": 0, "max_val": 10}},
        }
    )
    text = "Some reasoning. <verdict>Yes</verdict> <score>7.5</score>"
    result = extract_fields(text, specs)
    assert result is not None
    assert result["verdict"] == "yes"
    assert result["score"] == 7.5


def test_extract_fields_enum_normalization():
    specs = _resolve_fields(
        {
            "winner": {"type": "str", "scoring": {"mode": "enum", "values": ["A", "B", "tie"]}},
        }
    )
    text = "<winner>TIE</winner>"
    result = extract_fields(text, specs)
    assert result is not None
    assert result["winner"] == "tie"


def test_extract_fields_no_xml():
    specs = _resolve_fields({"answer": "str"})
    assert extract_fields("no xml here", specs) is None


def test_extract_fields_unknown_tag_ignored():
    specs = _resolve_fields({"answer": "str"})
    text = "<bogus>hello</bogus>"
    assert extract_fields(text, specs) is None


def test_extract_fields_partial_match():
    specs = _resolve_fields({"a": "str", "b": "str"})
    text = "<a>hello</a>"
    result = extract_fields(text, specs)
    assert result is not None
    assert result["a"] == "hello"
    assert "b" not in result


def test_extract_fields_coercion_failure():
    specs = _resolve_fields({"count": "int"})
    text = "<count>not_a_number</count>"
    assert extract_fields(text, specs) is None


def test_extract_fields_duplicate_tag_raises():
    """Duplicate schema tag in a response is ambiguous — parse must fail loud,
    never silently commit to last-wins. See G4.parser_dup_last_wins."""
    import pytest

    specs = _resolve_fields({"answer": "str"})
    with pytest.raises(ValueError, match="Duplicate XML tag 'answer'"):
        extract_fields("<answer>A</answer> <answer>B</answer>", specs)


def test_extract_fields_duplicate_non_schema_tag_tolerated():
    """Duplicate tags outside the schema are not our concern — don't false-positive."""
    specs = _resolve_fields({"answer": "str"})
    result = extract_fields("<foo>x</foo><foo>y</foo><answer>A</answer>", specs)
    assert result == {"answer": "A"}


# ===================================================================
# generate_format_instructions
# ===================================================================


def test_format_instructions_binary():
    specs = _resolve_fields({"verdict": {"type": "str", "scoring": "binary"}})
    instr = generate_format_instructions(specs)
    assert "<verdict>" in instr
    assert "yes or no" in instr


def test_format_instructions_enum():
    specs = _resolve_fields(
        {
            "winner": {"type": "str", "scoring": {"mode": "enum", "values": ["A", "B"]}},
        }
    )
    instr = generate_format_instructions(specs)
    assert "WINNER" in instr
    assert "A, B" in instr


def test_format_instructions_numeric():
    specs = _resolve_fields(
        {
            "score": {"type": "float", "scoring": {"mode": "numeric", "min_val": 1, "max_val": 5}},
        }
    )
    instr = generate_format_instructions(specs)
    assert "1" in instr and "5" in instr


def test_format_instructions_plain_with_desc():
    specs = _resolve_fields({"notes": {"type": "str", "description": "Your notes"}})
    instr = generate_format_instructions(specs)
    assert "Your notes" in instr


def test_format_instructions_plain_no_desc():
    specs = _resolve_fields({"answer": "str"})
    instr = generate_format_instructions(specs)
    assert "<answer>" in instr
    assert "str" in instr


# ===================================================================
# normalize_mcq — cascade extraction
# ===================================================================


def test_mcq_bare_letter():
    assert normalize_mcq("C") == "C"
    assert normalize_mcq("a") == "A"


def test_mcq_option_prefix():
    assert normalize_mcq("C) 10") == "C"
    assert normalize_mcq("(B) answer") == "B"


def test_mcq_answer_frame():
    assert normalize_mcq("the answer is B") == "B"
    assert normalize_mcq("The correct answer is D") == "D"
    assert normalize_mcq("answer: C") == "C"


def test_mcq_choose_frame():
    assert normalize_mcq("I choose A") == "A"
    assert normalize_mcq("I pick B") == "B"
    assert normalize_mcq("I select (C)") == "C"


def test_mcq_option_label():
    assert normalize_mcq("option B") == "B"
    assert normalize_mcq("choice D") == "D"


def test_mcq_first_line():
    assert normalize_mcq("B\nsome explanation") == "B"


def test_mcq_terminal_letter():
    assert normalize_mcq("After considering everything, my answer is D") == "D"


def test_mcq_hedged_returns_none():
    assert normalize_mcq("not sure about this") is None
    assert normalize_mcq("I'm unsure") is None
    assert normalize_mcq("uncertain about the answer") is None


def test_mcq_multi_returns_none():
    assert normalize_mcq("both A and B") is None
    assert normalize_mcq("either C or D") is None


def test_mcq_empty_returns_none():
    assert normalize_mcq("") is None
    assert normalize_mcq("<thinking>just thinking</thinking>") is None


def test_mcq_xml_stripped():
    assert normalize_mcq("<answer>C</answer>") == "C"
