"""System One records: a state, typed questions about it, and the label of each question."""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from prime_rl.configs.sft import SystemOneColumnsConfig

QuestionType = Literal["noul", "choice", "score"]

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10


@dataclass(frozen=True)
class Question:
    id: str
    type: QuestionType
    instructions: Any
    """What to decide, as given: text, or an object or array. Absent or null is ``""``."""
    criteria: Any
    """The allowed answers: noul an optional ``true``/``false`` -> description, choice option id -> description,
    score the level descriptions, lowest first."""

    @property
    def options(self) -> list[str]:
        """Option ids in canonical order: ``false``, ``true`` for noul; the criteria keys as given for choice;
        ``0``..``n-1`` for score."""
        if self.type == "noul":
            return ["false", "true"]
        if self.type == "choice":
            return list(self.criteria)
        return [str(level) for level in range(len(self.criteria))]


@dataclass(frozen=True)
class SystemOneRequest:
    """What a model sees: the record without its labels."""

    id: Any
    state: Any
    """The state to decide on: text, or an object or array."""
    questions: tuple[Question, ...]


def _decode(row: Mapping, column: str) -> Any:
    text = row.get(column)
    if not isinstance(text, str):
        raise ValueError(f"column {column!r} must hold JSON text, got {type(text).__name__}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"column {column!r} is not JSON text (plain text is stored as a JSON string): {error}")


def _decode_labels(row: Mapping, column: str, question_ids: set[str]) -> Mapping:
    """A label column as question id -> label. An absent column and a null both mean no labels."""
    given = {} if row.get(column) is None else _decode(row, column)
    if given is None:
        return {}
    if not isinstance(given, Mapping):
        raise ValueError(f"column {column!r} must hold an object (question id -> label)")
    if not set(given) <= question_ids:
        raise ValueError(f"column {column!r} has labels for unknown questions {sorted(set(given) - question_ids)}")
    return given


def _parse_question(question_id: str, definition: Any) -> Question:
    where = f"question {question_id!r}"
    if not isinstance(definition, Mapping):
        raise ValueError(f"{where}: the definition must be an object")
    if unknown := definition.keys() - {"type", "instructions", "criteria"}:
        raise ValueError(f"{where}: unknown field(s) {sorted(unknown)}")
    kind, criteria = definition.get("type"), definition.get("criteria")
    if kind == "noul":
        if criteria is not None and (not isinstance(criteria, Mapping) or not set(criteria) <= {"true", "false"}):
            raise ValueError(f"{where}: noul criteria may only describe true and false")
        if criteria is not None:
            criteria = {option: text for option, text in criteria.items() if text is not None} or None
    elif kind == "choice":
        if not isinstance(criteria, Mapping) or not 1 <= len(criteria) <= MAX_CHOICE_OPTIONS or "" in criteria:
            raise ValueError(
                f"{where}: choice criteria must map 1 to {MAX_CHOICE_OPTIONS} non-empty option ids to descriptions"
            )
    elif kind == "score":
        if (
            not isinstance(criteria, list)
            or not MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS
            or "" in criteria
            or len({str(level) for level in criteria}) != len(criteria)
        ):
            raise ValueError(
                f"{where}: score criteria must list {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} distinct non-empty levels"
            )
    else:
        raise ValueError(f"{where}: type must be noul, choice or score, got {kind!r}")
    instructions = definition.get("instructions")
    return Question(question_id, kind, "" if instructions is None else instructions, criteria)


def _hard_label(question: Question, label: Any) -> list[float]:
    """A hard label, true/false for noul, an option id for choice or a level for score, one-hot over the options."""
    if question.type == "noul" and isinstance(label, bool):
        label = "true" if label else "false"
    # pandas writes a whole level as a float in a column with nulls
    if question.type == "score" and isinstance(label, float) and label.is_integer():
        label = int(label)
    if question.type == "score" and isinstance(label, int) and not isinstance(label, bool):
        label = str(label)
    if label not in question.options:
        raise ValueError(f"question {question.id!r}: label {label!r} is not one of {question.options}")
    return [float(option == label) for option in question.options]


def _soft_label(question: Question, soft_label: Any) -> list[float]:
    where = f"question {question.id!r}"
    if not isinstance(soft_label, Mapping) or set(soft_label) != set(question.options):
        raise ValueError(f"{where}: soft label keys must be {question.options}")
    values = [soft_label[option] for option in question.options]
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) for p in values):
        raise ValueError(f"{where}: soft label values must be numbers")
    # Rounding to two decimals moves each probability by up to 0.005
    total, tolerance = sum(values), max(0.05, 0.005 * len(values))
    # The 1e-9 lets a sum at exactly the tolerance in decimal through despite float error.
    if any(not math.isfinite(p) or p < 0 for p in values) or total <= 0 or abs(total - 1) > tolerance + 1e-9:
        raise ValueError(
            f"{where}: the soft label must be a distribution (non-negative probabilities that sum to 1 within "
            f"{tolerance:g})"
        )
    return [p / total for p in values]


def parse_record(
    row: Mapping, columns: SystemOneColumnsConfig, source: str | None = None, split: str | None = None
) -> tuple[SystemOneRequest, list[list[float] | None]]:
    """A dataset row -> the checked request and each question's target over its options: the soft label if given,
    else the hard label one-hot, else None (the question is not trained). The columns hold JSON text. A record without
    an id is named by its source ``source``, the source's ``split`` and its row index."""
    record_id = row.get(columns.id)
    if record_id is None:
        record_id = "/".join(str(part) for part in (source, split, row.get("__index")) if part is not None)
    try:
        state = _decode(row, columns.state)
        definitions = _decode(row, columns.questions)
        if not isinstance(definitions, Mapping) or not definitions or "" in definitions:
            raise ValueError(f"column {columns.questions!r} must hold a non-empty object with non-empty question ids")
        questions = tuple(_parse_question(question_id, definition) for question_id, definition in definitions.items())
        hard = _decode_labels(row, columns.label, set(definitions))
        soft = _decode_labels(row, columns.soft_label, set(definitions))
        targets = []
        for q in questions:
            one_hot = None if hard.get(q.id) is None else _hard_label(q, hard[q.id])
            targets.append(one_hot if soft.get(q.id) is None else _soft_label(q, soft[q.id]))
    # Deeply nested JSON raises RecursionError, a number too large for a float OverflowError
    except (ValueError, RecursionError, OverflowError) as error:
        raise ValueError(f"record {record_id!r}: {error}") from None
    return SystemOneRequest(record_id, state, questions), targets
