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
    """What to decide, as given: text, or an object or array."""
    criteria: Any
    """The allowed answers, as given. noul: an optional object describing ``true`` and/or ``false``. choice: option
    id -> description (or None). score: the level descriptions, lowest level first."""

    @property
    def options(self) -> list[str]:
        """Option ids in canonical order: ``false``, ``true`` for noul; the criteria keys as given for choice;
        ``0``..``n-1`` for score."""
        if self.type == "noul":
            return ["false", "true"]
        if self.type == "choice":
            return list(self.criteria)
        return [str(level) for level in range(len(self.criteria))]

    @property
    def descriptions(self) -> list[Any]:
        """Each option's description (None where there is none), in the order of ``options``."""
        if self.type == "score":
            return list(self.criteria)
        return [(self.criteria or {}).get(option) for option in self.options]


@dataclass(frozen=True)
class SystemOneRecord:
    """What a model sees: no labels."""

    id: Any
    state: Any
    """The state to decide on: text, or an object or array."""
    questions: tuple[Question, ...]


@dataclass(frozen=True)
class Labels:
    """The labels a row gives for one question."""

    hard: int | None
    """The hard label as an index into ``Question.options``."""
    soft: list[float] | None
    """The soft label as a distribution in the order of ``Question.options``."""


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
    kind, criteria = definition.get("type"), definition.get("criteria")
    if definition.get("instructions") is None:
        raise ValueError(f"{where}: instructions are required")
    if kind == "noul":
        if criteria is not None and (not isinstance(criteria, Mapping) or not set(criteria) <= {"true", "false"}):
            raise ValueError(f"{where}: noul criteria may only describe true and false")
    elif kind == "choice":
        if not isinstance(criteria, Mapping) or not 1 <= len(criteria) <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"{where}: choice criteria must map 1 to {MAX_CHOICE_OPTIONS} options to descriptions")
    elif kind == "score":
        if not isinstance(criteria, list) or not MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS:
            raise ValueError(f"{where}: score criteria must list {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels")
    else:
        raise ValueError(f"{where}: type must be noul, choice or score, got {kind!r}")
    return Question(question_id, kind, definition["instructions"], criteria)


def _hard_label(question: Question, label: Any) -> int:
    """A hard label as an option index: true/false for noul, an option id for choice, a level for score."""
    if question.type == "noul" and isinstance(label, bool):
        label = "true" if label else "false"
    if question.type == "score" and isinstance(label, int) and not isinstance(label, bool):
        label = str(label)
    if label not in question.options:
        raise ValueError(f"question {question.id!r}: label {label!r} is not one of {question.options}")
    return question.options.index(label)


def _soft_label(question: Question, soft_label: Any) -> list[float]:
    where = f"question {question.id!r}"
    if not isinstance(soft_label, Mapping) or set(soft_label) != set(question.options):
        raise ValueError(f"{where}: soft label keys must be {question.options}")
    values = [soft_label[option] for option in question.options]
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) for p in values):
        raise ValueError(f"{where}: soft label values must be numbers")
    if any(not math.isfinite(p) or p < 0 for p in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-6):
        raise ValueError(f"{where}: the soft label must be a distribution")
    return [float(p) for p in values]


def parse_record(row: Mapping, columns: SystemOneColumnsConfig) -> tuple[SystemOneRecord, list[Labels]]:
    """A dataset row -> the checked record and one ``Labels`` per question, in record order.

    The state, questions and label columns hold JSON text. Any question may go without a label.
    """
    record_id = row[columns.id] if columns.id in row else row.get("__index")
    try:
        state = _decode(row, columns.state)
        definitions = _decode(row, columns.questions)
        if not isinstance(definitions, Mapping) or not definitions:
            raise ValueError(f"column {columns.questions!r} must hold a non-empty object")
        questions = tuple(_parse_question(question_id, definition) for question_id, definition in definitions.items())
        hard = _decode_labels(row, columns.label, set(definitions))
        soft = _decode_labels(row, columns.soft_label, set(definitions))
        labels = [
            Labels(
                hard=None if hard.get(q.id) is None else _hard_label(q, hard[q.id]),
                soft=None if soft.get(q.id) is None else _soft_label(q, soft[q.id]),
            )
            for q in questions
        ]
    except ValueError as error:
        raise ValueError(f"record {record_id!r}: {error}") from None
    return SystemOneRecord(record_id, state, questions), labels


def build_targets(record: SystemOneRecord, labels: list[Labels]) -> list[list[float] | None]:
    """Per question, over its options in canonical order: the soft label if given, else the hard label one-hot,
    else None (the question is not trained)."""
    targets: list[list[float] | None] = []
    for question, given in zip(record.questions, labels, strict=True):
        if given.soft is not None:
            targets.append(given.soft)
        elif given.hard is not None:
            targets.append([float(index == given.hard) for index in range(len(question.options))])
        else:
            targets.append(None)
    return targets
