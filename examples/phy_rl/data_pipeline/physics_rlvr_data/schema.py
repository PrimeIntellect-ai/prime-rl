from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

AnswerType = Literal[
    "numeric",
    "numerical",
    "symbolic",
    "expression",
    "multiple_choice",
    "multi_select",
    "set",
    "tuple",
    "interval",
    "string",
    "string_exact",
]

VerifierType = Literal[
    "numeric",
    "sympy",
    "expression",
    "mcq",
    "multi_select",
    "set",
    "tuple",
    "interval",
    "string",
    "string_exact",
    "custom",
]

PaperType = Literal["problem", "solution", "marking_scheme", "unknown"]
Split = Literal["train", "dev", "frozen_test"]


@dataclass(frozen=True)
class SourceManifestItem:
    source_id: str
    competition: str
    year: int | None
    paper_type: PaperType
    url: str | None
    local_path: str
    sha256: str
    language: str = "en"
    page_count: int | None = None
    license_status: str = "unknown"
    problem_number: str | None = None
    split: Split | None = None


@dataclass(frozen=True)
class PageClassification:
    page_number: int
    extraction_mode: Literal["embedded_text", "ocr", "hybrid", "failed"]
    text_coverage: float
    formula_density: float
    ocr_needed: bool


@dataclass(frozen=True)
class ExtractedBlock:
    page: int
    block_id: str
    type: Literal["paragraph", "equation", "table", "caption", "list", "unknown"]
    text: str
    latex: str | None = None
    bbox: list[float] | None = None
    ocr_engine: str = "embedded_text"
    confidence: float | None = None


@dataclass(frozen=True)
class ExtractedDocument:
    source_id: str
    sha256: str
    local_path: str
    page_classifications: list[PageClassification]
    blocks: list[ExtractedBlock]
    canonical_markdown: str
    extractor: str


@dataclass(frozen=True)
class Answer:
    value: str
    unit: str | None
    answer_type: AnswerType
    tolerance: float | str | None
    verifier: VerifierType
    equivalent_forms: list[str] = field(default_factory=list)
    subproblem_id: str | None = None


@dataclass(frozen=True)
class Provenance:
    pdf_url: str | None
    page_range: list[int] | None
    ocr_engine: str | None
    ocr_confidence: float | None
    source_hash: str


@dataclass(frozen=True)
class FinalItem:
    problem_id: str
    source: str
    competition: str
    year: int | None
    problem_number: str | None
    subproblem_id: str | None
    problem_text: str
    shared_context: str
    question: str
    official_solution: str
    answers: list[Answer]
    requires_diagram: bool
    language: str
    split: Split
    provenance: Provenance


@dataclass(frozen=True)
class RejectedItem:
    problem_id: str
    source: str
    rejection_reason: str
    detail: str
    item: dict[str, Any]


def to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)


def answer_from_dict(raw: dict[str, Any]) -> Answer:
    return Answer(
        value=str(raw.get("value", "")),
        unit=raw.get("unit"),
        answer_type=raw.get("answer_type", "string"),
        tolerance=raw.get("tolerance"),
        verifier=raw.get("verifier", "string"),
        equivalent_forms=list(raw.get("equivalent_forms", [])),
        subproblem_id=raw.get("subproblem_id"),
    )


def final_item_from_dict(raw: dict[str, Any]) -> FinalItem:
    provenance = Provenance(**raw["provenance"])
    answers = [answer_from_dict(answer) for answer in raw.get("answers", [])]
    return FinalItem(
        problem_id=raw["problem_id"],
        source=raw["source"],
        competition=raw.get("competition", ""),
        year=raw.get("year"),
        problem_number=raw.get("problem_number"),
        subproblem_id=raw.get("subproblem_id"),
        problem_text=raw.get("problem_text", ""),
        shared_context=raw.get("shared_context", ""),
        question=raw.get("question", ""),
        official_solution=raw.get("official_solution", ""),
        answers=answers,
        requires_diagram=bool(raw.get("requires_diagram", False)),
        language=raw.get("language", "en"),
        split=raw["split"],
        provenance=provenance,
    )


def validate_final_item(item: FinalItem) -> list[str]:
    errors: list[str] = []
    required_text = {
        "problem_id": item.problem_id,
        "source": item.source,
        "problem_text": item.problem_text,
        "question": item.question,
        "official_solution": item.official_solution,
        "language": item.language,
    }
    for name, value in required_text.items():
        if not value or not value.strip():
            errors.append(f"{name} is empty")
    if item.language != "en":
        errors.append("language must be en")
    if item.requires_diagram:
        errors.append("requires_diagram must be false for admitted items")
    if item.split not in {"train", "dev", "frozen_test"}:
        errors.append("split must be train, dev, or frozen_test")
    if not item.answers:
        errors.append("answers is empty")
    for index, answer in enumerate(item.answers):
        if not answer.value.strip():
            errors.append(f"answers[{index}].value is empty")
        if answer.answer_type in {"numeric", "numerical"} and answer.verifier != "numeric":
            errors.append(f"answers[{index}] numeric answer must use numeric verifier")
        if answer.answer_type in {"symbolic", "expression"} and answer.verifier not in {"sympy", "expression"}:
            errors.append(f"answers[{index}] symbolic answer must use sympy or expression verifier")
    return errors


def problem_artifact_id(item: SourceManifestItem) -> str:
    parts = [
        item.source_id,
        item.competition or "unknown_competition",
        str(item.year or "unknown_year"),
        item.problem_number or "unknown_problem",
        item.paper_type,
        item.sha256[:12],
    ]
    return "__".join(_clean_part(part) for part in parts)


def _clean_part(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_").lower() or "unknown"
