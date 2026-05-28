from __future__ import annotations

import re

from .schema import FinalItem, validate_final_item

DIAGRAM_PATTERNS = [
    r"\bdiagram\b",
    r"\bfigure\b",
    r"\bfig\.",
    r"\bgraph\b",
    r"\bplot\b",
    r"\bsketch\b",
    r"\bdraw\b",
    r"\bshown below\b",
    r"\bin the figure\b",
    r"\bfrom the graph\b",
]

PROOF_ONLY_PATTERNS = [
    r"\bprove that\b",
    r"\bshow that\b",
    r"\bdemonstrate that\b",
]

QUALITATIVE_PATTERNS = [
    r"\bexplain qualitatively\b",
    r"\bdescribe qualitatively\b",
    r"\bdiscuss\b",
    r"\bcomment on\b",
]

IMAGE_MARKERS = [
    r"<image_start>",
    r"\[problem_image",
    r"!\[",
]


def admissibility_rejection(item: FinalItem) -> tuple[str, str] | None:
    if not item.answers:
        return "answer_missing", "answers is empty"
    if not item.official_solution.strip():
        return "missing_solution", "official_solution is empty"
    schema_errors = validate_final_item(item)
    if schema_errors:
        return "verifier_failed", "; ".join(schema_errors)
    if item.requires_diagram:
        return "diagram_required", "requires_diagram is true"

    text = "\n".join([item.problem_text, item.shared_context, item.question])
    lowered = text.lower()
    if _matches_any(lowered, IMAGE_MARKERS):
        return "diagram_required", "image marker found"
    if _matches_any(lowered, DIAGRAM_PATTERNS):
        return "manual_review_needed", "possible diagram reference"
    if _matches_any(lowered, QUALITATIVE_PATTERNS):
        return "qualitative_only", "qualitative prompt language found"
    if _matches_any(lowered, PROOF_ONLY_PATTERNS) and not item.answers:
        return "proof_only", "proof language without final answer"
    return None


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
