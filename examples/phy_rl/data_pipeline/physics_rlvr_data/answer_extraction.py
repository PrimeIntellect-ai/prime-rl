from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_jsonl
from .schema import Answer, FinalItem, answer_from_dict, final_item_from_dict, to_dict
from .verifiers import extract_boxed


ANSWER_CUE_RE = re.compile(
    r"\b("
    r"final answer|result|therefore|thus|hence|we obtain|we get|we have|by solving|is given by|"
    r"required relation|required expression|obtain the expression|it follows|leads to|equal to|amounts to"
    r")\b",
    flags=re.IGNORECASE,
)
WEAK_CUE_RE = re.compile(r"\b(where|let|suppose|consider|definition|substituting|combining)\b", flags=re.IGNORECASE)
SECTION_RE = re.compile(
    r"^(?:\*{0,2})?\s*(?P<label>(?:task|part|question|solution of task)\s*\d+[a-z]?|"
    r"[A-E]\.\d+|[A-E]\b|\d+[a-z]\)|task\d+[a-z]\))",
    flags=re.IGNORECASE,
)
DISPLAY_MATH_RE = re.compile(r"\$\$(.*?)\$\$", flags=re.DOTALL)
INLINE_MATH_RE = re.compile(r"\$(?!\$)(.*?)(?<!\$)\$", flags=re.DOTALL)
NUMERIC_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*(?:\\times|x|\*)\s*10\^?\{?[+-]?\d+\}?)?$"
)


@dataclass(frozen=True)
class AnswerProposal:
    answer: Answer
    score: int
    evidence: str


def extract_answer_key(input_path: Path, out_path: Path, review_path: Path, *, min_score: int = 6) -> tuple[int, int]:
    accepted_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for raw in read_jsonl(input_path):
        if "rejection_reason" in raw:
            review_rows.append(
                {
                    "problem_id": raw.get("problem_id"),
                    "status": "rejected_candidate",
                    "reason": raw.get("rejection_reason"),
                    "detail": raw.get("detail"),
                }
            )
            continue

        item = final_item_from_dict(raw)
        if item.answers:
            accepted_rows.append(_answer_key_row(item, item.answers, [], status="already_answered"))
            continue

        proposals = extract_answer_proposals(item)
        accepted = _dedupe_proposals([proposal for proposal in proposals if proposal.score >= min_score])
        expected_answers = _expected_answer_count(item.question)
        if accepted and _has_answer_coverage(accepted, expected_answers):
            accepted_rows.append(
                _answer_key_row(
                    item,
                    [proposal.answer for proposal in accepted],
                    accepted,
                    status="auto_extracted_high_precision",
                )
            )
        else:
            review_rows.append(
                {
                    "problem_id": item.problem_id,
                    "source": item.source,
                    "year": item.year,
                    "problem_number": item.problem_number,
                    "status": "needs_manual_answer_key",
                    "reason": _review_reason(accepted, expected_answers),
                    "expected_answer_count": expected_answers,
                    "accepted_candidate_count": len(accepted),
                    "candidate_count": len(proposals),
                    "candidate_answers_by_subproblem": _group_candidate_answers(proposals),
                    "top_candidates": [
                        {
                            "value": proposal.answer.value,
                            "unit": proposal.answer.unit,
                            "answer_type": proposal.answer.answer_type,
                            "verifier": proposal.answer.verifier,
                            "subproblem_id": proposal.answer.subproblem_id,
                            "score": proposal.score,
                            "evidence": proposal.evidence,
                        }
                        for proposal in sorted(proposals, key=lambda candidate: candidate.score, reverse=True)[:10]
                    ],
                }
            )

    return write_jsonl(out_path, accepted_rows), write_jsonl(review_path, review_rows)


def extract_answer_draft(input_path: Path, out_path: Path, audit_path: Path) -> tuple[int, int]:
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for raw in read_jsonl(input_path):
        if "rejection_reason" in raw:
            audit_rows.append(
                {
                    "problem_id": raw.get("problem_id"),
                    "status": "no_answer_draft",
                    "reason": raw.get("rejection_reason"),
                    "detail": raw.get("detail"),
                }
            )
            continue

        item = final_item_from_dict(raw)
        proposals = extract_answer_proposals(item)
        if not proposals:
            proposals = _fallback_answer_proposals(item)
        selected = _select_draft_answers(item, proposals)
        status = _draft_status(item, selected, proposals)
        rows.append(
            _answer_key_row(
                item,
                [proposal.answer for proposal in selected],
                selected,
                status=status,
            )
        )
        audit_rows.append(
            {
                "problem_id": item.problem_id,
                "source": item.source,
                "year": item.year,
                "problem_number": item.problem_number,
                "status": status,
                "expected_answer_count": _expected_answer_count(item.question),
                "selected_answer_count": len(selected),
                "candidate_count": len(proposals),
                "selected_answers": [
                    {
                        "value": proposal.answer.value,
                        "unit": proposal.answer.unit,
                        "answer_type": proposal.answer.answer_type,
                        "verifier": proposal.answer.verifier,
                        "subproblem_id": proposal.answer.subproblem_id,
                        "score": proposal.score,
                        "evidence": proposal.evidence,
                    }
                    for proposal in selected
                ],
                "candidate_answers_by_subproblem": _group_candidate_answers(proposals),
            }
        )
    return write_jsonl(out_path, rows), write_jsonl(audit_path, audit_rows)


def extract_answer_proposals(item: FinalItem) -> list[AnswerProposal]:
    text = item.official_solution
    proposals = []
    proposals.extend(_explicit_answer_block_proposals(text))
    proposals.extend(_solution_block_proposals(text))
    for value in extract_boxed(text):
        proposals.append(_proposal(value, score=6, evidence="boxed answer", subproblem_id=None))

    lines = text.splitlines()
    for index, line in enumerate(lines):
        if _is_noise_line(line):
            continue
        context = _context(lines, index)
        score = _context_score(context)
        if score < 2:
            continue

        subproblem_id = _nearest_subproblem_id(lines, index)
        for value in _math_values(line):
            proposals.append(_proposal(value, score=score, evidence=_trim(context), subproblem_id=subproblem_id))

        for value, unit in _answer_colon_values(line):
            proposals.append(
                _proposal(value, score=max(score, 5), evidence=_trim(context), subproblem_id=subproblem_id, unit=unit)
            )

    return _dedupe_proposals(proposals)


def expected_subproblem_ids(question: str) -> list[str]:
    return _expected_subproblem_ids(question)


def normalize_subproblem_id(value: str) -> str:
    return _normalize_subproblem_id(value)


def solution_blocks_by_subproblem(text: str) -> dict[str, str]:
    blocks: dict[str, list[str]] = {}
    for subproblem_id, block in _solution_blocks(text):
        blocks.setdefault(subproblem_id, []).append(block)
    return {subproblem_id: "\n\n".join(parts).strip() for subproblem_id, parts in blocks.items()}


def fallback_answer_proposals(item: FinalItem) -> list[AnswerProposal]:
    return _fallback_answer_proposals(item)


def dedupe_answer_proposals(proposals: list[AnswerProposal]) -> list[AnswerProposal]:
    return _dedupe_proposals(proposals)


def top_answerish_proposals(proposals: list[AnswerProposal], *, limit: int) -> list[AnswerProposal]:
    return _top_answerish(proposals, limit=limit)


def answer_proposals_to_review(proposals: list[AnswerProposal], *, limit: int = 8) -> list[dict[str, Any]]:
    return [
        {
            "value": proposal.answer.value,
            "unit": proposal.answer.unit,
            "answer_type": proposal.answer.answer_type,
            "verifier": proposal.answer.verifier,
            "subproblem_id": proposal.answer.subproblem_id,
            "score": proposal.score,
            "evidence": proposal.evidence,
        }
        for proposal in sorted(proposals, key=lambda proposal: proposal.score, reverse=True)[:limit]
    ]


def _group_candidate_answers(proposals: list[AnswerProposal]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[AnswerProposal]] = {}
    for proposal in sorted(proposals, key=lambda candidate: candidate.score, reverse=True):
        key = proposal.answer.subproblem_id or "unknown"
        grouped.setdefault(key, []).append(proposal)
    return {
        key: [
            {
                "value": proposal.answer.value,
                "unit": proposal.answer.unit,
                "answer_type": proposal.answer.answer_type,
                "verifier": proposal.answer.verifier,
                "score": proposal.score,
                "evidence": proposal.evidence,
            }
            for proposal in values[:8]
        ]
        for key, values in grouped.items()
    }


def _solution_block_proposals(text: str) -> list[AnswerProposal]:
    blocks = _solution_blocks(text)
    proposals = []
    for subproblem_id, block in blocks:
        for value, evidence in _solution_answer_values(block):
            proposals.append(_proposal(value, score=5, evidence=evidence, subproblem_id=subproblem_id))
    return proposals


def _solution_blocks(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    blocks = []
    current_label: str | None = None
    index = 0
    while index < len(lines):
        section_match = SECTION_RE.match(lines[index].strip())
        if section_match:
            current_label = _normalize_subproblem_id(section_match.group("label"))
        if current_label and lines[index].strip().lower().rstrip(":") == "solution":
            start = index + 1
            end = start
            while end < len(lines):
                stripped = lines[end].strip()
                if _is_marking_heading(stripped):
                    break
                next_section = SECTION_RE.match(stripped)
                if next_section and _normalize_subproblem_id(next_section.group("label")) != current_label:
                    break
                end += 1
            block = "\n".join(lines[start:end]).strip()
            if block:
                blocks.append((current_label, block))
            index = end
            continue
        index += 1
    return blocks


def _solution_answer_values(block: str) -> list[tuple[str, str]]:
    values = []
    for sentence in _solution_sentences(block):
        lowered = sentence.lower()
        if not _has_solution_answer_cue(lowered):
            continue
        math_values = [match.strip() for match in DISPLAY_MATH_RE.findall(sentence)]
        without_display = DISPLAY_MATH_RE.sub("", sentence)
        math_values.extend(match.strip() for match in INLINE_MATH_RE.findall(without_display))
        for value in math_values:
            cleaned = _clean_value(value)
            if _valid_answer_value(cleaned):
                values.append((cleaned, _trim(sentence)))
    return values


def _solution_sentences(block: str) -> list[str]:
    normalized = re.sub(r"\n+", " ", block)
    return [part.strip() for part in re.split(r"(?<=[.;:])\s+", normalized) if part.strip()]


def _has_solution_answer_cue(lowered_sentence: str) -> bool:
    return any(
        cue in lowered_sentence
        for cue in [
            "hence",
            "then",
            "thus",
            "finally",
            "we get",
            "one finds",
            "which gives",
            "which writes",
            "so that",
            "the numerical value",
            "the value is",
            "is equal to",
        ]
    )


def _is_marking_heading(value: str) -> bool:
    normalized = value.lower().replace("sheme", "scheme")
    return normalized in {"marking scheme", "marker scheme", "mark scheme"}


def _explicit_answer_block_proposals(text: str) -> list[AnswerProposal]:
    proposals = []
    lines = [line.strip() for line in text.splitlines()]
    index = 0
    while index < len(lines):
        match = re.match(r"^Answer\s*:\s*(?P<value>.+)$", lines[index], flags=re.IGNORECASE)
        if not match:
            index += 1
            continue
        value = _clean_value(match.group("value"))
        unit = None
        answer_type = None
        lookahead = index + 1
        while lookahead < min(len(lines), index + 4):
            unit_match = re.match(r"^Unit\s*:\s*(?P<unit>.+)$", lines[lookahead], flags=re.IGNORECASE)
            type_match = re.match(
                r"^Answer\s+Type\s*:\s*(?P<answer_type>.+)$",
                lines[lookahead],
                flags=re.IGNORECASE,
            )
            if unit_match:
                unit = _clean_unit(unit_match.group("unit"))
            elif type_match:
                answer_type = type_match.group("answer_type")
            elif re.match(r"^Answer\s*:", lines[lookahead], flags=re.IGNORECASE):
                break
            lookahead += 1
        if answer_type is None:
            index = lookahead
            continue
        proposals.append(
            _proposal(
                value,
                score=7,
                evidence=_trim("\n".join(lines[index:lookahead])),
                subproblem_id=None,
                unit=unit,
                answer_type_hint=answer_type,
            )
        )
        index = lookahead
    return proposals


def _answer_key_row(
    item: FinalItem,
    answers: list[Answer],
    proposals: list[AnswerProposal],
    *,
    status: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "problem_id": item.problem_id,
        "source": item.source,
        "year": item.year,
        "problem_number": item.problem_number,
        "answers": [to_dict(answer) for answer in answers],
        "answer_extraction_status": status,
    }
    if proposals:
        row["answer_extraction_evidence"] = [
            {
                "value": proposal.answer.value,
                "subproblem_id": proposal.answer.subproblem_id,
                "score": proposal.score,
                "evidence": proposal.evidence,
            }
            for proposal in proposals
        ]
    return row


def _proposal(
    value: str,
    *,
    score: int,
    evidence: str,
    subproblem_id: str | None,
    unit: str | None = None,
    answer_type_hint: str | None = None,
) -> AnswerProposal:
    value, parsed_unit = _split_unit(value)
    answer_type, verifier = _answer_kind(value, answer_type_hint)
    return AnswerProposal(
        answer=Answer(
            value=value,
            unit=unit or parsed_unit,
            answer_type=answer_type,
            tolerance=0.05 if verifier == "numeric" else None,
            verifier=verifier,
            equivalent_forms=[],
            subproblem_id=subproblem_id,
        ),
        score=score,
        evidence=evidence,
    )


def _dedupe_proposals(proposals: list[AnswerProposal]) -> list[AnswerProposal]:
    best: dict[tuple[str | None, str], AnswerProposal] = {}
    for proposal in proposals:
        key = (proposal.answer.subproblem_id, _normalize_value(proposal.answer.value))
        current = best.get(key)
        if current is None or proposal.score > current.score:
            best[key] = proposal
    return sorted(best.values(), key=lambda proposal: (proposal.answer.subproblem_id or "", -proposal.score))


def _expected_answer_count(question: str) -> int:
    labels = set(_expected_subproblem_ids(question))
    if labels:
        return len(labels)
    bullet_count = len(re.findall(r"(?m)^\s*[-*]\s+", question))
    return bullet_count


def _expected_subproblem_ids(question: str) -> list[str]:
    labels = []
    for match in re.finditer(r"(?m)^\s*(?P<label>[A-E]\.\d+|Task\s+\d+[a-z]?|\d+[a-z]\))\b", question):
        label = _normalize_subproblem_id(match.group("label"))
        if label not in labels:
            labels.append(label)
    return labels


def _select_draft_answers(item: FinalItem, proposals: list[AnswerProposal]) -> list[AnswerProposal]:
    proposals = _dedupe_proposals(proposals)
    expected_labels = _expected_subproblem_ids(item.question)
    if expected_labels:
        selected = []
        by_subproblem: dict[str, list[AnswerProposal]] = {}
        unknown = []
        for proposal in proposals:
            if proposal.answer.subproblem_id:
                by_subproblem.setdefault(proposal.answer.subproblem_id, []).append(proposal)
            else:
                unknown.append(proposal)
        for label in expected_labels:
            selected.extend(_top_answerish(by_subproblem.get(label, []), limit=3))
        if selected:
            return selected
        return _top_answerish(proposals, limit=max(3, min(12, len(expected_labels))))

    return _top_answerish(proposals, limit=8)


def _top_answerish(proposals: list[AnswerProposal], *, limit: int) -> list[AnswerProposal]:
    ranked = sorted(proposals, key=lambda proposal: (_answerish_score(proposal), proposal.score), reverse=True)
    return ranked[:limit]


def _answerish_score(proposal: AnswerProposal) -> int:
    value = proposal.answer.value
    score = proposal.score
    if "=" in value:
        score += 4
    if NUMERIC_RE.match(value.strip()):
        score += 3
    if proposal.answer.unit:
        score += 2
    lowered = proposal.evidence.lower()
    if "numerical value" in lowered or "finally" in lowered or "hence" in lowered:
        score += 1
    if "marking scheme" in lowered or "marker scheme" in lowered:
        score -= 3
    if len(value) > 250:
        score -= 2
    return score


def _fallback_answer_proposals(item: FinalItem) -> list[AnswerProposal]:
    text = item.official_solution
    proposals = []
    math_values = []
    for value in DISPLAY_MATH_RE.findall(text):
        cleaned = _clean_value(value)
        if _valid_answer_value(cleaned):
            math_values.append(cleaned)
    for value in INLINE_MATH_RE.findall(text):
        cleaned = _clean_value(value)
        if _valid_answer_value(cleaned) and ("=" in cleaned or NUMERIC_RE.match(cleaned)):
            math_values.append(cleaned)
    for value in math_values[-8:]:
        proposals.append(_proposal(value, score=1, evidence="fallback: late solution math", subproblem_id=None))
    if proposals:
        return _dedupe_proposals(proposals)

    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped and not _is_noise_line(stripped) and not _is_marking_heading(stripped):
            return [_proposal(stripped, score=1, evidence="fallback: final non-empty solution line", subproblem_id=None)]
    return []


def _draft_status(item: FinalItem, selected: list[AnswerProposal], proposals: list[AnswerProposal]) -> str:
    if not selected:
        return "draft_missing_answer"
    expected = _expected_answer_count(item.question)
    if expected and len({proposal.answer.subproblem_id for proposal in selected if proposal.answer.subproblem_id}) < expected:
        return "draft_incomplete_needs_audit"
    if any(proposal.score < 5 for proposal in selected):
        return "draft_low_confidence_needs_audit"
    return "draft_extracted_needs_audit"


def _has_answer_coverage(proposals: list[AnswerProposal], expected_answers: int) -> bool:
    if expected_answers <= 1:
        return bool(proposals)
    distinct_subparts = {proposal.answer.subproblem_id for proposal in proposals if proposal.answer.subproblem_id}
    covered = len(distinct_subparts) if distinct_subparts else len(proposals)
    return covered >= expected_answers


def _review_reason(proposals: list[AnswerProposal], expected_answers: int) -> str:
    if not proposals:
        return "no high-confidence rule-verifiable answer candidates"
    return f"answer coverage too low for {expected_answers} expected sub-answers"


def _math_values(line: str) -> list[str]:
    values = [match.strip() for match in DISPLAY_MATH_RE.findall(line)]
    line_without_display = DISPLAY_MATH_RE.sub("", line)
    for match in INLINE_MATH_RE.findall(line_without_display):
        if _inline_math_can_be_answer(line_without_display, match):
            values.append(match.strip())
    return [_clean_value(value) for value in values if _valid_answer_value(value)]


def _answer_colon_values(line: str) -> list[tuple[str, str | None]]:
    matches = re.finditer(r"\banswer\s*:\s*(?P<value>[^,;]+?)(?:\s+unit\s*:\s*(?P<unit>[^,;]+))?$", line, re.I)
    return [(_clean_value(match.group("value")), _clean_unit(match.group("unit"))) for match in matches]


def _context(lines: list[str], index: int) -> str:
    start = max(0, index - 2)
    end = min(len(lines), index + 2)
    return "\n".join(line.strip() for line in lines[start:end] if line.strip())


def _context_score(context: str) -> int:
    lowered = context.lower()
    if "example of" in lowered:
        return 0
    score = 0
    if ANSWER_CUE_RE.search(lowered):
        score += 3
    if "final answer" in lowered or re.search(r"\banswer\s*:", lowered) or "boxed" in lowered:
        score += 2
    if "required" in lowered or "it follows" in lowered or "finally" in lowered:
        score += 1
    if WEAK_CUE_RE.search(lowered):
        score -= 1
    return score


def _nearest_subproblem_id(lines: list[str], index: int) -> str | None:
    for line in reversed(lines[max(0, index - 10) : index + 1]):
        match = SECTION_RE.match(line.strip())
        if match:
            return _normalize_subproblem_id(match.group("label"))
    return None


def _normalize_subproblem_id(value: str) -> str:
    text = re.sub(r"\*+", "", value.strip().lower())
    text = re.sub(r"\s+", "_", text)
    text = text.replace(")", "")
    return text


def _answer_kind(value: str, answer_type_hint: str | None = None) -> tuple[str, str]:
    hint = (answer_type_hint or "").strip().lower()
    if "numerical" in hint or "numeric" in hint:
        return "numeric", "numeric"
    if "symbolic" in hint or "expression" in hint:
        return "symbolic", "sympy"
    if "multiple" in hint or "choice" in hint:
        return "multiple_choice", "mcq"
    stripped = value.strip()
    if "=" in stripped:
        return "symbolic", "sympy"
    if NUMERIC_RE.match(stripped):
        return "numeric", "numeric"
    if re.fullmatch(r"[A-Ea-e]", stripped) and len(stripped) == 1:
        return "multiple_choice", "mcq"
    if any(token in stripped for token in ["=", "\\frac", "^", "_", "\\sqrt", "\\sin", "\\cos", "\\tan", "\\langle"]):
        return "symbolic", "sympy"
    return "string", "string"


def _split_unit(value: str) -> tuple[str, str | None]:
    text = value.strip()
    match = re.match(r"^(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?P<unit>[A-Za-z][A-Za-z/\^\-0-9 ]+)$", text)
    if not match:
        return text, None
    return match.group("number"), _clean_unit(match.group("unit"))


def _clean_value(value: str) -> str:
    text = value.strip()
    text = re.sub(r"^\(\d+(?:\.\d+)?\)\s*", "", text)
    text = text.rstrip(".,;")
    return text.strip()


def _clean_unit(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().rstrip(".,;")
    return cleaned or None


def _valid_answer_value(value: str) -> bool:
    cleaned = _clean_value(value)
    if not cleaned or len(cleaned) > 500:
        return False
    if re.fullmatch(r"[A-Za-z](?:_\d+)?|\\?[A-Za-z]+(?:_\d+)?", cleaned):
        return False
    if cleaned.startswith("!") or cleaned.lower() in {"image", "fig", "figure"}:
        return False
    return True


def _inline_math_can_be_answer(line: str, value: str) -> bool:
    lowered = line.lower()
    stripped = value.strip()
    if re.search(r"\banswer\s*:", lowered):
        return True
    if line.strip().endswith(":"):
        return False
    return "=" in stripped or NUMERIC_RE.match(stripped) is not None


def _is_noise_line(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("<!-- page:")


def _normalize_value(value: str) -> str:
    return re.sub(r"\s+", "", value.lower())


def _trim(value: str, *, limit: int = 400) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def answers_from_key_row(raw: dict[str, Any]) -> list[Answer]:
    return [answer_from_dict(answer) for answer in raw.get("answers", [])]
