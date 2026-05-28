from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from .answer_extraction import (
    AnswerProposal,
    answer_proposals_to_review,
    dedupe_answer_proposals,
    expected_subproblem_ids,
    extract_answer_proposals,
    fallback_answer_proposals,
    normalize_subproblem_id,
    top_answerish_proposals,
)
from .filters import admissibility_rejection
from .gemini_judge import candidate_row_for_judge, judge_subproblems_with_gemini
from .io import read_jsonl, write_jsonl
from .schema import Answer, FinalItem, RejectedItem, answer_from_dict, final_item_from_dict, to_dict
from .verifiers import verify_answer

SUBQUESTION_RE = re.compile(
    r"(?m)^(?P<prefix>\s*)(?P<label>[A-E]\.\d+|Task\s+\d+[a-z]?|\d+[a-z]\))\b(?P<rest>.*)$",
    flags=re.IGNORECASE,
)
INITIAL_SUBQUESTION_LABEL_RE = re.compile(r"^\s*(?:[A-E]\.\d+|Task\s+\d+[a-z]?|\d+[a-z]\))\b", re.I)
ANSWER_DIRECTIVE_RE = re.compile(
    r"\b(determine|calculate|compute|find|derive|express|evaluate|estimate|give|obtain)\b",
    flags=re.IGNORECASE,
)
UNVERIFIABLE_RE = re.compile(
    r"\b(explain|discuss|comment|describe qualitatively|sketch|draw|plot|prove|show that)\b",
    flags=re.IGNORECASE,
)


def build_rlvr_subproblems(
    input_path: Path,
    verified_path: Path,
    review_path: Path,
    rejected_path: Path,
    *,
    min_score: int = 3,
    judge_model: str | None = None,
) -> tuple[int, int, int]:
    verified: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for raw in read_jsonl(input_path):
        if "rejection_reason" in raw:
            rejected.append(raw)
            continue

        parent = final_item_from_dict(raw)
        subquestions = split_problem_into_subquestions(parent)
        solution_blocks = _solution_blocks_for_subquestions(parent.official_solution, subquestions)
        if judge_model is not None:
            judged_verified, judged_review, judged_rejected = _build_judged_parent_rows(
                parent,
                subquestions,
                solution_blocks,
                judge_model=judge_model,
            )
            verified.extend(judged_verified)
            review.extend(judged_review)
            rejected.extend(judged_rejected)
            continue

        for subquestion in subquestions:
            solution_alignment = _solution_alignment_method(subquestion, solution_blocks)
            item = _subproblem_item(parent, subquestion, solution_blocks)
            proposals = extract_answer_proposals(item)
            if not proposals:
                proposals = fallback_answer_proposals(item)
            selected = _select_verifier_safe_answers(item, proposals, min_score=min_score)
            expected_answers = _expected_answer_count(item.question)

            if not item.official_solution.strip():
                rejected.append(to_dict(_reject(item, "missing_solution_block", "No matching solution block")))
                continue
            if _is_unverifiable_prompt(item.question):
                rejected.append(to_dict(_reject(item, "unverifiable_prompt", "Subquestion asks for proof/explanation/drawing")))
                continue
            if not selected:
                review.append(_review_row(item, "no_verifier_safe_answer", proposals, expected_answers))
                continue
            if expected_answers and not _has_selected_answer_coverage(selected, expected_answers):
                review.append(_review_row(item, "answer_coverage_too_low", proposals, expected_answers, selected))
                continue
            if solution_alignment == "sequential":
                review.append(
                    _review_row(
                        item,
                        "sequential_solution_alignment_needs_review",
                        proposals,
                        expected_answers,
                        selected,
                    )
                )
                continue

            admitted = replace(item, answers=selected)
            rejection = admissibility_rejection(admitted)
            if rejection is not None:
                reason, detail = rejection
                if reason in {"manual_review_needed", "qualitative_only"}:
                    review.append(_review_row(admitted, reason, proposals, expected_answers, selected, detail=detail))
                else:
                    rejected.append(to_dict(_reject(admitted, reason, detail)))
                continue

            verified.append(to_dict(admitted))

    return (
        write_jsonl(verified_path, verified),
        write_jsonl(review_path, review),
        write_jsonl(rejected_path, rejected),
    )


def _build_judged_parent_rows(
    parent: FinalItem,
    subquestions: list[dict[str, str | None]],
    solution_blocks: dict[str, tuple[str, str]],
    *,
    judge_model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_items = []
    candidate_rows = []
    for subquestion in subquestions:
        item = _subproblem_item(parent, subquestion, solution_blocks)
        proposals = extract_answer_proposals(item)
        if not proposals:
            proposals = fallback_answer_proposals(item)
        candidate_items.append(item)
        candidate_rows.append(candidate_row_for_judge(item, proposals))

    by_label = {item.subproblem_id: item for item in candidate_items}
    allow_judge_labels = len(candidate_items) == 1 and candidate_items[0].subproblem_id is None
    if allow_judge_labels:
        by_label[None] = candidate_items[0]

    verified: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    try:
        judged_rows = judge_subproblems_with_gemini(parent, candidate_rows, model_name=judge_model)
    except (RuntimeError, ValueError) as exc:
        review.append(_judged_review(parent, {}, "judge_failed", detail=str(exc)))
        return verified, review, rejected

    for judged in judged_rows:
        subproblem_id = judged.get("subproblem_id")
        if isinstance(subproblem_id, str):
            subproblem_id = normalize_subproblem_id(subproblem_id)
        template = by_label.get(subproblem_id)
        if template is None and allow_judge_labels:
            template = candidate_items[0]
        if template is None and judged.get("keep", False) and judged.get("question") and judged.get("answers"):
            template = candidate_items[0]
        if template is None:
            review.append(_judged_review(parent, judged, "unknown_subproblem_id"))
            continue
        if not judged.get("keep", False):
            review.append(_judged_review(template, judged, "judge_rejected"))
            continue

        answers = [_judged_answer(answer, subproblem_id) for answer in judged.get("answers", [])]
        admitted = _judged_item(template, judged, answers, subproblem_id=subproblem_id)
        unsafe_answers = [answer for answer in admitted.answers if not _is_verifier_safe_answer(answer)]
        if unsafe_answers:
            review.append(_judged_review(admitted, judged, "judge_answer_not_verifier_safe"))
            continue
        rejection = admissibility_rejection(admitted)
        if rejection is not None:
            reason, detail = rejection
            if reason in {"manual_review_needed", "qualitative_only"}:
                review.append(_judged_review(admitted, judged, reason, detail=detail))
            else:
                rejected.append(to_dict(_reject(admitted, reason, detail)))
            continue
        verified.append(to_dict(admitted))

    return verified, review, rejected


def _judged_answer(raw: dict[str, Any], subproblem_id: str | None) -> Answer:
    answer = answer_from_dict(raw)
    return replace(answer, subproblem_id=subproblem_id)


def _judged_item(
    template: FinalItem,
    judged: dict[str, Any],
    answers: list[Answer],
    *,
    subproblem_id: str | None,
) -> FinalItem:
    shared_context = str(judged.get("shared_context") or template.shared_context).strip()
    question = str(judged.get("question") or template.question).strip()
    problem_id = template.problem_id
    if subproblem_id != template.subproblem_id:
        problem_id = _subproblem_problem_id(template.problem_id, subproblem_id)
    return FinalItem(
        problem_id=problem_id,
        source=template.source,
        competition=template.competition,
        year=template.year,
        problem_number=template.problem_number,
        subproblem_id=subproblem_id,
        problem_text="\n\n".join(part for part in [shared_context, question] if part).strip(),
        shared_context=shared_context,
        question=question,
        official_solution=str(judged.get("official_solution") or template.official_solution).strip(),
        answers=answers,
        requires_diagram=template.requires_diagram,
        language=template.language,
        split=template.split,
        provenance=template.provenance,
    )


def _judged_review(
    item: FinalItem,
    judged: dict[str, Any],
    reason: str,
    *,
    detail: str | None = None,
) -> dict[str, Any]:
    return {
        "problem_id": item.problem_id,
        "source": item.source,
        "year": item.year,
        "problem_number": item.problem_number,
        "subproblem_id": item.subproblem_id,
        "status": "needs_review",
        "reason": reason,
        "detail": detail or judged.get("reason"),
        "question": judged.get("question") or item.question,
        "shared_context": judged.get("shared_context") or item.shared_context,
        "solution_excerpt": _trim(str(judged.get("official_solution") or item.official_solution), limit=1200),
        "selected_answers": judged.get("answers", []),
    }


def split_problem_into_subquestions(parent: FinalItem) -> list[dict[str, str | None]]:
    text = parent.question.strip()
    matches = list(SUBQUESTION_RE.finditer(text))
    if not matches:
        return [{"subproblem_id": parent.subproblem_id, "shared_context": parent.shared_context, "question": text}]

    shared_context = text[: matches[0].start()].strip()
    subquestions = []
    carry_context = ""
    for index, match in enumerate(matches):
        label = normalize_subproblem_id(match.group("label"))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start() : end].strip()
        question, trailing_context = _split_labeled_question_block(block)
        context = "\n\n".join(part for part in [shared_context, carry_context] if part).strip()
        subquestions.append({"subproblem_id": label, "shared_context": context, "question": question})
        carry_context = trailing_context
    return subquestions


def _split_labeled_question_block(block: str) -> tuple[str, str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", block) if part.strip()]
    if len(paragraphs) <= 1:
        return block, ""
    return paragraphs[0], "\n\n".join(paragraphs[1:])


def _subproblem_item(
    parent: FinalItem,
    subquestion: dict[str, str | None],
    solution_blocks: dict[str, tuple[str, str]],
) -> FinalItem:
    subproblem_id = subquestion["subproblem_id"]
    question = str(subquestion["question"] or "").strip()
    shared_context = str(subquestion["shared_context"] or "").strip()
    problem_text = "\n\n".join(part for part in [shared_context, question] if part).strip()
    solution = solution_blocks.get(str(subproblem_id), ("", "missing"))[0]
    if not solution_blocks and subproblem_id is None and len(question) > 0:
        solution = parent.official_solution
    return FinalItem(
        problem_id=_subproblem_problem_id(parent.problem_id, subproblem_id),
        source=parent.source,
        competition=parent.competition,
        year=parent.year,
        problem_number=parent.problem_number,
        subproblem_id=subproblem_id,
        problem_text=problem_text,
        shared_context=shared_context,
        question=question,
        official_solution=solution,
        answers=[],
        requires_diagram=parent.requires_diagram,
        language=parent.language,
        split=parent.split,
        provenance=parent.provenance,
    )


def _solution_blocks_for_subquestions(text: str, subquestions: list[dict[str, str | None]]) -> dict[str, tuple[str, str]]:
    labels = [str(subquestion["subproblem_id"]) for subquestion in subquestions if subquestion["subproblem_id"]]
    explicit = _explicit_solution_blocks(text)
    if explicit:
        return explicit
    heading_blocks = _heading_solution_blocks(text, labels)
    if heading_blocks:
        return heading_blocks
    sequential = _sequential_solution_blocks(text, labels)
    if sequential:
        return sequential
    return {}


def _explicit_solution_blocks(text: str) -> dict[str, tuple[str, str]]:
    from .answer_extraction import solution_blocks_by_subproblem

    return {label: (block, "explicit") for label, block in solution_blocks_by_subproblem(text).items()}


def _heading_solution_blocks(text: str, labels: list[str]) -> dict[str, tuple[str, str]]:
    if not labels:
        return {}
    matches = []
    for match in SUBQUESTION_RE.finditer(text):
        label = normalize_subproblem_id(match.group("label"))
        if label in labels:
            matches.append((label, match))
    if not matches:
        return {}
    blocks = {}
    for index, (label, match) in enumerate(matches):
        end = matches[index + 1][1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start() : end].strip()
        if block:
            blocks[label] = (block, "heading")
    return blocks


def _sequential_solution_blocks(text: str, labels: list[str]) -> dict[str, tuple[str, str]]:
    if not labels or not text.strip():
        return {}
    chunks = _solution_chunks(text)
    if len(chunks) < 2:
        return {}
    if len(chunks) >= len(labels):
        return {label: (chunks[index], "sequential") for index, label in enumerate(labels)}
    return {}


def _solution_alignment_method(
    subquestion: dict[str, str | None],
    solution_blocks: dict[str, tuple[str, str]],
) -> str:
    subproblem_id = subquestion["subproblem_id"]
    if subproblem_id is None:
        return "full_problem"
    return solution_blocks.get(str(subproblem_id), ("", "missing"))[1]


def _solution_chunks(text: str) -> list[str]:
    page_chunks = [chunk.strip() for chunk in re.split(r"(?m)^<!-- page:\d+ -->\s*$", text) if chunk.strip()]
    if len(page_chunks) > 1:
        return page_chunks
    heading_pattern = re.compile(
        r"(?mi)^\s*(?:first|second|third|fourth|fifth)\s+(?:approach|part|task|problem)\b|^\s*(?:part|problem)\s+[A-E0-9]\b"
    )
    matches = list(heading_pattern.finditer(text))
    chunks = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        chunk = text[match.start() : end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _select_verifier_safe_answers(item: FinalItem, proposals: list[AnswerProposal], *, min_score: int) -> list[Answer]:
    candidates = [
        proposal
        for proposal in dedupe_answer_proposals(proposals)
        if proposal.score >= min_score
        and _is_verifier_safe_answer(proposal.answer)
        and _is_targeted_or_boxed(item.question, proposal)
        and _is_allowed_for_auto_admission(item, proposal)
    ]
    expected_answers = _expected_answer_count(item.question)
    limit = max(1, min(4, expected_answers or 1))
    ranked = sorted(
        top_answerish_proposals(candidates, limit=max(limit * 3, limit)),
        key=lambda proposal: _subquestion_answer_score(item.question, proposal),
        reverse=True,
    )
    return [proposal.answer for proposal in ranked[:limit]]


def _is_allowed_for_auto_admission(item: FinalItem, proposal: AnswerProposal) -> bool:
    if "identify the regime" in item.question.lower():
        return False
    if "exponent" in item.question.lower() and not re.search(r"\bgamma\b|\\gamma|^\s*[+-]?\d+(?:\.\d+)?\s*$", proposal.answer.value):
        return False
    return proposal.score >= 6 or proposal.evidence == "boxed answer" or "answer:" in proposal.evidence.lower()


def _is_verifier_safe_answer(answer: Answer) -> bool:
    if answer.verifier in {"string", "string_exact", "custom"}:
        return False
    if answer.answer_type in {"string", "string_exact"}:
        return False
    if answer.verifier in {"numeric", "sympy", "expression", "mcq", "multi_select", "set", "tuple", "interval"}:
        return verify_answer(answer.value, answer)
    return False


def _expected_answer_count(question: str) -> int:
    body = INITIAL_SUBQUESTION_LABEL_RE.sub("", question, count=1).strip()
    explicit = len(expected_subproblem_ids(body))
    if explicit:
        return explicit
    directives = len(ANSWER_DIRECTIVE_RE.findall(body))
    numeric_prompts = len(
        re.findall(
            r"\b(numerical value|in terms of|value of|values of|radius|velocity|energy|mass|frequency|wavelength|angular frequency)\b",
            body,
            re.I,
        )
    )
    if "equation of motion" in body.lower() and "angular frequency" in body.lower():
        return 2
    return max(1, min(4, directives, max(1, numeric_prompts))) if directives else 1


def _has_selected_answer_coverage(answers: list[Answer], expected_answers: int) -> bool:
    if len(answers) < expected_answers:
        return False
    if expected_answers <= 1:
        return True
    return len({_answer_slot_key(answer) for answer in answers}) >= expected_answers


def _answer_slot_key(answer: Answer) -> str:
    value = answer.value.strip()
    if re.search(r"\\times\s*10|\\text\{|[0-9]+\.[0-9]+", value):
        return f"numeric:{value}"
    lhs_match = re.match(r"^\s*([A-Za-z](?:_\{?[A-Za-z0-9,]+\}?|_[A-Za-z0-9]+)?)\s*=", value)
    if lhs_match:
        return _latex_identifier_text(lhs_match.group(1))
    return value


def _is_targeted_or_boxed(question: str, proposal: AnswerProposal) -> bool:
    targets = _question_target_symbols(question)
    if not targets:
        return True
    if proposal.evidence == "boxed answer":
        return True
    if proposal.answer.answer_type in {"numeric", "numerical"} and re.search(r"\bnumerical value\b|\bcalculate\b|\bcompute\b", question, re.I):
        return True
    lhs_match = re.match(r"^\s*([A-Za-z](?:_\{?[A-Za-z0-9,]+\}?|_[A-Za-z0-9]+)?)\s*=", proposal.answer.value.strip())
    return bool(lhs_match and _latex_identifier_text(lhs_match.group(1)) in targets)


def _subquestion_answer_score(question: str, proposal: AnswerProposal) -> int:
    score = proposal.score
    value = proposal.answer.value.strip()
    evidence = proposal.evidence.lower()
    lhs_match = re.match(r"^\s*([A-Za-z](?:_\{?[A-Za-z0-9,]+\}?|_[A-Za-z0-9]+)?)\s*=", value)
    if lhs_match and _latex_identifier_text(lhs_match.group(1)) in _question_target_symbols(question):
        score += 8
    if proposal.evidence == "boxed answer":
        score += 6
    if any(cue in evidence for cue in ["hence", "therefore", "finally", "we get", "one finds", "which gives"]):
        score += 2
    if proposal.answer.unit:
        score += 1
    if len(value) > 180:
        score -= 2
    return score


def _question_target_symbols(question: str) -> set[str]:
    symbols = set()
    for sentence in re.split(r"(?<=[.;?])\s+", question):
        for match in ANSWER_DIRECTIVE_RE.finditer(sentence):
            segment = sentence[match.end() :]
            segment = re.split(r"\bin terms of\b|\busing\b|\bfrom\b|\bwith\b", segment, maxsplit=1, flags=re.I)[0]
            symbols.update(_math_identifiers(segment))
    show_match = re.search(r"\bshow that\b(?P<segment>.*?)(?:\.|$)", question, re.I)
    if show_match:
        symbols.update(_equation_lhs_identifiers(show_match.group("segment")))
    return symbols


def _math_identifiers(text: str) -> set[str]:
    symbols = set()
    for math_value in re.findall(r"\$(?!\$)(.*?)(?<!\$)\$", text):
        lhs_symbols = _equation_lhs_identifiers(math_value)
        if lhs_symbols:
            symbols.update(lhs_symbols)
            continue
        cleaned = _latex_identifier_text(math_value)
        if cleaned and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", cleaned):
            symbols.add(cleaned)
    return symbols


def _equation_lhs_identifiers(text: str) -> set[str]:
    symbols = set()
    for lhs in re.findall(r"([A-Za-z](?:_\{?[A-Za-z0-9,]+\}?|_[A-Za-z0-9]+)?(?:\([^)]*\))?)\s*=", text):
        cleaned = _latex_identifier_text(re.sub(r"\(.*\)$", "", lhs))
        if cleaned:
            symbols.add(cleaned)
    return symbols


def _latex_identifier_text(value: str) -> str:
    text = value.strip()
    text = re.sub(r"\\(?:vec|overrightarrow|mathbf|mathrm|text)\{([^}]*)\}", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\", "")
    text = re.sub(r"\s+", "", text)
    return text


def _is_unverifiable_prompt(question: str) -> bool:
    lowered = question.lower()
    if "show that" in lowered and ANSWER_DIRECTIVE_RE.search(question):
        return False
    return UNVERIFIABLE_RE.search(question) is not None


def _review_row(
    item: FinalItem,
    reason: str,
    proposals: list[AnswerProposal],
    expected_answers: int,
    selected: list[Answer] | None = None,
    *,
    detail: str | None = None,
) -> dict[str, Any]:
    return {
        "problem_id": item.problem_id,
        "source": item.source,
        "year": item.year,
        "problem_number": item.problem_number,
        "subproblem_id": item.subproblem_id,
        "status": "needs_review",
        "reason": reason,
        "detail": detail,
        "expected_answer_count": expected_answers,
        "selected_answer_count": len(selected or []),
        "candidate_count": len(proposals),
        "question": item.question,
        "solution_excerpt": _trim(item.official_solution, limit=1200),
        "selected_answers": [to_dict(answer) for answer in selected or []],
        "top_candidates": answer_proposals_to_review(proposals, limit=12),
    }


def _reject(item: FinalItem, reason: str, detail: str) -> RejectedItem:
    return RejectedItem(
        problem_id=item.problem_id,
        source=item.source,
        rejection_reason=reason,
        detail=detail,
        item=to_dict(item),
    )


def _subproblem_problem_id(parent_problem_id: str, subproblem_id: str | None) -> str:
    if not subproblem_id:
        return parent_problem_id
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", subproblem_id).strip("_").lower()
    return f"{parent_problem_id}__{clean}"


def _trim(value: str, *, limit: int) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
