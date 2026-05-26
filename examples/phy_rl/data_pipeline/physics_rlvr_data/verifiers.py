from __future__ import annotations

import math
import re
from collections.abc import Iterable

from .schema import Answer

_SCI_RE = re.compile(r"^(-?)\s*(\d+(?:\.\d+)?)\s*(?:\\times|x|\*)\s*10\^?\{?(-?\d+)\}?$")


def extract_boxed(text: str) -> list[str]:
    results: list[str] = []
    i = 0
    marker = r"\boxed{"
    while i < len(text):
        start_marker = text.find(marker, i)
        if start_marker == -1:
            break
        start = start_marker + len(marker)
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[start : j - 1].strip())
        i = start_marker + 1
    return results


def verify_prediction(prediction: str, answers: Iterable[Answer]) -> float:
    boxed = extract_boxed(prediction)
    answers_list = list(answers)
    if not boxed or not answers_list:
        return 0.0
    correct = 0
    unused = boxed.copy()
    for answer in answers_list:
        match = next((candidate for candidate in unused if verify_answer(candidate, answer)), None)
        if match is not None:
            correct += 1
            unused.remove(match)
    return correct / len(answers_list)


def verify_answer(prediction: str, answer: Answer) -> bool:
    values = [answer.value, *answer.equivalent_forms]
    if answer.verifier == "numeric" or answer.answer_type in {"numeric", "numerical"}:
        return any(_numeric_match(prediction, value, answer.tolerance) for value in values)
    if answer.verifier in {"sympy", "expression"} or answer.answer_type in {"symbolic", "expression"}:
        return any(_symbolic_match(prediction, value) for value in values)
    if answer.verifier == "mcq" or answer.answer_type == "multiple_choice":
        return _normalize_choice(prediction) == _normalize_choice(answer.value)
    if answer.verifier in {"set", "multi_select"} or answer.answer_type in {"set", "multi_select"}:
        return _normalize_set(prediction) == _normalize_set(answer.value)
    if answer.verifier in {"string", "string_exact"} or answer.answer_type in {"string", "string_exact"}:
        return _normalize_text(prediction) == _normalize_text(answer.value)
    return False


def _numeric_match(prediction: str, expected: str, tolerance: float | str | None) -> bool:
    pred_num = _parse_number(prediction)
    exp_num = _parse_number(expected)
    if pred_num is None or exp_num is None:
        return _normalize_text(prediction) == _normalize_text(expected)
    rel_tol = float(tolerance) if tolerance not in (None, "") else 0.05
    if exp_num == 0:
        return abs(pred_num) <= rel_tol
    return math.isclose(pred_num, exp_num, rel_tol=rel_tol, abs_tol=rel_tol * 1e-12)


def _parse_number(value: str) -> float | None:
    text = value.strip()
    text = re.sub(r"\\(?:text|mathrm|mbox)\{[^}]*\}", "", text)
    text = text.replace(r"\,", "").replace(",", "").strip()
    sci_match = _SCI_RE.match(text)
    if sci_match:
        sign, mantissa, exponent = sci_match.groups()
        multiplier = -1.0 if sign else 1.0
        return multiplier * float(mantissa) * 10 ** int(exponent)
    try:
        return float(text)
    except ValueError:
        return None


def _symbolic_match(prediction: str, expected: str) -> bool:
    pred_text = _strip_latex_wrappers(prediction)
    exp_text = _strip_latex_wrappers(expected)
    if pred_text == exp_text:
        return True
    try:
        from sympy import simplify  # noqa: PLC0415
        from sympy.parsing.latex import parse_latex  # noqa: PLC0415
    except ImportError:
        return False
    try:
        return bool(simplify(parse_latex(pred_text) - parse_latex(exp_text)) == 0)
    except Exception:
        return False


def _normalize_choice(value: str) -> str:
    text = _normalize_text(value)
    match = re.search(r"\b([a-e])\b", text)
    return match.group(1) if match else text


def _normalize_set(value: str) -> tuple[str, ...]:
    text = value.strip().strip("{}[]()")
    parts = [part.strip() for part in re.split(r"[,;]", text) if part.strip()]
    return tuple(sorted(_normalize_text(part) for part in parts))


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", _strip_latex_wrappers(value).lower()).strip()


def _strip_latex_wrappers(value: str) -> str:
    text = value.strip()
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = re.sub(r"\\(?:text|mathrm|mbox)\{([^}]*)\}", r"\1", text)
    return text.strip()
