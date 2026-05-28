from __future__ import annotations

import json
import re
from typing import Any

from .answer_extraction import answer_proposals_to_review
from .gemini_extract import DEFAULT_MODEL, generate_gemini_content
from .schema import FinalItem, to_dict

JUDGE_PROMPT = """You are curating physics RLVR training rows from OCR text.

Return only JSON with this schema:
{
  "items": [
    {
      "subproblem_id": "A.1 or null",
      "keep": true,
      "reason": "short audit note",
      "shared_context": "context needed to answer the question",
      "question": "single full question",
      "official_solution": "solution text supporting the answer",
      "answers": [
        {
          "value": "verifiable answer value",
          "unit": null,
          "answer_type": "numeric|symbolic|expression|multiple_choice|multi_select|set|tuple|interval",
          "tolerance": 0.05,
          "verifier": "numeric|sympy|expression|mcq|multi_select|set|tuple|interval",
          "equivalent_forms": [],
          "subproblem_id": "same label or null"
        }
      ]
    }
  ]
}

Keep only rows with a complete question and answer pair that can be checked by a deterministic verifier.
Reject proof-only, explanation-only, drawing-only, qualitative, ambiguous, or diagram-dependent prompts.
Use the candidate answers as evidence, but correct obvious OCR formatting when needed.
Do not invent answers that are not supported by the solution text."""


def judge_subproblems_with_gemini(
    parent: FinalItem,
    candidate_rows: list[dict[str, Any]],
    *,
    model_name: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    payload = {
        "parent": {
            "problem_id": parent.problem_id,
            "source": parent.source,
            "year": parent.year,
            "problem_number": parent.problem_number,
            "problem_text": parent.problem_text,
            "official_solution": parent.official_solution,
        },
        "candidate_rows": candidate_rows,
    }
    text = generate_gemini_content(
        model_name,
        [
            {"text": JUDGE_PROMPT},
            {"text": json.dumps(payload, ensure_ascii=True)},
        ],
        max_output_tokens=8192,
        temperature=0.0,
    )
    return list(_json_object(text).get("items", []))


def candidate_row_for_judge(item: FinalItem, proposals: list[Any]) -> dict[str, Any]:
    return {
        "subproblem_id": item.subproblem_id,
        "shared_context": item.shared_context,
        "question": item.question,
        "official_solution": item.official_solution,
        "heuristic_answers": answer_proposals_to_review(proposals, limit=12),
        "current_item": to_dict(item),
    }


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    return json.loads(stripped)
