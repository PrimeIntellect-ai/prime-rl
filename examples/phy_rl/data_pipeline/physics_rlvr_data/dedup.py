from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

from .io import ensure_parent
from .schema import FinalItem


def canonical_text(text: str) -> str:
    lowered = text.lower()
    asciiish = "".join(char if char.isalnum() else " " for char in lowered)
    return re.sub(r"\s+", " ", asciiish).strip()


def problem_hash(item: FinalItem) -> str:
    payload = "\n".join(
        [
            canonical_text(item.problem_text),
            canonical_text(item.shared_context),
            canonical_text(item.question),
            canonical_answers(item),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_answers(item: FinalItem) -> str:
    parts = []
    for answer in item.answers:
        parts.append(
            "|".join(
                [
                    canonical_text(answer.value),
                    canonical_text(answer.unit or ""),
                    answer.answer_type,
                    answer.verifier,
                ]
            )
        )
    return "\n".join(sorted(parts))


def deduplicate(items: list[FinalItem], report_path: Path) -> list[FinalItem]:
    ensure_parent(report_path)
    seen: dict[str, FinalItem] = {}
    kept: list[FinalItem] = []
    with report_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["duplicate_problem_id", "kept_problem_id", "hash"],
        )
        writer.writeheader()
        for item in items:
            digest = problem_hash(item)
            if digest in seen:
                writer.writerow(
                    {
                        "duplicate_problem_id": item.problem_id,
                        "kept_problem_id": seen[digest].problem_id,
                        "hash": digest,
                    }
                )
                continue
            seen[digest] = item
            kept.append(item)
    return kept
