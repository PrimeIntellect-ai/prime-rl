from __future__ import annotations

import random
from typing import Sequence

from prime_rl.optimizer.gepa.config import OperatorsConfig


REFLECT_TEMPLATE = (
    "You are optimizing a system instruction for a model on a benchmark.\n"
    "Given a list of failure examples (short summaries), propose concise edits to the instruction\n"
    "that will improve accuracy without adding verbosity or vagueness.\n"
    "Return only the edited instruction text. Keep it under the specified character limit."
)


def reflect(prompt: str, failures: Sequence[str], cfg: OperatorsConfig, rng: random.Random) -> str:
    # Placeholder: lightweight heuristic reflection combining failures into a short clause
    if not failures:
        return prompt
    clause = "; ".join(failures[:3])
    edited = (
        prompt
        + "\nConstraints: Avoid previous mistakes such as: "
        + clause
        + ". Be explicit, deterministic, and adhere to required answer format."
    )
    if len(edited) > cfg.max_prompt_chars:
        edited = edited[: cfg.max_prompt_chars]
    return edited


