from __future__ import annotations

import random
import re
from typing import Callable

from prime_rl.optimizer.gepa.config import OperatorsConfig


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars]


def op_tighten_instruction(prompt: str) -> str:
    return re.sub(r"(?i)please |kindly ", "", prompt)


def op_enforce_format_tags(prompt: str) -> str:
    if "<final_answer>" not in prompt:
        prompt += "\nAlways put the final answer inside <final_answer>...</final_answer>."
    return prompt


def op_add_reasoning_hint(prompt: str) -> str:
    if "</think>" not in prompt and "<think>" not in prompt:
        prompt += "\nThink step-by-step inside <think>...</think> before writing the final answer."
    return prompt


def op_remove_vagueness(prompt: str) -> str:
    prompt = re.sub(r"(?i)try to|attempt to|maybe|possibly|could you ", "", prompt)
    return prompt


def mutate(prompt: str, cfg: OperatorsConfig, rng: random.Random) -> str:
    ops: list[Callable[[str], str]] = [
        op_tighten_instruction,
        op_enforce_format_tags,
        op_add_reasoning_hint,
        op_remove_vagueness,
    ]
    num_ops = 1 + (rng.random() < 0.5)
    for _ in range(num_ops):
        op = rng.choice(ops)
        prompt = op(prompt)
    return _truncate(prompt, cfg.max_prompt_chars)


def crossover(a: str, b: str, cfg: OperatorsConfig, rng: random.Random) -> str:
    if len(a) < 16 or len(b) < 16:
        return a if rng.random() < 0.5 else b
    ia = rng.randrange(len(a) // 4, 3 * len(a) // 4)
    ib = rng.randrange(len(b) // 4, 3 * len(b) // 4)
    child = a[:ia] + "\n" + b[ib:]
    return _truncate(child, cfg.max_prompt_chars)


