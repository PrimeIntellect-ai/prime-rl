"""Deterministic Verifiers fixture for Memento boundary-token RL smoke tests."""

from __future__ import annotations

from typing import Any

from datasets import Dataset
import verifiers as vf


BOUNDARY_HISTORY = (
    "<|block_start|>"
    "I add 15 and 27. 15 + 27 = 42."
    "<|block_end|>"
    "<|summary_start|>"
    "15 + 27 = 42"
    "<|summary_end|>"
    "\nThe answer is 42."
)


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for message in completion:
            if isinstance(message, dict):
                parts.append(str(message.get("content", "")))
            else:
                parts.append(str(message))
        return "\n".join(parts)
    return str(completion)


def exact_answer_reward(prompt: Any, completion: Any, answer: str | None = None, **_: Any) -> float:
    text = _completion_text(completion)
    if answer is None:
        return 0.0
    return 1.0 if str(answer) in text else 0.0


def _make_prompt(example_idx: int, answer: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": "What is 15 + 27?",
        },
        {
            "role": "assistant",
            "content": BOUNDARY_HISTORY,
        },
        {
            "role": "user",
            "content": (
                "Using the previous result, what is 42 * 2? "
                f"Return only the number {answer}. Problem id: {example_idx}."
            ),
        },
    ]


def load_environment(num_examples: int = 8, answer: str = "84", **_: Any):
    dataset = Dataset.from_list(
        [
            {
                "prompt": _make_prompt(i, answer),
                "answer": answer,
                "info": {"fixture": "memento-boundary", "example_idx": i},
            }
            for i in range(num_examples)
        ]
    )
    rubric = vf.Rubric(funcs=[exact_answer_reward], weights=[1.0])
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
