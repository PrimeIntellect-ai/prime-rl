from __future__ import annotations

import re

import verifiers as vf
from datasets import Dataset

TRAIN_EXAMPLES = [
    {"question": "Solve: 3 + 5 =", "answer": "8"},
    {
        "question": "If all birds can fly and penguins are birds, can penguins fly",
        "answer": "No",
    },
]

EVAL_EXAMPLES = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Calculate: 12 * 7 =", "answer": "84"},
    {"question": 'Is the statement "All cats are mammals" true or false?', "answer": "True"},
    {"question": "What comes next in the sequence: 2, 4, 6, 8, ?", "answer": "10"},
    {"question": 'Translate "Hello" to Spanish:', "answer": "Hola"},
    {"question": "What is 15% of 200?", "answer": "30"},
    {"question": "Name one primary color:", "answer": "Red"},
    {"question": "How many days are in a week?", "answer": "7"},
]


def _completion_text(completion: str | list[object], **_: object) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                role = item.get("role")
                if role in (None, "assistant"):
                    parts.append(str(item.get("content", "")))
                continue

            role = getattr(item, "role", None)
            content = getattr(item, "content", None)
            if content is not None and role in (None, "assistant"):
                parts.append(str(content))
                continue

            text = str(item)
            match = re.search(r"content=(['\"])(.*?)\1", text, flags=re.DOTALL)
            if match:
                parts.append(match.group(2).encode("utf-8").decode("unicode_escape"))
            elif text:
                parts.append(text)
        return "".join(parts).strip()
    return str(completion).strip()


def raw_concision_reward(completion: str | list[object], answer: str, **_: object) -> float:
    text = _completion_text(completion)
    return -float(abs(len(text) - len(answer.strip())))


def normalized_concision_reward(
    completion: str | list[object],
    answer: str,
    reward_floor: float = -2000.0,
    **_: object,
) -> float:
    raw = raw_concision_reward(completion=completion, answer=answer)
    return max(0.0, min(1.0, (raw - reward_floor) / (0.0 - reward_floor)))


def completion_length(completion: str | list[object], **_: object) -> float:
    return float(len(_completion_text(completion)))


def _to_prompt_row(row: dict[str, str]) -> dict[str, object]:
    return {
        "prompt": [{"role": "user", "content": row["question"]}],
        "answer": row["answer"],
    }


def _dataset_from_rows(rows: list[dict[str, str]]) -> Dataset:
    return Dataset.from_list([_to_prompt_row(row) for row in rows])


def load_environment(**kwargs) -> vf.Environment:
    reward_floor = float(kwargs.get("reward_floor", -2000.0))

    def normalized_with_floor(completion, answer, **kw):
        return normalized_concision_reward(
            completion=completion,
            answer=answer,
            reward_floor=reward_floor,
            **kw,
        )

    rubric = vf.Rubric(
        funcs=[
            raw_concision_reward,
            normalized_with_floor,
            completion_length,
        ],
        weights=[1.0, 0.0, 0.0],
    )

    return vf.SingleTurnEnv(
        dataset=_dataset_from_rows(TRAIN_EXAMPLES),
        eval_dataset=_dataset_from_rows(EVAL_EXAMPLES),
        rubric=rubric,
        env_id="concision-gemma",
        env_args=kwargs,
        sampling_args={"n": 1, "extra_body": {}},
    )
