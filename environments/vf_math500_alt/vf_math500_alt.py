import json
from typing import Dict

import verifiers as vf
from datasets import load_dataset
from math_verify import parse, verify


def math_verify_reward_function(completion: str, verification_info: Dict):
    ground_truth = verification_info["ground_truth"]
    if isinstance(ground_truth, (str, float, int)):
        ground_truth = [ground_truth]

    # We always take the final solution
    if "</think>" in completion:
        completion = completion.split("</think>")[1]

    # 0 in case parsing cannot be completed
    try:
        math_verify_parsed = parse(completion, parsing_timeout=5)
    except BaseException:
        return 0.0

    # 0 if parsing is problematic
    if len(math_verify_parsed) < 2:
        return 0.0

    # We now fallback to semantic verification
    for gt in ground_truth:
        try:
            if verify(
                parse(f"\\boxed{{{gt}}}", parsing_timeout=5),
                math_verify_parsed,
                timeout_seconds=5,
            ):
                return 1.0
        except BaseException:
            continue

    # Very unlikely to be correct after the above matches
    return 0.0


def load_environment() -> vf.SingleTurnEnv:
    eval_dataset = load_dataset("PrimeIntellect/MATH-500", split="train").map(
        lambda example: {
            "prompt": [{"role": "user", "content": example["prompt"]}],
            "answer": "Not used",
            "info": json.loads(example["verification_info"]),
        }
    )
    eval_dataset = eval_dataset.remove_columns(["problem_id", "task_type", "verification_info"])
    assert eval_dataset.column_names == ["prompt", "answer", "info"], eval_dataset.column_names

    def correct_answer_reward_fn(completion, info) -> float:
        completion_text = completion[-1]["content"]
        reward = math_verify_reward_function(completion_text, info)
        return reward

    rubric = vf.Rubric(
        funcs=[
            correct_answer_reward_fn,
        ],
        weights=[1.0],
    )

    vf_env = vf.SingleTurnEnv(
        eval_dataset=eval_dataset,
        rubric=rubric,
    )
    return vf_env
