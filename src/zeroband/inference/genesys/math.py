from typing import Dict
from zeroband.inference.genesys.math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy


def compute_math_reward(completion: str, verification_info: Dict):
    model_response = completion
    ground_truths = verification_info["ground_truth"]

    # Extract solution.
    if "</think>" in model_response:
        model_solution = model_response.split("</think>")[1]
    else:
        return 0

    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return 0

    if ground_truths is None:
        return 0

    # Convert single answer to list for uniform processing
    if isinstance(ground_truths, (str, float, int)):
        ground_truths = [ground_truths]

    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)

    if not processed_ground_truths:
        return 0

    # Check against all possible correct answers
    for ground_truth in processed_ground_truths:
        is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
        if is_correct:
            return 1

    return 0
