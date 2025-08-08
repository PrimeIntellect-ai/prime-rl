from typing import Dict

from math_verify import parse, verify

from prime_rl.orchestrator.genesys.math_utils import (
    extract_answer,
    grade_answer_mathd,
    grade_answer_sympy,
)


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
    except Exception:
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
        except Exception:
            continue
    
    # Very unlikely to be correct after the above matches
    return 0.0
