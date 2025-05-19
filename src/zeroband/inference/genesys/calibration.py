from typing import Dict
from zeroband.inference.genesys.math_utils import extract_answer

def grade_answer_calibration(model_answer: int, true_passrate: float, eps: int = 10) -> bool:
    """
    Check if the model's answer (a number between 0 and 100) is within eps% of the true passrate.
    """
    if abs(model_answer - (true_passrate * 100)) <= eps:
        return True
    return False


def compute_calibration_reward(completion: str, verification_info: Dict) -> int:
    model_response = completion
    true_passrate = verification_info["passrate"]

    # Extract solution.
    if "</think>" in model_response:
        model_solution = model_response.split("</think>")[1]
    else:
        return 0

    # Model must answer with passrate between 0 and 100
    # TODO: edit this if answer formatting instructions change in prompt
    if "\\boxed" in model_solution:
        model_solution = extract_answer(model_solution)        
    try:
        model_answer = int(model_solution)
        assert 0 <= model_answer <= 100
    except ValueError:
        return 0
    except AssertionError:
        return 0

    if true_passrate is None:
        return 0

    is_correct = grade_answer_calibration(model_answer, true_passrate)
    if is_correct:
        return 1

    return 0



if __name__ == '__main__':
    out = compute_calibration_reward("hello \\boxed{15}", dict(passrate=14))
    print(out)