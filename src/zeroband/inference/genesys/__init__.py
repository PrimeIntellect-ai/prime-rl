from typing import Callable, Literal

from zeroband.inference.genesys.code import evaluate_code
from zeroband.inference.genesys.code_output_prediction import verify_code_output_prediction
from zeroband.inference.genesys.math import compute_math_reward
from zeroband.inference.genesys.reasoning_gym import verify_reasoning_gym
from zeroband.inference.genesys.sanskrit_morphology import compute_sanskrit_morphology_reward

TaskType = Literal["verifiable_math", "prime_rl_code", "reasoning_gym", "code_output_prediction", "sanskrit_morphology"]


def get_reward_function(task_type: TaskType) -> Callable[[str, dict], float]:
    try:
        return _REWARD_FUNCTIONS[task_type]
    except KeyError:
        raise ValueError(f"Invalid task type: {task_type}")


_REWARD_FUNCTIONS: dict[TaskType, Callable] = {
    "verifiable_math": compute_math_reward,
    "prime_rl_code": evaluate_code,
    "reasoning_gym": verify_reasoning_gym,
    "code_output_prediction": verify_code_output_prediction,
    "sanskrit_morphology": compute_sanskrit_morphology_reward,
}
