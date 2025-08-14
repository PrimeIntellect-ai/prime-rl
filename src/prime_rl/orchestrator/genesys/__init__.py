from typing import Callable, Literal

from prime_rl.orchestrator.genesys.math import math_verify_reward_function
from prime_rl.orchestrator.genesys.deepcoder import verify_deepcoder


def null_reward(*args, **kwargs):
    return 0.0


TaskType = Literal[
    "verifiable_math",
    "null_reward",
]


def get_reward_function(task_type: TaskType) -> Callable[[str, dict], float]:
    try:
        return _REWARD_FUNCTIONS[task_type]
    except KeyError:
        raise ValueError(f"Invalid task type: {task_type}")


_REWARD_FUNCTIONS: dict[TaskType, Callable] = {
    "verifiable_math": math_verify_reward_function,
    "deepcoder": verify_deepcoder,
    "null_reward": null_reward,
}
