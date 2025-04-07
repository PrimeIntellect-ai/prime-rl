from zeroband.rewards.math import compute_math_reward
from zeroband.rewards.prime_code import evaluate_code

REWARD_FUNCTIONS = {"verifiable_math": compute_math_reward, "prime_rl_code": evaluate_code}
