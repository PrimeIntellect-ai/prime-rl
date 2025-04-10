from zeroband.rewards.math import compute_math_reward
from zeroband.rewards.kod_code import verify_kod_code
from zeroband.rewards.gsm_infinite import verify_gsm_infinite


REWARD_FUNCTIONS = {
    "verifiable_math": compute_math_reward,
    "kod_code": verify_kod_code,  # pending justus's pr
    "gsm_infinite": verify_gsm_infinite,
}
