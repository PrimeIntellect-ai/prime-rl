ZERO_ADVANTAGE_EPS = 1e-8


def get_training_rollout_mask(
    advantages: list[float],
    skip_verification: bool,
    zero_advantage_eps: float = ZERO_ADVANTAGE_EPS,
) -> list[bool]:
    """Return whether each rollout should be kept for training."""
    if skip_verification:
        return [True] * len(advantages)
    return [abs(advantage) > zero_advantage_eps for advantage in advantages]
