"""Token-level GAE over model actions in a multi-turn trajectory."""

import math
from collections.abc import Sequence


def skip_observation_gae(
    values: Sequence[float],
    action_mask: Sequence[bool],
    reward: float,
    *,
    gamma: float,
    policy_lambda: float,
    value_lambda: float,
    bootstrap_value: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Return policy advantages and critic targets, aligned to the full trace.

    Environment observation positions carry zero outputs. The terminal reward
    lands on the last model action, and consecutive actions in the recursion
    can be separated by any number of observation tokens.
    """
    if len(values) != len(action_mask):
        raise ValueError("values and action_mask must have the same length")
    if (
        any(not math.isfinite(value) for value in values)
        or not math.isfinite(reward)
        or not math.isfinite(bootstrap_value)
    ):
        raise ValueError("values, reward, and bootstrap_value must be finite")
    if not 0 <= gamma <= 1 or not 0 <= policy_lambda <= 1 or not 0 <= value_lambda <= 1:
        raise ValueError("gamma and GAE lambdas must be in [0, 1]")

    advantages = [0.0] * len(values)
    returns = [0.0] * len(values)
    action_indices = [index for index, sampled in enumerate(action_mask) if sampled]
    next_value = bootstrap_value
    policy_gae = 0.0
    value_gae = 0.0
    for index in reversed(action_indices):
        terminal = index == action_indices[-1]
        delta = (reward if terminal else 0.0) + gamma * next_value - values[index]
        policy_gae = delta + gamma * policy_lambda * policy_gae
        value_gae = delta + gamma * value_lambda * value_gae
        advantages[index] = policy_gae
        returns[index] = value_gae + values[index]
        next_value = values[index]
    return advantages, returns
