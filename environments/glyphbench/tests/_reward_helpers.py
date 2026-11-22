"""Shared helpers for reward-bound auditing.

Lives outside ``test_*`` so non-test scripts (e.g. scripts/audit_reward_bounds.py)
can import the rollout helper without picking up pytest collection.
"""
from __future__ import annotations

import random

from glyphbench.core import make_env


def _random_rollout_return(
    env_id: str,
    seed: int = 0,
    *,
    max_steps: int | None = None,
) -> tuple[float, int]:
    """Run a single random-action rollout; return (cumulative_reward, length).

    If ``max_steps`` is given, the rollout is capped at
    ``min(env.max_turns, max_steps)`` to bound runtime for fast envs whose
    reward violations surface within a few hundred random steps anyway.
    """
    env = make_env(env_id)
    rng = random.Random(seed)
    env.reset(seed=seed)
    n_actions = env.action_spec.n
    cap = env.max_turns if max_steps is None else min(env.max_turns, max_steps)
    total = 0.0
    length = 0
    for _ in range(cap):
        action = rng.randrange(n_actions)
        _, r, terminated, truncated, _ = env.step(action)
        total += float(r)
        length += 1
        if terminated or truncated:
            break
    return total, length
