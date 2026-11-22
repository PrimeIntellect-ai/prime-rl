"""Run a random agent across multiple environments and print results."""

import random

import numpy as np

from glyphbench.core import make_env
from glyphbench.core.task_selection import list_task_ids

SEEDS = [0, 1, 2]
MAX_STEPS = 200


def run_random_episode(env_id: str, seed: int) -> tuple[float, int]:
    """Run one episode with random actions. Returns (total_reward, length)."""
    env = make_env(env_id, max_turns=MAX_STEPS)
    obs, info = env.reset(seed=seed)
    rng = random.Random(seed)
    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        action = rng.randrange(env.action_spec.n)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    env.close()
    return total_reward, steps


def main() -> None:
    env_ids = list_task_ids(exclude_suites=["agentick", "craftaxfull", "nethack"])
    print(f"Running random agent on {len(env_ids)} environments, {len(SEEDS)} seeds each\n")

    for env_id in env_ids[:10]:  # first 10 for demo; remove slice for all
        returns = []
        lengths = []
        for seed in SEEDS:
            ret, length = run_random_episode(env_id, seed)
            returns.append(ret)
            lengths.append(length)
        mean_ret = np.mean(returns)
        mean_len = np.mean(lengths)
        print(f"{env_id:<50s}  return={mean_ret:+.3f}  length={mean_len:.0f}")


if __name__ == "__main__":
    main()
