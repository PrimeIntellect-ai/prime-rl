"""Quickstart: create an environment, observe, and step through it."""

import random

from glyphbench.core import make_env

# Pick any registered environment.
env = make_env("glyphbench/minigrid-empty-5x5-v0")
obs, info = env.reset(seed=42)

print("=== Observation ===")
print(obs)
print()
print(f"Available actions: {env.action_spec.names}")
print(f"Action space size: {env.action_spec.n}")
print()

# Take a few steps
rng = random.Random(42)
for i in range(5):
    action = rng.randrange(env.action_spec.n)
    action_name = env.action_spec.names[action]
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i + 1}: action={action_name}, reward={reward:.1f}, done={terminated or truncated}")

env.close()
