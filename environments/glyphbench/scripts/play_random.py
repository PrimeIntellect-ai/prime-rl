#!/usr/bin/env python3
"""Play any glyphbench env with random actions, printing each step to the terminal.

Usage:
    uv run python scripts/play_random.py glyphbench/craftaxfull-v0
    uv run python scripts/play_random.py glyphbench/minigrid-doorkey-5x5-v0 --seed 42 --steps 50
    uv run python scripts/play_random.py glyphbench/minihack-eat-v0 --delay 0.3
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time

import numpy as np

import glyphbench  # noqa: F401 — trigger env registration
from glyphbench.core import BaseGlyphEnv, make_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-agent viewer for glyphbench envs")
    parser.add_argument("env_id", help="Env ID, e.g. glyphbench/craftaxfull-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200, help="Max steps to run")
    parser.add_argument("--max-turns", type=int, help="Override the environment's turn limit")
    parser.add_argument("--delay", type=float, default=0.1, help="Seconds between steps (0 = no delay)")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    if args.max_turns is not None and args.max_turns <= 0:
        parser.error("--max-turns must be positive")
    if not math.isfinite(args.delay) or args.delay < 0:
        parser.error("--delay must be finite and nonnegative")

    kwargs = {} if args.max_turns is None else {"max_turns": args.max_turns}
    with contextlib.closing(make_env(args.env_id, **kwargs)) as env:
        _play(env, args)


def _play(env: BaseGlyphEnv, args: argparse.Namespace) -> None:
    action_names = env.action_spec.names
    n_actions = env.action_spec.n
    rng = np.random.default_rng(args.seed)

    obs, info = env.reset(args.seed)

    print(f"\n{'=' * 60}")
    print(f"  {args.env_id}  |  seed={args.seed}  |  actions={n_actions}")
    print(f"{'=' * 60}")
    print(f"\n--- SYSTEM PROMPT ---\n{env.system_prompt()[:400]}...")
    print(f"\n--- STEP 0 (reset) ---\n{obs}\n")

    total_reward = 0.0
    terminated = False
    truncated = False
    for step in range(1, args.steps + 1):
        action = int(rng.integers(0, n_actions))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"--- STEP {step}: {action_names[action]} | reward={reward:+.2f} | total={total_reward:.2f} ---")
        print(obs)

        if reward != 0:
            print(f"  >>> REWARD: {reward:+.2f}")
        if terminated:
            print(f"\n  TERMINATED (total reward: {total_reward:.2f})")
            break
        if truncated:
            print(f"\n  TRUNCATED (total reward: {total_reward:.2f})")
            break

        print()
        if args.delay > 0:
            time.sleep(args.delay)

    if not terminated and not truncated:
        print(f"\n  Stopped after {args.steps} steps (total reward: {total_reward:.2f})")


if __name__ == "__main__":
    main()
