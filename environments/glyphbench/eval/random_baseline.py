#!/usr/bin/env python
"""Run a random-action baseline on every environment. Fast (no LLM needed).

Writes random_baseline.json: {env_id: {mean_return, std_return, ...}}.
Useful as a zero-skill reference when interpreting model scores. The
benchmark itself reports raw episodic returns — there is no enforced
normalisation.

Usage:
    uv run python eval/random_baseline.py
    uv run python eval/random_baseline.py --episodes 50 --include-suite minigrid
    uv run python eval/random_baseline.py --include-all  # eval every suite
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import glyphbench  # noqa: F401 — register envs
from glyphbench.core import list_task_ids, make_env
from glyphbench.eval_defaults import DEFAULT_EXCLUDE_SUITES


def run_random_episodes(
    env_id: str, seeds: list[int], max_turns: int | None = None,
) -> dict[str, Any]:
    """Run a uniform-random agent on one env across all seeds.

    If max_turns is None, each env uses its own natural budget (defined by
    the env class). Overriding it distorts the baseline for envs whose
    difficulty depends on the step budget.
    """
    returns = []
    lengths = []
    if not seeds:
        raise ValueError("at least one episode seed is required")
    for seed in seeds:
        env = make_env(env_id) if max_turns is None else make_env(env_id, max_turns=max_turns)
        try:
            env.reset(int(seed))
            total_reward = 0.0
            steps = 0
            done = False
            while not done:
                action = int(env.rng.integers(0, env.action_spec.n))
                _, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
        finally:
            env.close()
        returns.append(total_reward)
        lengths.append(steps)

    return {
        "env_id": env_id,
        "n_episodes": len(seeds),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "median_return": float(np.median(returns)),
        "mean_length": float(np.mean(lengths)),
        "per_episode_returns": [float(r) for r in returns],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GlyphBench random baseline")
    parser.add_argument("--episodes", type=int, default=25, help="Episodes per env")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Override env's natural max_turns (default: None = use env's own budget)",
    )
    parser.add_argument(
        "--include-suite",
        action="append",
        default=[],
        help="Restrict to these suite names. Repeat for multiple. Default: no restriction.",
    )
    parser.add_argument(
        "--exclude-suite",
        action="append",
        default=None,
        help="Skip these suite names. Repeat for multiple. "
        "Default: ['atari', 'craftaxfull', 'nethack', 'agentick']. "
        "Pass --include-all to disable.",
    )
    parser.add_argument(
        "--include-task",
        action="append",
        default=[],
        help="Exact env id or fnmatch pattern. Repeat for multiple.",
    )
    parser.add_argument(
        "--exclude-task",
        action="append",
        default=[],
        help="Exact env id or fnmatch pattern. Repeat for multiple.",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Disable default suite exclusions; eval everything.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("eval/random_baseline.json"),
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.max_turns is not None and args.max_turns < 1:
        parser.error("--max-turns must be positive")

    include_suites = args.include_suite or None
    if args.include_all:
        exclude_suites: list[str] = []
    elif args.exclude_suite is None:
        exclude_suites = list(DEFAULT_EXCLUDE_SUITES)
    else:
        exclude_suites = args.exclude_suite
    include_tasks = args.include_task or None
    exclude_tasks = args.exclude_task or None

    env_ids = list_task_ids(
        include_suites=include_suites,
        exclude_suites=exclude_suites,
        include_tasks=include_tasks,
        exclude_tasks=exclude_tasks,
    )

    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 2**31, size=args.episodes).tolist()

    print(f"Running random baseline on {len(env_ids)} envs, {args.episodes} episodes each")

    results: dict[str, Any] = {}
    start = time.time()

    for idx, env_id in enumerate(env_ids):
        t0 = time.time()
        try:
            res = run_random_episodes(env_id, seeds, args.max_turns)
            results[env_id] = res
            elapsed = time.time() - t0
            print(
                f"[{idx+1}/{len(env_ids)}] {env_id}: "
                f"return={res['mean_return']:+.3f} len={res['mean_length']:.0f} ({elapsed:.1f}s)"
            )
        except Exception as e:
            print(f"[{idx+1}/{len(env_ids)}] {env_id}: ERROR {e}")
            results[env_id] = {"env_id": env_id, "error": str(e)}

    total = time.time() - start
    print(f"\nDone in {total:.0f}s. {len(results)} envs.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
