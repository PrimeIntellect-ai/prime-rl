#!/usr/bin/env python
"""Record a GIF of a random-agent rollout for every GlyphBench env.

For each env, runs the random agent for --steps steps with a fixed seed,
saves the trajectory as JSONL, then uses replay_trajectory.export_gif to
render it as an animated GIF.

Usage:
    uv run python scripts/record_random_gifs.py
    uv run --extra assets python scripts/record_random_gifs.py --steps 20 --output gifs/
    uv run python scripts/record_random_gifs.py --suite minigrid --steps 30
    uv run python scripts/record_random_gifs.py --env glyphbench/atari-pong-v0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import glyphbench  # noqa: F401
from glyphbench.core import all_glyphbench_env_ids, make_env

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_trajectory import export_gif  # noqa: E402


def rollout_random(env_id: str, seed: int, steps: int | None = None) -> list[dict]:
    """Run a random agent and return a list of step dicts matching the
    trajectory JSONL schema consumed by replay_trajectory.py.

    If `steps` is None, the env's own natural `max_turns` governs episode
    length — this matches how `eval/random_baseline.py` measures scores,
    so the rendered GIF reflects the same rollout distribution as the
    scoring baseline. Pass an explicit `steps` only for short demo clips.
    """
    env = make_env(env_id) if steps is None else make_env(env_id, max_turns=steps)
    obs, info = env.reset(seed)
    action_names = env.action_spec.names

    trajectory: list[dict] = []
    cum_reward = 0.0
    turn = 0
    while True:
        turn += 1
        action = int(env.rng.integers(0, env.action_spec.n))
        new_obs, reward, terminated, truncated, _ = env.step(action)
        cum_reward += float(reward)
        trajectory.append({
            "turn": turn,
            "observation": obs,  # observation *before* the action
            "action_idx": int(action),
            "action_name": action_names[int(action)],
            "reward": float(reward),
            "cumulative_reward": float(cum_reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "parse_failed": False,
            "llm_response": "",
        })
        obs = new_obs
        if terminated or truncated:
            break
    env.close()
    return trajectory


def safe_slug(env_id: str) -> str:
    return env_id.replace("/", "__")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=None,
                        help="Override env's natural max_turns (default: None = use env's own budget)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("gifs"),
                        help="Output dir for GIFs (one per env)")
    parser.add_argument("--save-trajectories", type=Path, default=None,
                        help="Optional: also save trajectory JSONLs here")
    parser.add_argument("--suite", type=str, help="Filter by suite name substring")
    parser.add_argument("--env", type=str, help="Run a single env only")
    parser.add_argument("--font-size", type=int, default=14)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-render GIFs even if they already exist")
    parser.add_argument("--grid-only", action="store_true",
                        help="Render only the [Grid] block (no header/Legend/HUD/Memory)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.save_trajectories:
        args.save_trajectories.mkdir(parents=True, exist_ok=True)

    if args.env:
        env_ids = [args.env]
    else:
        env_ids = [e for e in all_glyphbench_env_ids() if "dummy" not in e]
        if args.suite:
            env_ids = [e for e in env_ids if args.suite in e]

    print(f"Recording {len(env_ids)} envs @ {args.steps} steps -> {args.output}")
    fails: list[tuple[str, str]] = []

    for i, env_id in enumerate(env_ids, 1):
        slug = safe_slug(env_id)
        gif_path = args.output / f"{slug}.gif"
        if gif_path.exists() and not args.overwrite:
            print(f"[{i}/{len(env_ids)}] {env_id}: skip (exists)")
            continue

        try:
            traj = rollout_random(env_id, args.seed, steps=args.steps)

            if args.save_trajectories:
                out = args.save_trajectories / f"{slug}__seed{args.seed}.jsonl"
                with out.open("w") as f:
                    for step in traj:
                        f.write(json.dumps(step) + "\n")

            if not traj:
                print(f"[{i}/{len(env_ids)}] {env_id}: EMPTY trajectory, skip")
                continue

            export_gif(traj, gif_path, font_size=args.font_size, grid_only=args.grid_only)
            print(f"[{i}/{len(env_ids)}] {env_id}: {gif_path.name} ({len(traj)} frames)")
        except Exception as e:
            fails.append((env_id, str(e)[:150]))
            print(f"[{i}/{len(env_ids)}] {env_id}: ERROR {e}")

    print(f"\nDone. {len(env_ids) - len(fails)} succeeded, {len(fails)} failed.")
    for eid, err in fails[:10]:
        print(f"  {eid}: {err}")


if __name__ == "__main__":
    main()
