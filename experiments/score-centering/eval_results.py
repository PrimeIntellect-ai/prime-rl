"""Export task-level evaluation rewards, errors, and uncertainty."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from prime_rl.monitors.file.traces.chunks import chunk_numbers, open_chunk
from prime_rl.monitors.file.traces.index import episode_kind, summarize_episode


def read_episodes(run):
    directory = run / "monitors/file/traces/stream"
    for number in sorted(chunk_numbers(directory)):
        with open_chunk(directory, number) as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    continue
                episode = json.loads(line)
                if episode_kind(episode) != "eval":
                    continue
                summary = summarize_episode(0, episode)
                errors = list(episode.get("errors") or [])
                for trace in episode.get("traces") or []:
                    errors.extend(trace.get("errors") or [])
                work = (episode.get("run") or {}).get("work") or {}
                yield {
                    "run": run.name,
                    "env": summary["env"],
                    "step": work.get("step"),
                    "task": (episode.get("task") or {}).get("key"),
                    "episode": episode["id"],
                    "ok": bool(episode.get("ok")),
                    "reward": summary["reward"],
                    "turns": summary["turns"],
                    "output_tokens": summary["output_tokens"],
                    "truncated": summary["truncated"],
                    "error_types": ";".join(sorted({e["type"] for e in errors})),
                    "error_messages": " | ".join(e.get("message", "") for e in errors),
                }


def task_interval(values):
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    if all(value in (0, 1) for value in values):
        z = 1.959963984540054
        n = len(values)
        denominator = 1 + z * z / n
        center = (mean + z * z / (2 * n)) / denominator
        half = z * math.sqrt(mean * (1 - mean) / n + z * z / (4 * n * n)) / denominator
        return {"method": "Wilson 95% across scored tasks", "low": center - half, "high": center + half}
    import numpy as np

    draws = np.random.default_rng(42).choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"method": "task bootstrap 95%, 10000 draws", "low": float(low), "high": float(high)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [row for run in args.runs for row in read_episodes(run)]
    if rows:
        with (args.output / "eval-episodes.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["run"], row["env"], row["step"]].append(row)
    reports = []
    for (run, env, step), episodes in groups.items():
        tasks = defaultdict(list)
        lower = defaultdict(list)
        upper = defaultdict(list)
        for row in episodes:
            if row["ok"] and isinstance(row["reward"], (int, float)) and math.isfinite(row["reward"]):
                tasks[row["task"]].append(row["reward"])
                lower[row["task"]].append(row["reward"])
                upper[row["task"]].append(row["reward"])
            else:
                lower[row["task"]].append(0.0)
                upper[row["task"]].append(1.0)
        bounds = {
            name: sum(sum(values) / len(values) for values in by_task.values()) / len(by_task)
            for name, by_task in (("lower", lower), ("upper", upper))
        }
        means = [sum(rewards) / len(rewards) for rewards in tasks.values()]
        reports.append(
            {
                "run": run,
                "env": env,
                "step": step,
                "episodes": len(episodes),
                "failed_episodes": sum(not row["ok"] for row in episodes),
                "scored_tasks": len(means),
                "valid_task_reward_mean": sum(means) / len(means) if means else None,
                "interval": task_interval(means),
                "all_task_reward_bounds": bounds,
                "note": "Valid-task estimates exclude failed episodes. Bounds assign missing rewards 0 or 1; these are not confidence intervals.",
            }
        )
    (args.output / "eval-summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
