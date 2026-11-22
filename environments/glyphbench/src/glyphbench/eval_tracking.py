"""Aggregate GlyphBench eval outputs and synchronize them to W&B.

The normal Verifiers harness writes ``metadata.json`` + ``results.jsonl``;
the standalone pro harness writes ``metrics.json``.  This module normalizes
both formats so the Azure wrappers can publish comparable episode, score,
failure, latency, and token metrics without putting credentials in results.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from contextlib import suppress
from pathlib import Path
from typing import Any


def safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _normal_raw_score(info: dict[str, Any]) -> tuple[float | None, str | None]:
    raw = _number(info.get("raw_score"))
    label = info.get("raw_score_label")
    if raw is not None:
        return raw, str(label or "score")
    env_info = info.get("last_env_info")
    if not isinstance(env_info, dict):
        return None, None
    achievements = _number(env_info.get("num_achievements"))
    if achievements is None and isinstance(env_info.get("achievements"), list):
        achievements = float(len(env_info["achievements"]))
    if achievements is not None:
        return achievements, "achievements"
    score = _number(env_info.get("nethack_blstats_score"))
    if score is None:
        score = _number(env_info.get("score"))
    return (score, "score") if score is not None else (None, None)


def _achievement_names(info: dict[str, Any]) -> list[str]:
    direct = info.get("achievement_names")
    if isinstance(direct, list):
        return sorted({str(name) for name in direct})
    env_info = info.get("last_env_info")
    if not isinstance(env_info, dict):
        return []
    names = env_info.get("achievements")
    if isinstance(names, list):
        return sorted({str(name) for name in names})
    return sorted(
        key.split("/", 1)[1]
        for key, value in env_info.items()
        if key.startswith("Achievements/") and value
    )


def load_normal_episodes(results_root: Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for results_path in sorted(results_root.glob("**/results.jsonl")):
        metadata_path = results_path.with_name("metadata.json")
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            with suppress(OSError, TypeError, json.JSONDecodeError):
                metadata = _load_object(metadata_path)
        with results_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                info = row.get("info") if isinstance(row.get("info"), dict) else {}
                metrics = (
                    row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
                )
                usage = (
                    row.get("token_usage")
                    if isinstance(row.get("token_usage"), dict)
                    else {}
                )
                raw_score, raw_label = _normal_raw_score(info)
                episodes.append(
                    {
                        "task": str(info.get("env_id") or metadata.get("env_id") or "glyphbench"),
                        "seed": info.get("seed"),
                        "reward": _number(row.get("reward")),
                        "raw_score": raw_score,
                        "raw_score_label": raw_label,
                        "achievement_names": _achievement_names(info),
                        "error": row.get("error") is not None,
                        "metrics": {
                            str(key): value
                            for key, value in metrics.items()
                            if _number(value) is not None
                        },
                        "input_tokens": _number(usage.get("input_tokens")) or 0.0,
                        "output_tokens": _number(usage.get("output_tokens")) or 0.0,
                        "gif_path": info.get("gif_path"),
                        "source": f"{results_path}:{line_number}",
                    }
                )
    return episodes


def load_pro_episodes(results_root: Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for path in sorted(results_root.glob("**/metrics.json")):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(rows, list):
            continue
        config: dict[str, Any] = {}
        config_path = path.with_name("config.json")
        if config_path.exists():
            with suppress(OSError, TypeError, json.JSONDecodeError):
                config = _load_object(config_path)
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            metrics = {
                str(key): value
                for key, value in row.items()
                if _number(value) is not None
                and key
                not in {
                    "episode",
                    "seed",
                    "episode_return",
                    "raw_score",
                    "total_input_tokens",
                    "total_output_tokens",
                }
            }
            episodes.append(
                {
                    "task": str(config.get("task_id") or row.get("task_id") or "glyphbench"),
                    "seed": row.get("seed"),
                    "reward": _number(row.get("episode_return")),
                    "raw_score": _number(row.get("raw_score")),
                    "raw_score_label": row.get("raw_score_label"),
                    "achievement_names": (
                        sorted({str(name) for name in row["achievement_names"]})
                        if isinstance(row.get("achievement_names"), list)
                        else []
                    ),
                    "error": bool(row.get("error")),
                    "metrics": metrics,
                    "input_tokens": _number(row.get("total_input_tokens")) or 0.0,
                    "output_tokens": _number(row.get("total_output_tokens")) or 0.0,
                    "gif_path": str(path.with_name(f"episode_{row.get('episode', index)}.gif")),
                    "source": f"{path}:{index}",
                }
            )
    return episodes


def aggregate_episodes(episodes: list[dict[str, Any]]) -> dict[str, float]:
    if not episodes:
        return {}

    def values(key: str) -> list[float]:
        return [number for row in episodes if (number := _number(row.get(key))) is not None]

    rewards = values("reward")
    raw_scores = values("raw_score")
    summary: dict[str, float] = {
        "eval/episodes": float(len(episodes)),
        "eval/errored_episodes": float(sum(bool(row.get("error")) for row in episodes)),
        "eval/input_tokens_total": sum(float(row["input_tokens"]) for row in episodes),
        "eval/output_tokens_total": sum(float(row["output_tokens"]) for row in episodes),
    }
    if rewards:
        summary.update(
            {
                "eval/reward_mean": statistics.mean(rewards),
                "eval/return_mean": statistics.mean(rewards),
                "eval/reward_max": max(rewards),
                "eval/return_max": max(rewards),
                "eval/reward_min": min(rewards),
            }
        )
    if raw_scores:
        summary.update(
            {
                "eval/raw_score_mean": statistics.mean(raw_scores),
                "eval/raw_score_max": max(raw_scores),
                "eval/raw_score_min": min(raw_scores),
            }
        )
        achievements = [
            score for row in episodes
            if row.get("raw_score_label") == "achievements"
            and (score := _number(row.get("raw_score"))) is not None
        ]
        if achievements:
            summary["eval/achievements_mean"] = statistics.mean(achievements)
            summary["eval/achievements_max"] = max(achievements)

    metric_names = sorted(
        {
            str(name)
            for row in episodes
            for name in row.get("metrics", {})
        }
    )
    for name in metric_names:
        metric_values = [
            number
            for row in episodes
            if (number := _number(row.get("metrics", {}).get(name))) is not None
        ]
        if metric_values:
            summary[f"eval/metrics/{safe_key(name)}_mean"] = statistics.mean(metric_values)
    return summary


def proxy_health_metrics(results_root: Path) -> dict[str, float]:
    totals: dict[str, float] = {}
    for path in results_root.glob("**/azure_proxy_metrics.json"):
        try:
            payload = _load_object(path)
        except (OSError, TypeError, json.JSONDecodeError):
            continue
        for key, value in payload.items():
            number = _number(value)
            if number is not None:
                metric = f"health/azure_proxy/{safe_key(str(key))}"
                totals[metric] = totals.get(metric, 0.0) + number
    return totals


def _episode_log(row: dict[str, Any], index: int) -> dict[str, Any]:
    values: dict[str, Any] = {
        "episode/index": float(index),
        "episode/task": str(row["task"]),
        "episode/error": float(bool(row.get("error"))),
        "episode/input_tokens": float(row["input_tokens"]),
        "episode/output_tokens": float(row["output_tokens"]),
    }
    for source, target in (
        ("seed", "episode/seed"),
        ("reward", "episode/reward"),
        ("raw_score", "episode/raw_score"),
    ):
        number = _number(row.get(source))
        if number is not None:
            values[target] = number
    if row.get("raw_score_label"):
        values["episode/raw_score_label"] = str(row["raw_score_label"])
        if row["raw_score_label"] == "achievements" and "episode/raw_score" in values:
            values["episode/achievements"] = values["episode/raw_score"]
    achievement_names = row.get("achievement_names")
    if isinstance(achievement_names, list):
        values["episode/achievement_names"] = ", ".join(achievement_names)
        for name in achievement_names:
            values[f"episode/achievements/{safe_key(str(name))}"] = 1.0
    for name, value in row.get("metrics", {}).items():
        number = _number(value)
        if number is not None:
            values[f"episode/metrics/{safe_key(str(name))}"] = number
    return values


def sync_wandb(
    *,
    results_root: Path,
    harness: str,
    model: str,
    task: str | None = None,
    run_name: str,
    run_id: str,
    entity: str | None,
    project: str,
    group: str | None,
    reasoning_effort: str | None,
) -> str:
    episodes = (
        load_pro_episodes(results_root)
        if harness == "pro"
        else load_normal_episodes(results_root)
    )
    summary = aggregate_episodes(episodes)
    summary.update(proxy_health_metrics(results_root))
    if not episodes:
        raise RuntimeError(f"no {harness} eval episodes found under {results_root}")
    tasks = [task] if task else sorted({str(episode["task"]) for episode in episodes})
    suites = sorted({
        task_id.removeprefix("glyphbench/").split("-", 1)[0]
        for task_id in tasks
        if task_id.startswith("glyphbench/")
    })

    import wandb

    os.environ.setdefault("WANDB_SILENT", "true")
    run = wandb.init(
        entity=entity or None,
        project=project,
        id=run_id,
        resume="allow",
        name=run_name,
        group=group or None,
        job_type="evaluation",
        tags=[
            "glyphbench",
            *suites,
            "azure",
            harness,
            safe_key(model),
            *([f"reasoning-{reasoning_effort}"] if reasoning_effort else []),
        ],
        config={
            "harness": harness,
            "model": model,
            "task": tasks[0] if len(tasks) == 1 else tasks,
            "reasoning_effort": reasoning_effort,
        },
        allow_val_change=True,
        settings=wandb.Settings(init_timeout=60),
    )
    if run is None:
        raise RuntimeError("wandb.init returned no run")
    gifs_logged = 0
    for index, episode in enumerate(episodes):
        episode_values = _episode_log(episode, index)
        gif_path = episode.get("gif_path")
        if gif_path and Path(str(gif_path)).is_file():
            episode_values["episode/gif"] = wandb.Video(
                str(gif_path),
                caption=f"{model} {harness} seed={episode.get('seed')}",
                format="gif",
            )
            gifs_logged += 1
        run.log(episode_values)
    summary["eval/gifs_logged"] = float(gifs_logged)
    summary["monitor/synced_at_unix"] = time.time()
    run.log(summary)
    run.summary.update(summary)
    status = {
        "entity": entity,
        "project": project,
        "run_id": run_id,
        "url": run.url,
        "harness": harness,
        "model": model,
        "episodes": len(episodes),
        "synced_at_unix": summary["monitor/synced_at_unix"],
    }
    (results_root / "wandb.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    run.finish()
    return str(status["url"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--harness", choices=["normal", "pro"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", help="Override the task label inferred from episode results")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "glyphbench-evals"))
    parser.add_argument("--group", default=os.environ.get("WANDB_GROUP"))
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get("AZURE_OPENAI_REASONING_EFFORT"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    url = sync_wandb(
        results_root=args.results_root,
        harness=args.harness,
        model=args.model,
        task=args.task,
        run_name=args.run_name,
        run_id=args.run_id or safe_key(args.run_name),
        entity=args.entity,
        project=args.project,
        group=args.group,
        reasoning_effort=args.reasoning_effort,
    )
    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
