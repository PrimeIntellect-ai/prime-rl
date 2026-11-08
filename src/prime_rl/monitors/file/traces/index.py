"""The episode index - one compact row per episode, so a reader never has to parse
the stream to browse it.

The file monitor writes a row as each episode lands, while the episode is still in
hand, and records the chunk and byte offset it wrote the episode at so a reader can
seek straight to one. A stream another producer wrote has no index, so its reader
derives the same rows itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import verifiers.v1 as vf


def index_record(episode: vf.Episode) -> dict[str, Any]:
    """The part of the episode's record (`to_record`) that `summarize_episode` reads:
    everything but the nodes and model calls. Built one trace at a time, so no single
    call holds the GIL for a whole long episode."""
    record = episode.model_dump(mode="json", exclude={"traces"}, exclude_none=True)
    record["traces"] = [
        trace.model_dump(mode="json", exclude={"nodes", "calls"}, exclude_none=True) for trace in episode.traces
    ]
    return record


def walk_timing(obj: dict, prefix: str, out: dict[str, float]) -> None:
    """Flatten a trace timing tree to phase -> seconds."""
    if isinstance(obj.get("duration"), (int, float)):
        out[prefix] = out.get(prefix, 0.0) + obj["duration"]
    for key, value in obj.items():
        if isinstance(value, dict):
            walk_timing(value, f"{prefix}/{key}" if prefix else key, out)


def episode_kind(rec: dict) -> str:
    """The kind of work an episode did. The file monitor stamps it as the episode
    lands; a record without the stamp is read off its run info instead."""
    for trace in rec.get("traces") or []:
        if (kind := (trace.get("info") or {}).get("kind")) in ("train", "eval"):
            return kind
    run = rec.get("run") or {}
    return (run.get("work") or {}).get("type") or run.get("type") or "eval"


def summarize_episode(line: int, rec: dict, offset: int | None = None) -> dict:
    """One index row: what the table, the filters, the chart and the sort need.

    ``line`` numbers the episode within the stream from 1, so the last of n reads as
    n — it is what a reader sees and what addresses the episode."""
    rewards, advantages = [], []
    input_tokens = rec.get("num_input_tokens")
    output_tokens = rec.get("num_output_tokens")
    branches = 0
    stop_condition = None
    truncated = False
    timeout = bool(rec.get("is_timeout"))
    reward_parts: dict[str, list[float]] = {}
    metric_parts: dict[str, list[float]] = {}
    timing: dict[str, float] = {}
    for trace in rec.get("traces") or []:
        branches += trace.get("num_branches") or 0
        # skip reward-less seats (e.g. a judge) in the episode mean
        if trace.get("rewards") and isinstance(trace.get("reward"), (int, float)):
            rewards.append(trace["reward"])
        for name, r in (trace.get("rewards") or {}).items():
            if isinstance(r, dict) and isinstance(r.get("score"), (int, float)):
                reward_parts.setdefault(name, []).append(r["score"])
        for name, value in (trace.get("metrics") or {}).items():
            if isinstance(value, (int, float)):
                metric_parts.setdefault(name, []).append(value)
        if isinstance(trace.get("timing"), dict):
            walk_timing(trace["timing"], "", timing)
        advantage = (trace.get("info") or {}).get("advantage")
        if advantage is not None:
            advantages.append(advantage)
        stop_condition = trace.get("stop_condition", stop_condition)
        truncated = truncated or bool(trace.get("is_truncated"))
        timeout = timeout or bool(trace.get("is_timeout"))
    first_info = ((rec.get("traces") or [{}])[0].get("info")) or {}
    return {
        "rewards": {name: sum(v) / len(v) for name, v in reward_parts.items()},
        "metrics": {name: sum(v) / len(v) for name, v in metric_parts.items()},
        "timing": timing,
        "cost": (rec.get("usage") or {}).get("cost"),
        "line": line,
        "offset": offset,
        "id": rec.get("id"),
        "kind": episode_kind(rec),
        "trace_ids": [trace_id for trace in rec.get("traces") or [] if (trace_id := trace.get("id"))],
        # the env's name is the orchestrator's key for it (two sources can share a taskset
        # id); a record without one falls back to the id
        "env": (rec.get("env") or {}).get("name") or (rec.get("env") or {}).get("id"),
        "group": (rec.get("group") or {}).get("id"),
        "ok": rec.get("ok"),
        "num_errors": len(rec.get("errors") or []),
        "reward": sum(rewards) / len(rewards) if rewards else None,
        "advantage": sum(advantages) / len(advantages) if advantages else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "turns": rec.get("num_turns"),
        "branches": branches,
        "stop_condition": stop_condition,
        "truncated": truncated,
        "timeout": timeout,
        # when the episode landed and how long it was alive: the stream's x axis,
        # and the one duration worth sorting a stream by
        "dispatch": (first_info.get("dispatch") or {}).get("time"),
        "arrival": (first_info.get("arrival") or {}).get("time"),
        "duration": _episode_duration(first_info, timing),
    }


def _episode_duration(info: dict, timing: dict[str, float]) -> float | None:
    """Wall-clock seconds from dispatch to arrival, or the summed phases for a record
    that predates those stamps."""
    dispatched = (info.get("dispatch") or {}).get("time")
    arrived = (info.get("arrival") or {}).get("time")
    if isinstance(dispatched, (int, float)) and isinstance(arrived, (int, float)):
        return max(0.0, arrived - dispatched)
    phases = [value for key, value in timing.items() if "/" not in key]
    return sum(phases) if phases else None


def index_row(line: int, rec: dict, chunk: int, offset: int) -> dict:
    """The row the file monitor writes. The nested reward/metric/timing maps are left
    out: only the per-episode series view needs them, and they dominate the size."""
    row = summarize_episode(line, rec, offset)
    row["chunk"] = chunk
    for key in ("rewards", "metrics", "timing"):
        row.pop(key, None)
    return row
