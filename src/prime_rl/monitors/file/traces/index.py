"""The episode index - one compact row per episode, so a reader never has to parse
the stream to browse it.

The file monitor writes a row as each episode lands, when the record is already in
hand and summarising it is nearly free, and records the chunk and byte offset it wrote
the episode at so a reader can seek straight to one. A stream another producer wrote
has no index, so its reader derives the same rows itself.
"""

import verifiers.v1 as vf


def walk_timing(obj: dict, prefix: str, out: dict[str, float]) -> None:
    """Flatten a trace timing tree to phase -> seconds (same walk as the viewer)."""
    if isinstance(obj.get("duration"), (int, float)):
        out[prefix] = out.get(prefix, 0.0) + obj["duration"]
    elif isinstance(obj.get("start"), (int, float)) and isinstance(obj.get("end"), (int, float)):
        out[prefix] = out.get(prefix, 0.0) + (obj["end"] - obj["start"])
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


TRUNCATING_STOPS = frozenset(
    {"max_turns", "max_input_tokens", "max_output_tokens", "max_total_tokens", "compaction_failed"}
)
"""The stop conditions ``verifiers.v1.Trace.is_truncated`` counts as truncation."""


def trace_truncated(trace: dict) -> bool:
    """``Trace.is_truncated`` for a dumped trace: a framework limit stopped it, or its
    last successful model call ran out of length."""
    if trace.get("stop_condition") in TRUNCATING_STOPS:
        return True
    last = next((call for call in reversed(trace.get("calls") or []) if call.get("error") is None), None)
    return bool(last and last.get("finish_reason") == "length")


def summarize_episode(line: int, rec: dict, offset: int | None = None) -> dict:
    """One index row: what the table, the filters, the chart and the sort need.

    ``line`` numbers the episode within the stream from 1, so the last of n reads as
    n — it is what a reader sees and what addresses the episode."""
    rewards, advantages = [], []
    episode = vf.WireEpisode.model_validate(rec, extra="ignore")
    input_tokens = episode.num_input_tokens
    output_tokens = episode.num_output_tokens
    turns = branches = 0
    stop_condition = None
    truncated = False
    reward_parts: dict[str, list[float]] = {}
    metric_parts: dict[str, list[float]] = {}
    timing: dict[str, float] = {}
    costs: list[float] = []
    for trace in rec.get("traces") or []:
        costs.extend(
            usage["cost"]
            for call in trace.get("calls") or []
            if isinstance(usage := call.get("usage") or {}, dict) and isinstance(usage.get("cost"), (int, float))
        )
        nodes = trace.get("nodes") or []
        parents = {node.get("parent") for node in nodes if "parent" in node}
        branches += max(0, len(nodes) - len(parents))
        if trace.get("rewards"):  # skip reward-less seats (e.g. a judge) in the episode mean
            rewards.append(
                sum(
                    (r.get("score") or 0) * (r.get("weight") if r.get("weight") is not None else 1)
                    for r in trace["rewards"].values()
                    if isinstance(r, dict)
                )
            )
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
        for node in nodes:
            if (node.get("message") or {}).get("role") == "assistant":
                turns += 1
        stop_condition = trace.get("stop_condition", stop_condition)
        truncated = truncated or trace_truncated(trace)
    first_info = ((rec.get("traces") or [{}])[0].get("info")) or {}
    return {
        "rewards": {name: sum(v) / len(v) for name, v in reward_parts.items()},
        "metrics": {name: sum(v) / len(v) for name, v in metric_parts.items()},
        "timing": timing,
        "cost": sum(costs) if costs else None,
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
        "turns": turns,
        "branches": branches,
        "stop_condition": stop_condition,
        "truncated": truncated,
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
