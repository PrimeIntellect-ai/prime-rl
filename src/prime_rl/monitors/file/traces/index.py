"""The episode index - one compact row per episode, so a reader never has to parse
the stream to browse it.

The file monitor writes a row as each episode lands, when the record is already in
hand and summarising it is nearly free, and records the chunk and byte offset it wrote
the episode at so a reader can seek straight to one. A stream another producer wrote
has no index, so its reader derives the same rows itself.
"""


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
    {"max_turns", "max_input_tokens", "max_output_tokens", "max_total_tokens", "max_turn_tokens", "compaction_failed"}
)
"""The stop conditions ``verifiers.v1.Trace.is_truncated`` counts as truncation."""


def trace_truncated(trace: dict) -> bool:
    """``Trace.is_truncated`` for a dumped trace: a framework limit stopped it, or its
    last successful model call ran out of length."""
    if trace.get("stop_condition") in TRUNCATING_STOPS:
        return True
    last = next((call for call in reversed(trace.get("calls") or []) if call.get("error") is None), None)
    return bool(last and last.get("finish_reason") == "length")


def terminal_errors(rec: dict) -> list[dict]:
    """Project terminal trace and episode errors to compact joinable metadata.

    Error histories can include failures that a retry recovered from, so an error
    is terminal only when its owning trace or episode did not finish successfully.
    The last error describes that terminal failure; ``error_count`` retains the
    full error-history size without copying tracebacks or task/runtime payloads.
    When a trace is terminal, episode errors may be retry history from an earlier
    attempt, so only the trace failure is projected; the record has no marker that
    can reliably distinguish a simultaneous terminal episode error from that history.
    """

    def projection(source: dict, *, trace: dict | None = None) -> dict:
        error = source["errors"][-1]
        task = source.get("task") or rec.get("task") or {}
        agent = (trace or {}).get("agent") or {}
        runtime = agent.get("runtime") or {}
        return {
            "run": (rec.get("run") or {}).get("id"),
            "group": (rec.get("group") or {}).get("id"),
            "episode": rec.get("id"),
            "trace": (trace or {}).get("id"),
            "task": task.get("key"),
            "stop_condition": source.get("stop_condition"),
            "type": error.get("type"),
            "message": error.get("message"),
            "status_code": error.get("status_code"),
            "runtime_id": runtime.get("id"),
            "error_count": len(source.get("errors") or []),
        }

    errors = [
        projection(trace, trace=trace)
        for trace in rec.get("traces") or []
        if not trace.get("ok") and trace.get("errors")
    ]
    if not rec.get("ok") and rec.get("errors") and not errors:
        errors.append(projection(rec))
    return errors


def summarize_episode(line: int, rec: dict, offset: int | None = None) -> dict:
    """One index row: what the table, the filters, the chart and the sort need.

    ``line`` numbers the episode within the stream from 1, so the last of n reads as
    n — it is what a reader sees and what addresses the episode."""
    rewards, advantages = [], []
    input_tokens = rec.get("num_input_tokens")
    output_tokens = rec.get("num_output_tokens")
    turns = branches = 0
    stop_condition = None
    truncated = False
    timeout = bool(rec.get("is_timeout"))
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
            if node.get("sampled"):
                turns += 1
        stop_condition = trace.get("stop_condition", stop_condition)
        truncated = truncated or trace_truncated(trace)
        timeout = timeout or bool(trace.get("is_timeout"))
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
        # All retained error history, including trace-only retries. Consumers use
        # this to label successful episodes as recovered and failed ones as errored.
        "num_errors": len(rec.get("errors") or [])
        + sum(len(trace.get("errors") or []) for trace in rec.get("traces") or []),
        "terminal_errors": terminal_errors(rec),
        "reward": sum(rewards) / len(rewards) if rewards else None,
        "advantage": sum(advantages) / len(advantages) if advantages else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "turns": turns,
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
