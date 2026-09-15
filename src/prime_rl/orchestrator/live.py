"""The live view of in-flight episodes: what a served rollout looks like mid-run.

Env servers stream each trace as it changes (``verifiers.v1.serve.delta``); the
dispatcher keeps every in-flight episode's ``EpisodeAssembly`` and this module turns
them into rows the periodic log and the dashboard read — one row per live trace (or per
episode still waiting for its first trace), with the phase it is in and what it has
spent so far.
"""

from __future__ import annotations

import time
from typing import Any

from prime_rl.orchestrator.types import InflightEpisode

STAGES = ("pending", "boot", "setup", "running", "finalize", "scoring", "done", "error")

SNIPPET_CHARS = 160


def stage(trace: dict[str, Any]) -> str:
    """The phase a trace is in, read off its timing spans the way the rollout sets them."""
    if trace.get("errors"):
        return "error"
    timing = trace.get("timing") or {}

    def started(span: str) -> bool:
        return bool((timing.get(span) or {}).get("start"))

    if (timing.get("scoring") or {}).get("end"):
        return "done"
    for span, name in (
        ("scoring", "scoring"),
        ("finalize", "finalize"),
        ("agent", "running"),
        ("setup", "setup"),
        ("boot", "boot"),
    ):
        if started(span):
            return name
    return "pending"


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def trace_row(trace: dict[str, Any]) -> dict[str, Any]:
    calls = trace.get("calls") or []
    usage = [call.get("usage") or {} for call in calls]
    nodes = trace.get("nodes") or []
    last = " ".join(message_text(nodes[-1].get("message") or {}).split()) if nodes else ""
    costs = [u["cost"] for u in usage if u.get("cost") is not None]
    return {
        "trace": trace.get("id"),
        "agent": (trace.get("agent") or {}).get("name", "agent"),
        "stage": stage(trace),
        "turns": len(calls),
        "input_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
        "output_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
        "cost": sum(costs) if costs else None,
        "stop_condition": trace.get("stop_condition"),
        "errors": len(trace.get("errors") or []),
        "last": last[:SNIPPET_CHARS],
    }


def task_label(task: Any) -> str:
    data = task.data
    name = getattr(data, "name", None)
    return str(name) if name else f"idx={data.idx}"


def rows(inflight: list[InflightEpisode]) -> list[dict[str, Any]]:
    now = time.monotonic()
    out = []
    for meta in inflight:
        base = {
            "kind": meta.kind,
            "env": meta.env_name,
            "group": str(meta.group_id),
            "task": task_label(meta.task),
            "policy_version": meta.policy_version,
            "elapsed": now - meta.started_at if meta.started_at else None,
        }
        traces = list(meta.assembly.traces.values()) if meta.assembly is not None else []
        if not traces:
            out.append({**base, "trace": None, "agent": None, "stage": "pending", "turns": 0})
        for trace in traces:
            out.append({**base, **trace_row(trace)})
    return out


def stage_counts(rows: list[dict[str, Any]]) -> str:
    """``boot 1 · running 3`` — the live rows by phase, in lifecycle order."""
    counts = {name: 0 for name in STAGES}
    for row in rows:
        counts[row["stage"]] = counts.get(row["stage"], 0) + 1
    return " · ".join(f"{name} {n}" for name, n in counts.items() if n)
