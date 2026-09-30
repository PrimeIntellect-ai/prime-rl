"""The dispatcher's side of the live traces.

Env servers stream every in-flight rollout as deltas (``verifiers.v1.serve.delta``). The
dispatcher relays each delta to the monitors stamped with who dispatched it, and keeps
only what its own progress line needs: the phase and turn count of each live trace.
"""

from __future__ import annotations

import time
from typing import Any

from prime_rl.monitors.file.traces.live import STAGES, stage
from prime_rl.orchestrator.types import InflightEpisode, LiveTrace


def dispatch_info(meta: InflightEpisode) -> dict[str, Any]:
    """Who an in-flight episode is, stamped on the first line of each of its live traces
    and on its pending placeholder."""
    data = meta.task.data
    name = getattr(data, "name", None)
    return {
        "id": meta.dispatch_id,
        "kind": meta.kind,
        "env": meta.env_name,
        "group": str(meta.group_id),
        "task": str(name) if name else f"idx={data.idx}",
        "policy_version": meta.policy_version,
        "step": meta.step,
        "started": time.time() - (time.monotonic() - meta.started_at) if meta.started_at else None,
    }


def pending_event(meta: InflightEpisode) -> dict[str, Any]:
    """The episode was dispatched; until a trace streams, this is all there is to show."""
    return {"pending": meta.dispatch_id, "dispatch": dispatch_info(meta)}


def dispatched_event(meta: InflightEpisode) -> dict[str, Any]:
    """The episode is no longer pending: a trace streamed, or the episode left the in-flight set."""
    return {"dispatched": meta.dispatch_id}


def apply(meta: InflightEpisode, delta: dict[str, Any]) -> None:
    """Fold one delta into the episode's phase/turn bookkeeping."""
    trace_id = delta["trace"]
    if delta.get("discard"):
        meta.live.pop(trace_id, None)
        return
    trace = meta.live.setdefault(trace_id, LiveTrace())
    trace.turns += len(delta.get("calls") or [])
    changed = delta.get("set") or {}
    if delta.get("errors"):
        trace.stage = "error"
    elif "timing" in changed:
        trace.stage = stage({"timing": changed["timing"]})


def stage_counts(inflight: list[InflightEpisode]) -> str:
    """``boot 1 · running 3`` — the in-flight traces by phase, in lifecycle order; an
    episode with no trace yet counts as pending."""
    counts = dict.fromkeys(STAGES, 0)
    for meta in inflight:
        if not meta.live:
            counts["pending"] += 1
        for trace in meta.live.values():
            counts[trace.stage] = counts.get(trace.stage, 0) + 1
    return " · ".join(f"{name} {n}" for name, n in counts.items() if n)
