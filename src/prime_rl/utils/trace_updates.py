"""Trace updates — post-hoc facts about a completed trace, recorded beside it.

An episode is serialized once, when it arrives. Everything a consumer learns later
is an update naming the trace: an ``info`` dict merged into the trace's own, plus
per-token streams over a branch, full-length across the branch's token prefix (nulls
mark unknown positions, so a stream never depends on the producer's loss mask).

Each producer appends its own file under the file monitor's ``annotations/``
directory, so every file has exactly one writer. Readers fold the updates onto the
arrival records in write order, newest wins.
"""

from typing import Any

UPDATE_VERSION = 1

STREAM_FIELDS = ("advantages", "trainer_logprobs", "entropies")
"""Per-token streams an update may carry, compacted onto node fields of the same name."""


def index_suffix(producer_file: str) -> str:
    """The name of a producer's annotation index."""
    return producer_file.replace(".jsonl", ".index.jsonl")


def update_index_row(update: dict[str, Any], offset: int) -> dict[str, Any]:
    """An update's scalars plus where its record sits. The per-token streams stay in
    the annotation file: a reader browsing a run needs the scalars, and only the
    episode it opens needs the streams."""
    return {"trace_id": update.get("trace_id"), "offset": offset, "info": update.get("info") or {}}


def make_update(
    trace_id: str,
    *,
    info: dict[str, Any] | None = None,
    branches: dict[int, dict[str, list]] | None = None,
) -> dict[str, Any]:
    """One update record. ``branches`` maps a branch index to its token streams."""
    return {
        "version": UPDATE_VERSION,
        "trace_id": trace_id,
        "info": info or {},
        "branches": [{"index": index, **streams} for index, streams in (branches or {}).items()],
    }


def branch_node_paths(nodes: list[dict]) -> list[list[int]]:
    """Root-to-leaf node indexes, one path per branch, in leaf order."""
    if not nodes:
        return []
    parents = {parent for node in nodes if isinstance((parent := node.get("parent")), int) and 0 <= parent < len(nodes)}
    paths = []
    for leaf in (index for index in range(len(nodes)) if index not in parents):
        path: list[int] = []
        seen: set[int] = set()
        node_index: int | None = leaf
        while isinstance(node_index, int) and 0 <= node_index < len(nodes) and node_index not in seen:
            seen.add(node_index)
            path.append(node_index)
            node_index = nodes[node_index].get("parent")
        paths.append(list(reversed(path)))
    return paths


def _deep_merge(old: Any, new: Any) -> Any:
    if isinstance(old, dict) and isinstance(new, dict):
        return {**old, **{key: _deep_merge(old.get(key), value) for key, value in new.items()}}
    return new


def fold_trace_updates(trace: dict, updates: list[dict]) -> int:
    """Apply updates onto a raw trace record in write order, newest winning.

    Merges each ``info`` and projects every branch stream onto the branch's nodes,
    compact over each node's mask like the node's own ``logprobs``. A node takes a
    stream only when it is fully covered with non-null values at every sampled
    position, so a truncated stream leaves the tail nodes untouched. Returns how many
    nodes carry trainer logprobs afterwards."""
    nodes = trace.get("nodes") or []
    paths = branch_node_paths(nodes)
    for update in updates:
        if update.get("info"):
            trace["info"] = _deep_merge(trace.get("info") or {}, update["info"])
        for branch in update.get("branches") or []:
            branch_index = branch.get("index")
            if not isinstance(branch_index, int) or not 0 <= branch_index < len(paths):
                continue
            for field in STREAM_FIELDS:
                stream = branch.get(field)
                if not stream:
                    continue
                cursor = 0
                for node_index in paths[branch_index]:
                    node = nodes[node_index]
                    token_ids = node.get("token_ids") or []
                    span = stream[cursor : cursor + len(token_ids)]
                    cursor += len(token_ids)
                    if len(span) < len(token_ids):
                        break
                    values = [v for v, sampled in zip(span, node.get("mask") or []) if sampled]
                    if not values or any(v is None for v in values):
                        continue
                    node[field] = values
    return sum(1 for node in nodes if node.get("trainer_logprobs"))
