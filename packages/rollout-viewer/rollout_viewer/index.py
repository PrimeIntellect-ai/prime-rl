"""Index + transcript-shard writers — the serialization layer over a backend.

``write_index`` flattens episodes into ``episodes.parquet`` (one row each, exactly
``INDEX_COLUMNS``, zstd-compressed). The ``diagnostics`` column carries per-episode
and per-member rollups of the joined ``StepDiagnostics``, re-aggregated as
``Σsum / Σcount`` across an episode's steps (never mean-of-means; see ``contracts.md``).
The ``tokens`` column carries per-episode + per-member precomputed prompt/completion
token totals, summed from each Step's ``n_{prompt,completion}_tokens`` (None when no
step in scope carried a count — a missing total is not zero).

``write_transcript_shard`` writes one gzipped jsonl shard per training step: each
line is a stripped ``Episode.model_dump()`` (full messages + joined diagnostics,
no token-id arrays). The index row's ``transcript_shard`` + ``transcript_line``
point at the episode's line; order within a shard is episode order in the source.

``update_runs_registry`` keeps ``runs.json`` in sync (idempotent per step).
"""

from __future__ import annotations

import gzip
import io
import json
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from rollout_viewer.schema import Episode, StepDiagnostics, ValueSummary
from rollout_viewer.storage import (
    INDEX_COLUMNS,
    RUNS_REGISTRY,
    StorageBackend,
    index_path,
    setup_path,
    transcript_path,
)

# Quantile/extrema fields re-aggregated only as a worst-case envelope (p99/max take
# the max, min takes the min). Means are recomputed exactly from Σsum/Σcount, so we
# never report a mean-of-means.
_QUANTITIES: tuple[str, ...] = ("importance_ratio", "mismatch_kl", "entropy")


def _accumulate(acc: dict, q: str, vs: ValueSummary) -> None:
    """Fold one step's ``ValueSummary`` for quantity ``q`` into an accumulator."""
    a = acc.setdefault(q, {"sum": 0.0, "count": 0, "p90": None, "p99": None, "min": None, "max": None})
    a["sum"] += vs.sum
    a["count"] += vs.count
    if vs.p90 is not None:
        a["p90"] = vs.p90 if a["p90"] is None else max(a["p90"], vs.p90)
    if vs.p99 is not None:
        a["p99"] = vs.p99 if a["p99"] is None else max(a["p99"], vs.p99)
    if vs.min is not None:
        a["min"] = vs.min if a["min"] is None else min(a["min"], vs.min)
    if vs.max is not None:
        a["max"] = vs.max if a["max"] is None else max(a["max"], vs.max)


def _finalize(acc: dict) -> dict:
    """Turn a quantity accumulator into the rollup dict with an exact mean."""
    out: dict[str, dict] = {}
    for q, a in acc.items():
        if a["count"] == 0:
            raise ValueError(f"diagnostics rollup for {q!r} has zero count — cannot mean")
        rollup = {"sum": a["sum"], "count": a["count"], "mean": a["sum"] / a["count"]}
        for k in ("p90", "p99", "min", "max"):
            if a[k] is not None:
                rollup[k] = a[k]
        out[q] = rollup
    return out


def _rollup_episode(ep: Episode) -> dict:
    """Per-episode + per-member diagnostic rollups, plus status histograms.

    Matches ``contracts.md`` → "Diagnostics rollup schema": one entry per scope
    (the literal ``"episode"`` plus each ``member_id``); each scope has a status
    histogram over the ``DiagnosticsStatus`` literals + ``n_steps``; each quantity
    block (``importance_ratio`` / ``mismatch_kl`` / ``entropy``) carries ``sum`` +
    ``count`` ALWAYS and is omitted entirely only when no step in that scope has a
    ``ValueSummary`` for it. Quantiles are a worst-case envelope; ``mean`` is exact
    ``sum/count``.

    Every Step must already carry a joined ``StepDiagnostics`` (the parser sets
    ``status="absent"`` for a no-sidecar run). A ``None`` here means the join was
    bypassed — fail loud rather than inventing a ``"none"`` status the contract
    does not define.
    """
    buckets: dict[str, dict] = {"episode": {"acc": {}, "status": {}, "n_steps": 0}}
    for member in ep.members:
        buckets[member] = {"acc": {}, "status": {}, "n_steps": 0}

    def _bump(target: str, diag: StepDiagnostics | None) -> None:
        if diag is None:
            raise ValueError(
                f"step in episode example_id={ep.example_id!r} (run_id={ep.run_id!r}, "
                f"step={ep.step}) has no joined StepDiagnostics — the diagnostics "
                f"join was bypassed. The parser must stamp every Step (status="
                f"'absent' for no-sidecar runs) before rolling up."
            )
        b = buckets[target]
        b["n_steps"] += 1
        b["status"][diag.status] = b["status"].get(diag.status, 0) + 1
        for q in _QUANTITIES:
            vs = getattr(diag, q)
            if vs is not None:
                _accumulate(b["acc"], q, vs)

    for s in ep.steps:
        _bump("episode", s.diagnostics)
        if s.member_id is not None:
            _bump(s.member_id, s.diagnostics)

    out: dict[str, dict] = {}
    for name, b in buckets.items():
        entry: dict = {"status": b["status"], "n_steps": b["n_steps"]}
        entry.update(_finalize(b["acc"]))
        out[name] = entry
    return out


def _rollup_tokens(ep: Episode) -> dict:
    """Per-episode + per-member precomputed token totals.

    Mirrors the ``diagnostics`` rollup's scope layout (the literal ``"episode"`` plus
    each ``member_id``). Each scope carries summed ``prompt`` / ``completion`` token
    counts over its steps plus ``n_steps`` and ``n_counted`` (steps that actually
    carried a count — a step with ``n_*_tokens=None`` is excluded from the sum, never
    coerced to 0). ``None`` sums (no step in scope carried a count) stay ``None`` — a
    missing total is not zero.
    """
    buckets: dict[str, dict] = {"episode": _empty_token_bucket()}
    for member in ep.members:
        buckets[member] = _empty_token_bucket()

    def _bump(target: str, s) -> None:
        b = buckets[target]
        b["n_steps"] += 1
        counted = False
        if s.n_prompt_tokens is not None:
            b["prompt"] = (b["prompt"] or 0) + s.n_prompt_tokens
            counted = True
        if s.n_completion_tokens is not None:
            b["completion"] = (b["completion"] or 0) + s.n_completion_tokens
            counted = True
        if counted:
            b["n_counted"] += 1

    for s in ep.steps:
        _bump("episode", s)
        if s.member_id is not None:
            _bump(s.member_id, s)
    return buckets


def _empty_token_bucket() -> dict:
    return {"prompt": None, "completion": None, "n_steps": 0, "n_counted": 0}


def _index_row(ep: Episode, line: int) -> dict:
    """One ``episodes.parquet`` row (exactly ``INDEX_COLUMNS``)."""
    winner = None
    answers = None
    if ep.mar is not None:
        winner = ep.mar.categorical.get("winner")
        # per-seat FINAL answer letters (keyed "final_answer/<seat>") → {seat: letter},
        # so the list can show the answer spread across a question's group of debates.
        seat_answers = {k.split("/", 1)[1]: v for k, v in ep.mar.categorical.items() if k.startswith("final_answer/")}
        answers = seat_answers or None
    return {
        "run_id": ep.run_id,
        "step": ep.step,
        "example_id": ep.example_id,
        "trajectory_id": ep.trajectory_id,
        "rollout_id": ep.rollout_id,
        "kind": ep.kind,
        "members": json.dumps(ep.members),
        "reward": ep.reward,
        "advantage": ep.advantage,
        "is_filtered": ep.is_filtered,
        "error": ep.error,
        "winner": winner,
        "answers": json.dumps(answers) if answers is not None else None,
        "metrics": json.dumps(ep.metrics),
        "diagnostics": json.dumps(_rollup_episode(ep)),
        "tokens": json.dumps(_rollup_tokens(ep)),
        "transcript_shard": transcript_path(ep.run_id, ep.step),
        "transcript_line": line,
    }


# Explicit Arrow schema so the parquet columns are typed and ordered exactly as
# INDEX_COLUMNS, regardless of null-only columns in a given batch.
_ARROW_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("step", pa.int64()),
        ("example_id", pa.string()),
        ("trajectory_id", pa.string()),
        ("rollout_id", pa.string()),
        ("kind", pa.string()),
        ("members", pa.string()),
        ("reward", pa.float64()),
        ("advantage", pa.float64()),
        ("is_filtered", pa.bool_()),
        ("error", pa.string()),
        ("winner", pa.string()),
        ("answers", pa.string()),
        ("metrics", pa.string()),
        ("diagnostics", pa.string()),
        ("tokens", pa.string()),
        ("transcript_shard", pa.string()),
        ("transcript_line", pa.int64()),
    ]
)

assert tuple(_ARROW_SCHEMA.names) == INDEX_COLUMNS, "index Arrow schema drifted from INDEX_COLUMNS"


def build_index_rows(episodes: list[Episode], transcript_lines: list[int] | None = None) -> list[dict]:
    """Index rows in source order.

    ``transcript_line`` points at the episode's line in its step shard. By default
    it is the per-shard 0-based offset (one shard per training step). When
    ``transcript_lines`` is given (retention-aware sync), it overrides the offset
    positionally — e.g. ``-1`` for an episode whose transcript was not retained.
    """
    if transcript_lines is not None and len(transcript_lines) != len(episodes):
        raise ValueError(f"transcript_lines length {len(transcript_lines)} != episodes {len(episodes)}")
    line_by_shard: dict[str, int] = {}
    rows: list[dict] = []
    for i, ep in enumerate(episodes):
        if transcript_lines is not None:
            line = transcript_lines[i]
        else:
            shard = transcript_path(ep.run_id, ep.step)
            line = line_by_shard.get(shard, 0)
            line_by_shard[shard] = line + 1
        rows.append(_index_row(ep, line))
    return rows


def serialize_index(episodes: list[Episode], transcript_lines: list[int] | None = None) -> bytes:
    """Episodes → parquet+zstd bytes (exactly ``INDEX_COLUMNS``)."""
    rows = build_index_rows(episodes, transcript_lines)
    columns = {name: [r[name] for r in rows] for name in INDEX_COLUMNS}
    table = pa.table(columns, schema=_ARROW_SCHEMA)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    return buf.getvalue()


def write_index(
    episodes: list[Episode],
    backend: StorageBackend,
    run_id: str,
    transcript_lines: list[int] | None = None,
) -> str:
    """Write the full ``index/{run_id}/episodes.parquet`` for ``run_id``.

    ``episodes`` is the complete set for this run (all steps); the parquet is a
    whole-run rewrite, not an append — DuckDB reads one file per run.
    """
    for ep in episodes:
        if ep.run_id != run_id:
            raise ValueError(f"episode run_id={ep.run_id!r} does not match index run_id={run_id!r}")
    backend.write(index_path(run_id), serialize_index(episodes, transcript_lines))
    return index_path(run_id)


def serialize_transcript_shard(episodes_of_step: list[Episode]) -> bytes:
    """Stripped episodes → gzipped jsonl bytes, one ``model_dump()`` per line."""
    raw = io.BytesIO()
    for ep in episodes_of_step:
        line = json.dumps(ep.model_dump(), ensure_ascii=False)
        raw.write(line.encode("utf-8"))
        raw.write(b"\n")
    out = io.BytesIO()
    # mtime=0 → byte-stable shards (deterministic re-syncs, content-addressable).
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as gz:
        gz.write(raw.getvalue())
    return out.getvalue()


def write_transcript_shard(
    episodes_of_step: list[Episode],
    backend: StorageBackend,
    run_id: str,
    step: int,
) -> str:
    """Write ``transcripts/{run_id}/step-{step:05d}.jsonl.gz`` for one step."""
    for ep in episodes_of_step:
        if ep.run_id != run_id or ep.step != step:
            raise ValueError(f"episode ({ep.run_id!r}, step={ep.step}) does not match shard ({run_id!r}, step={step})")
    path = transcript_path(run_id, step)
    backend.write(path, serialize_transcript_shard(episodes_of_step))
    return path


def write_setup(setup: dict, backend: StorageBackend, run_id: str) -> str:
    """Write ``index/{run_id}/setup.json`` — the run's distilled config setup."""
    backend.write(setup_path(run_id), json.dumps(setup, indent=2).encode("utf-8"))
    return setup_path(run_id)


def update_runs_registry(backend: StorageBackend, run_id: str, step: int) -> list:
    """Idempotently record that ``run_id`` has data for ``step`` in ``runs.json``."""
    runs = backend.list_runs()
    by_id = {r.run_id: r for r in runs}
    now = datetime.now(timezone.utc).isoformat()
    existing = by_id.get(run_id)
    steps = [step] if existing is None else sorted(set(existing.steps) | {step})

    # Rebuild registry as plain dicts (RunRef is frozen; we serialize to JSON).
    records: list[dict] = []
    for r in runs:
        if r.run_id == run_id:
            continue
        records.append({"run_id": r.run_id, "steps": list(r.steps), "updated": r.updated})
    records.append({"run_id": run_id, "steps": steps, "updated": now})
    records.sort(key=lambda r: r["run_id"])
    backend.write(RUNS_REGISTRY, json.dumps(records, indent=2).encode("utf-8"))
    return records
