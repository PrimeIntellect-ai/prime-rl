"""FastAPI backend + DuckDB-over-parquet index + SSE (the ``rollout-serve`` entrypoint).

The viewer reads everything through an injected :class:`StorageBackend` (LocalFS in
tests, HF/R2 in prod). On cold start it performs a *wake-rebuild*: pull each run's
``index/{run_id}/episodes.parquet`` via the backend into an on-disk cache dir and
register it with DuckDB. The index lives on disk (DuckDB reads parquet directly),
never wholesale in RAM. Transcripts are fetched per drill-down from the gzipped
per-step shards and the requested line is returned as a full, untruncated
:class:`Episode`.

Endpoints (see contracts.md / storage.py for the layout this honors):
  GET /api/runs
  GET /api/runs/{run_id}/episodes?sort=&order=&filter=&limit=&offset=
  GET /api/runs/{run_id}/schedule
  GET /api/runs/{run_id}/setup
  GET /api/episodes/{run_id}/{step}/{line}
  GET /api/compare?runs=a,b&metric=...
  GET /api/stream  (SSE: emits on registry version / new-step change)
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import re
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import duckdb
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

from rollout_viewer.schedule import Schedule, align, infer_run_schedule, step_actor
from rollout_viewer.schema import Episode
from rollout_viewer.storage import (
    INDEX_COLUMNS,
    RunRef,
    StorageBackend,
    index_path,
    setup_path,
    transcript_path,
)
from rollout_viewer.storage_backends import LocalFSBackend

# Sort columns the episodes query accepts. ``diag.*`` and ``metrics.*`` resolve to
# json_extract over the JSON index columns; everything else is a real parquet column.
_SORTABLE_BASE: frozenset[str] = frozenset(
    {"step", "example_id", "reward", "advantage", "is_filtered", "kind", "winner"}
)
_DIAG_PREFIX = "diag."
_METRIC_PREFIX = "metrics."
_TOKENS_PREFIX = "tokens."
# Conservative identifier whitelist for the dotted JSON path segments.
_JSON_PATH_SEG = re.compile(r"^[A-Za-z0-9_./-]+$")


def _poll_seconds() -> float:
    """SSE backend poll cadence (~2s). Overridable for fast tests."""
    return float(os.environ.get("ROLLOUT_VIEWER_POLL_SECONDS", "2.0"))


@dataclass
class _RegistrySnapshot:
    """Cheap version token for the online feed: runs + their steps."""

    runs: dict[str, list[int]]

    @classmethod
    def from_refs(cls, refs: list[RunRef]) -> "_RegistrySnapshot":
        return cls(runs={r.run_id: sorted(r.steps) for r in refs})


class IndexCache:
    """On-disk DuckDB index over the per-run ``episodes.parquet`` pulled via the backend.

    The wake-rebuild copies each run's parquet from the backend into ``cache_dir``
    and reads it through DuckDB (zero-copy parquet scan). The DuckDB connection is
    in-memory but only holds query state — the data stays on disk in the cached
    parquet files. ``refresh`` is idempotent and re-pulls so new steps appear.
    """

    def __init__(self, backend: StorageBackend, cache_dir: Path) -> None:
        self.backend = backend
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(database=":memory:")
        self._lock = threading.Lock()
        # run_id -> absolute path of its cached parquet (None when no index yet).
        self._parquet: dict[str, Path] = {}
        # Observers fired after every successful re-pull, so caches derived from
        # the index (e.g. the per-run schedule) invalidate when steps appear.
        self._on_refresh: list[Callable[[], None]] = []

    def on_refresh(self, hook: Callable[[], None]) -> None:
        self._on_refresh.append(hook)

    def refresh(self) -> None:
        """Re-pull each run's index parquet from the backend onto disk."""
        with self._lock:
            refs = self.backend.list_runs()
            seen: set[str] = set()
            for ref in refs:
                seen.add(ref.run_id)
                logical = index_path(ref.run_id)
                if not self.backend.exists(logical):
                    # A registered run with no index parquet yet: skip — its
                    # episodes endpoint will 404 until the sync writes one.
                    self._parquet.pop(ref.run_id, None)
                    continue
                data = self.backend.read(logical)
                dst = self.cache_dir / f"{ref.run_id}.parquet"
                dst.write_bytes(data)
                self._parquet[ref.run_id] = dst
            for stale in set(self._parquet) - seen:
                self._parquet.pop(stale, None)
        for hook in self._on_refresh:
            hook()

    def parquet_for(self, run_id: str) -> Path:
        with self._lock:
            p = self._parquet.get(run_id)
        if p is not None and p.exists():
            return p
        # Lazy refresh-on-miss: the online flow syncs runs/steps AFTER serve is up,
        # so a just-synced run isn't in the cache until the next SSE poll. Re-pull
        # once before giving up — but only if the registry actually knows the run,
        # so a bogus id doesn't trigger a full re-pull on every request.
        if any(ref.run_id == run_id for ref in self.backend.list_runs()):
            self.refresh()
            with self._lock:
                p = self._parquet.get(run_id)
            if p is not None and p.exists():
                return p
        raise HTTPException(status_code=404, detail=f"no index for run {run_id!r}")

    def known_runs(self) -> list[str]:
        with self._lock:
            return list(self._parquet)


# A diagnostics-rollup key is dotted ``<scope>.<quantity>.<field>`` (e.g.
# ``episode.mismatch_kl.mean``). The quantity blocks carry sum/count so a rollup
# field re-aggregates as Σsum/Σcount; quantities are pinned in contracts.md.
_DIAG_QUANTITIES: frozenset[str] = frozenset({"importance_ratio", "mismatch_kl", "entropy"})


def _json_double_expr(col: str, path: str) -> str:
    """DuckDB expression: extract dotted ``path`` from JSON ``col`` as a DOUBLE."""
    if not _JSON_PATH_SEG.match(path):
        raise HTTPException(status_code=400, detail=f"bad json path: {path!r}")
    json_pointer = "$." + ".".join(f'"{seg}"' for seg in path.split("."))
    return f"TRY_CAST(json_extract_string({col}, '{json_pointer}') AS DOUBLE)"


def _select_expr(sort: str) -> str:
    """Resolve a sort key to a DuckDB expression over the episodes parquet.

    ``diag.<path>`` / ``metrics.<path>`` index into the JSON columns; bare names
    must be in the whitelist. Anything else raises (fail loud — no silent fallback
    to a default column that would hide a typo'd sort key).
    """
    if sort in _SORTABLE_BASE:
        return sort
    for prefix, col in (
        (_DIAG_PREFIX, "diagnostics"),
        (_METRIC_PREFIX, "metrics"),
        (_TOKENS_PREFIX, "tokens"),
    ):
        if sort.startswith(prefix):
            return _json_double_expr(col, sort[len(prefix) :])
    raise HTTPException(status_code=400, detail=f"unsortable key: {sort!r}")


def _compare_exprs(metric: str) -> tuple[str, str]:
    """Resolve a ``compare`` metric to ``(value_sql, n_sql)`` aggregates per step.

    Two regimes, per contracts.md → "Diagnostics rollup schema":
      - **Diagnostics rollup field** (``diag.<scope>.<quantity>.<field>`` where
        ``<quantity>`` is a known per-token quantity): re-aggregate the underlying
        ValueSummary as ``SUM(sum) / SUM(count)`` — the only mean-of-means-free
        cross-episode aggregate — with ``n = SUM(count)`` (total token count
        contributing). The requested ``<field>`` (mean/p99/...) is ignored: a true
        cross-episode mean is Σsum/Σcount regardless, and quantiles don't merge.
      - **Plain metric column** (``metrics.<key>``, ``diag.<scope>.status...``, or a
        base column): ``value = SUM(expr) / COUNT(expr)`` with ``n = COUNT(expr)``.
        ``COUNT`` skips SQL NULLs, so a missing/masked per-episode value drops out of
        both numerator and denominator — never an AVG over per-episode means and
        never a NULL-inflated denominator.
    """
    if metric.startswith(_DIAG_PREFIX):
        path = metric[len(_DIAG_PREFIX) :]
        segs = path.split(".")
        # diag.<scope>.<quantity>.<field> → re-aggregate the ValueSummary.
        if len(segs) == 3 and segs[1] in _DIAG_QUANTITIES:
            scope, quantity, _field = segs
            sum_expr = _json_double_expr("diagnostics", f"{scope}.{quantity}.sum")
            count_expr = _json_double_expr("diagnostics", f"{scope}.{quantity}.count")
            value_sql = f"SUM({sum_expr}) / NULLIF(SUM({count_expr}), 0)"
            n_sql = f"CAST(SUM({count_expr}) AS BIGINT)"
            return value_sql, n_sql

    expr = _select_expr(metric)
    value_sql = f"SUM({expr}) / NULLIF(COUNT({expr}), 0)"
    n_sql = f"COUNT({expr})"
    return value_sql, n_sql


def create_app(backend: StorageBackend, *, cache_dir: str | os.PathLike[str] | None = None) -> FastAPI:
    """Build the FastAPI app with a concrete ``backend`` injected."""
    cdir = (
        Path(cache_dir)
        if cache_dir is not None
        else Path(os.environ.get("ROLLOUT_VIEWER_CACHE", ".rollout-viewer-cache"))
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Cold-start wake-rebuild: pull all run indices onto disk.
        app.state.index.refresh()
        yield

    app = FastAPI(title="rollout-viewer", lifespan=lifespan)
    app.state.backend = backend
    app.state.index = IndexCache(backend, cdir)
    # Per-run inferred schedule, memoized by run_id. Invalidated on every index
    # refresh (new step → the modal turn pattern can move), so a growing run
    # re-infers its schedule instead of serving a stale one.
    app.state.schedule_cache: dict[str, Schedule] = {}
    app.state.schedule_lock = threading.Lock()

    def _invalidate_schedules() -> None:
        with app.state.schedule_lock:
            app.state.schedule_cache.clear()

    app.state.index.on_refresh(_invalidate_schedules)

    def _run_episodes(run_id: str) -> list[Episode]:
        """Load every retained episode of a run (all step shards) as Episodes.

        Used to infer the run schedule and to back the examples scrubber. Reads
        the transcript shards directly via the backend; non-retained episodes have
        no shard line and are simply absent here (their index row still exists).
        """
        parquet = app.state.index.parquet_for(run_id)
        cur = app.state.index.con.execute("SELECT DISTINCT step FROM read_parquet(?) ORDER BY step", [str(parquet)])
        steps = [int(r[0]) for r in cur.fetchall()]
        episodes: list[Episode] = []
        for step in steps:
            shard = transcript_path(run_id, step)
            if not backend.exists(shard):
                continue
            blob = gzip.decompress(backend.read(shard))
            for raw_line in blob.splitlines():
                if raw_line.strip():
                    episodes.append(Episode(**json.loads(raw_line)))
        return episodes

    def _schedule_for(run_id: str) -> Schedule:
        with app.state.schedule_lock:
            sched = app.state.schedule_cache.get(run_id)
        if sched is not None:
            return sched
        episodes = _run_episodes(run_id)
        if not episodes:
            raise HTTPException(
                status_code=404,
                detail=f"no retained episodes to infer a schedule for run {run_id!r}",
            )
        sched = infer_run_schedule(episodes)
        with app.state.schedule_lock:
            app.state.schedule_cache[run_id] = sched
        return sched

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return [{"run_id": r.run_id, "steps": r.steps, "updated": r.updated} for r in backend.list_runs()]

    @app.get("/api/runs/{run_id}/episodes")
    def list_episodes(
        run_id: str,
        sort: str = Query("reward"),
        order: str = Query("desc"),
        filter: str | None = Query(None),
        limit: int = Query(100, ge=1, le=10_000),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        parquet = app.state.index.parquet_for(run_id)
        order_kw = order.lower()
        if order_kw not in ("asc", "desc"):
            raise HTTPException(status_code=400, detail=f"bad order: {order!r}")
        sort_expr = _select_expr(sort)

        # Index rows only — never transcripts. NULLS LAST so masked/absent metrics
        # sort to the bottom regardless of direction (a None is not a real value).
        cols = ", ".join(INDEX_COLUMNS)
        sql = f"SELECT {cols} FROM read_parquet(?)"
        params: list[Any] = [str(parquet)]
        if filter:
            # Raw DuckDB WHERE predicate over INDEX_COLUMNS. Free-form by design
            # for the private / single-tenant HF-Space tier: it runs READ-ONLY over
            # the cached parquet (DuckDB connection only ever SELECTs), and a
            # malformed predicate raises (caught -> 400). The cached parquet is a
            # throwaway pull, so even a pathological predicate touches no source data.
            # TODO: structured filter DSL before any public deploy — a raw predicate
            # is acceptable only while the deployment is private/trusted.
            sql += f" WHERE {filter}"
        sql += f" ORDER BY {sort_expr} {order_kw.upper()} NULLS LAST LIMIT ? OFFSET ?"
        params += [limit, offset]

        try:
            cur = app.state.index.con.execute(sql, params)
            colnames = [d[0] for d in cur.description]
            rows = cur.fetchall()
        except duckdb.Error as exc:
            raise HTTPException(status_code=400, detail=f"query failed: {exc}") from exc

        out: list[dict[str, Any]] = []
        for row in rows:
            rec = dict(zip(colnames, row))
            # JSON-encoded index columns -> structured values for the client.
            for jcol in ("members", "answers", "metrics", "diagnostics", "tokens"):
                v = rec.get(jcol)
                if isinstance(v, str):
                    rec[jcol] = json.loads(v)
            out.append(rec)
        return {"run_id": run_id, "rows": out, "count": len(out), "offset": offset}

    @app.get("/api/episodes/{run_id}/{step}/{line}")
    def get_episode(run_id: str, step: int, line: int) -> dict[str, Any]:
        shard = transcript_path(run_id, step)
        if line < 0:
            # NOT_RETAINED sentinel (contracts.md → Retention selection): under
            # ρ<1 this episode's transcript was deliberately not shipped, though
            # its metrics + diagnostics ARE in the index. Answer explicitly — never
            # fetch line -1, never 404-as-if-missing (it isn't missing, it's elided).
            return {
                "retained": False,
                "run_id": run_id,
                "step": step,
                "transcript_line": line,
                "transcript_shard": shard,
                "detail": (
                    "transcript not retained (ρ<1); metrics + diagnostics are in "
                    "the index but the full transcript was not shipped to the store"
                ),
            }
        if not backend.exists(shard):
            raise HTTPException(status_code=404, detail=f"no transcript shard: {shard}")
        blob = gzip.decompress(backend.read(shard))
        # One stripped Episode.model_dump() per line; return the FULL episode at
        # `line`. No truncation of messages/steps anywhere on this path.
        target: dict[str, Any] | None = None
        for i, raw_line in enumerate(blob.splitlines()):
            if i == line:
                target = json.loads(raw_line)
                break
        if target is None:
            raise HTTPException(
                status_code=404,
                detail=f"line {line} not in shard {shard}",
            )
        # Re-validate against the schema so a malformed shard fails loud here, not
        # silently in the client. The dumped form is returned verbatim (untruncated).
        episode = Episode(**target)
        out = episode.model_dump()
        # Attach this episode's alignment to the run schedule (regime-generic) so
        # the timeline can render matched/truncated/deviated/extra per turn without
        # the client re-deriving the schedule. Transcript content stays byte-identical.
        schedule = _schedule_for(run_id)
        out["alignment"] = [
            {
                "index": t.index,
                "status": t.status,
                "actor": t.slot.actor if t.slot is not None else step_actor(t.step),
                "phase": (t.slot.phase if t.slot is not None else None)
                or (t.step.phase if t.step is not None else None),
            }
            for t in align(episode, schedule)
        ]
        return out

    def _schedule_payload(schedule: Schedule) -> dict[str, Any]:
        return {
            "regime": schedule.regime,
            "source": schedule.source,
            "slots": [{"index": s.index, "actor": s.actor, "phase": s.phase} for s in schedule.slots],
        }

    @app.get("/api/runs/{run_id}/schedule")
    def run_schedule(run_id: str) -> dict[str, Any]:
        """The run's inferred (or config) schedule — the timeline's expected axis."""
        return _schedule_payload(_schedule_for(run_id))

    @app.get("/api/runs/{run_id}/setup")
    def run_setup(run_id: str) -> dict[str, Any]:
        """The run's declared setup (distilled config) + its recovered schedule.

        ``setup.json`` is the config side written at sync time (``--configs-dir``);
        ``schedule`` is the protocol we recover from the rollout data, so the
        frontend can render declared-config alongside recovered-protocol in one view.
        """
        logical = setup_path(run_id)
        if not backend.exists(logical):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"no setup for run {run_id!r} — sync it with --configs-dir to "
                    "distill the run's orchestrator.toml into setup.json"
                ),
            )
        setup = json.loads(backend.read(logical))
        setup["schedule"] = _schedule_payload(_schedule_for(run_id))
        return setup

    @app.get("/api/runs/{run_id}/examples/{example_id}")
    def run_example(run_id: str, example_id: str) -> dict[str, Any]:
        """A question's rollouts across training steps (the step-scrubber's data).

        Queries the index parquet WHERE example_id=?, grouped by step. Each entry
        is a pointer the client resolves via /api/episodes — never the transcript
        itself (the step scrubber is an index, not a bulk transcript pull).
        """
        parquet = app.state.index.parquet_for(run_id)
        cur = app.state.index.con.execute(
            "SELECT step, transcript_line, trajectory_id, reward, winner, kind "
            "FROM read_parquet(?) WHERE example_id = ? "
            "ORDER BY step, transcript_line",
            [str(parquet), example_id],
        )
        rows = cur.fetchall()
        by_step: dict[int, list[dict[str, Any]]] = {}
        for step, line, traj, reward, winner, kind in rows:
            by_step.setdefault(int(step), []).append(
                {
                    "step": int(step),
                    "transcript_line": int(line),
                    "trajectory_id": traj,
                    "reward": reward,
                    "winner": winner,
                    "kind": kind,
                }
            )
        return {
            "example_id": example_id,
            "by_step": [{"step": step, "episodes": by_step[step]} for step in sorted(by_step)],
        }

    @app.get("/api/compare")
    def compare(
        runs: str = Query(..., description="comma-separated run ids"),
        metric: str = Query(..., description="diag.* / metrics.* / a base column"),
    ) -> dict[str, Any]:
        run_ids = [r for r in (s.strip() for s in runs.split(",")) if r]
        if not run_ids:
            raise HTTPException(status_code=400, detail="no runs given")
        # value/n aggregates per the pinned rule: Σsum/Σcount for diagnostics
        # rollup fields, SUM(expr)/COUNT(expr) for plain metrics. NEVER AVG over
        # per-episode means, and COUNT skips NULLs so the denominator is the true
        # number of contributing episodes, not COUNT(*).
        value_sql, n_sql = _compare_exprs(metric)
        series: dict[str, list[dict[str, float]]] = {}
        for run_id in run_ids:
            parquet = app.state.index.parquet_for(run_id)
            sql = f"SELECT step, {value_sql} AS value, {n_sql} AS n FROM read_parquet(?) GROUP BY step ORDER BY step"
            try:
                cur = app.state.index.con.execute(sql, [str(parquet)])
                rows = cur.fetchall()
            except duckdb.Error as exc:
                raise HTTPException(status_code=400, detail=f"compare query failed: {exc}") from exc
            # n is NULL only when no episode in the step carried the quantity
            # (SUM over an empty set); report it as 0, an honest count, not a guess.
            series[run_id] = [{"step": int(s), "value": v, "n": int(n) if n is not None else 0} for (s, v, n) in rows]
        return {"metric": metric, "series": series}

    @app.get("/api/stream")
    async def stream(request: Request) -> EventSourceResponse:
        async def gen() -> AsyncIterator[dict[str, Any]]:
            last: _RegistrySnapshot | None = None
            poll = _poll_seconds()
            while True:
                if await request.is_disconnected():
                    break
                refs = await asyncio.to_thread(backend.list_runs)
                snap = _RegistrySnapshot.from_refs(refs)
                if snap != last:
                    # Registry version moved (new run or new step): refresh the
                    # on-disk index and notify clients to re-pull.
                    await asyncio.to_thread(app.state.index.refresh)
                    last = snap
                    yield {
                        "event": "update",
                        "data": json.dumps({"runs": {rid: steps for rid, steps in snap.runs.items()}}),
                    }
                await asyncio.sleep(poll)

        return EventSourceResponse(gen())

    _web_a = Path(__file__).parent / "web_a"
    _web_b = Path(__file__).parent / "web_b"

    @app.get("/", include_in_schema=False)
    def _index() -> HTMLResponse:
        # Chooser: flip between the two design directions and judge live.
        return HTMLResponse(
            "<!doctype html><meta charset=utf-8><title>rollout-viewer</title>"
            '<body style="background:oklch(20% .012 235);color:oklch(86% .02 235);'
            "font:15px 'Spline Sans Mono',ui-monospace,monospace;display:grid;"
            'place-items:center;height:100vh;margin:0;gap:22px">'
            '<div style="letter-spacing:.04em;color:oklch(62% .02 235)">rollout-viewer — pick a direction</div>'
            '<div style="display:flex;gap:22px">'
            '<a href="/a/" style="color:oklch(80% .13 188);border:1px solid oklch(34% .02 235);'
            'padding:20px 30px;border-radius:8px;text-decoration:none">A · instrument / oscilloscope</a>'
            '<a href="/b/" style="color:oklch(74% .07 245);border:1px solid oklch(34% .02 235);'
            'padding:20px 30px;border-radius:8px;text-decoration:none">B · wire-protocol analyzer</a>'
            "</div></body>"
        )

    # Zero-build static SPAs (vanilla ES modules), served under their direction path.
    from fastapi.staticfiles import StaticFiles

    # Frequently edited dev bundle → never let the browser serve a stale SPA without
    # checking. ``no-cache`` still permits ETag/Last-Modified revalidation (cheap 304s),
    # it just forbids using a cached copy without revalidating first.
    class _NoCacheStatic(StaticFiles):
        async def get_response(self, path: str, scope):
            resp = await super().get_response(path, scope)
            resp.headers["Cache-Control"] = "no-cache"
            return resp

    if _web_a.is_dir():
        app.mount("/a", _NoCacheStatic(directory=_web_a, html=True), name="web_a")
    if _web_b.is_dir():
        app.mount("/b", _NoCacheStatic(directory=_web_b, html=True), name="web_b")

    return app


def main() -> None:
    """``rollout-serve`` entrypoint: serve over a LocalFS store via uvicorn."""
    import uvicorn

    store_root = os.environ.get("ROLLOUT_VIEWER_STORE")
    if not store_root:
        raise SystemExit("ROLLOUT_VIEWER_STORE is required (path to the store root: runs.json + index/ + transcripts/)")
    backend = LocalFSBackend(store_root)
    app = create_app(backend)
    host = os.environ.get("ROLLOUT_VIEWER_HOST", "127.0.0.1")
    port = int(os.environ.get("ROLLOUT_VIEWER_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
