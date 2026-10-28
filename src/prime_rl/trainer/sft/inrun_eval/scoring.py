"""pass@k of a flow store, the way the collection's measure runs count it (prime_flow ``measure.pass_at_k``).

A store's trace index (``<store>/traces/index.jsonl``, one ``TraceRow`` per landed trace) holds every solver draw:
``labels.flow.job`` (``<task slug>-<key>-qc``), ``labels.flow.call`` (``solver/draw<n>-<version>``), ``ok``,
``error``, ``stop_condition`` and ``reward``. Per task and draw the last *finished* row counts (ok and not stopped by
an error; HarnessError reruns land rows of their own); a draw without one is an error, a draw without any row is
missing. Errors and missing draws count as failures, and are reported separately.

The one exception is a source whose flow run the optional safety cap (``[inrun_eval] budget_hours``) stopped: its
draws that were still running at the stop, or had not started, are ``cut``. Which ones: prime-flow's call record of
each draw (``<store>/calls/<job>/<digest>.json``): ``open`` (running), or ``failed`` / ``finished`` at or after the
stop (the stop ended it, or its row never landed), or no record and no row (never started). A draw that had already
errored before the stop (its record ``failed`` earlier, or an error row and no record) stays an error. A cut draw is
not an outcome: it is excluded from pass@k, which is then computed per task over its n = k - cut remaining draws, for
every reported k <= n (a task left with fewer than k draws is not in pass@k's average; ``pass@<k>_tasks`` says how
many are). The summary's ``coverage`` states the share of planned draws that are not cut. Without a cap nothing is
cut.

pass@k = 1 - C(n - c, k) / C(n, k) per task (c = draws with reward >= the pass threshold), averaged over tasks; with
n = k = 8, pass@1 is the mean of c / 8 and pass@8 the share of tasks solved at least once.
"""

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from prime_rl.trainer.sft.inrun_eval.tasks import TaskRow, job_slug, slug_index

DRAW_CALL = re.compile(r"^solver/draw(?P<n>\d+)(?:-[0-9A-Za-z]+)?$")
SOLVER_SEAT = "solver"
CALLS = "calls"
"""prime-flow's call records under a store (``calls/<job>/<digest>.json``, ``prime_flow.calls`` at ed23e23)."""


@dataclass(frozen=True)
class CallState:
    """A solver draw's prime-flow call record: ``open`` (running, or cut off), ``finished`` or ``failed``."""

    status: str
    updated: float | None


def read_call_states(store: Path) -> dict[tuple[str, int], CallState] | None:
    """(job, draw) -> the draw's call record state, None when the store keeps no call records."""
    root = Path(store) / CALLS
    if not root.is_dir():
        return None
    states: dict[tuple[str, int], CallState] = {}
    for path in root.glob("*/*.json"):
        try:
            record = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError):
            continue
        identity = record.get("identity") if isinstance(record, dict) else None
        if not isinstance(identity, dict) or record.get("seat") not in (None, SOLVER_SEAT):
            continue
        match = DRAW_CALL.match(str(identity.get("call") or ""))
        if match is None or not identity.get("job"):
            continue
        key = (str(identity["job"]), int(match.group("n")))
        updated = record.get("updated")
        state = CallState(str(record.get("status") or "open"), float(updated) if updated is not None else None)
        if key not in states or (state.updated or 0) >= (states[key].updated or 0):
            states[key] = state
    return states


def cut_at_stop(state: CallState | None, landed: bool, stopped_at: float | None) -> bool:
    """Whether a draw without a finished row was cut by the stop (else it had errored before it)."""
    if state is None:
        return not landed  # never started; an error row without a record had errored
    if state.status == "open":
        return True
    return stopped_at is None or state.updated is None or state.updated >= stopped_at


def pass_at_k(n: int, c: int, k: int) -> float:
    """The unbiased pass@k estimate from c passing of n draws."""
    if k > n:
        raise ValueError(f"pass@{k} needs at least {k} draws, got {n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def report_ks(n: int) -> list[int]:
    """The k reported for n draws per task: 1, 2, 4, ... up to n, and n itself."""
    ks, k = [], 1
    while k < n:
        ks.append(k)
        k *= 2
    return ks + [n]


def stopped_by_error(stop_condition: str | None) -> bool:
    stop = stop_condition or ""
    return stop == "error" or stop.endswith("_error")


def is_finished(row: dict) -> bool:
    return bool(row.get("ok")) and not stopped_by_error(row.get("stop_condition"))


def read_jsonl(path: Path) -> list[dict]:
    """Every complete JSON line (a torn last line from a killed writer is skipped)."""
    rows = []
    if not path.is_file():
        return rows
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def draw_rows(rows: list[dict]) -> list[tuple[str, int, dict]]:
    """(job, draw, row) of the solver draw rows, in file order."""
    out = []
    for row in rows:
        labels = row.get("labels") or {}
        flow = labels.get("flow") if isinstance(labels, dict) else None
        if not isinstance(flow, dict):
            continue
        seat = flow.get("seat") or row.get("agent")
        match = DRAW_CALL.match(str(flow.get("call") or ""))
        if seat != SOLVER_SEAT or match is None or not flow.get("job"):
            continue
        out.append((str(flow["job"]), int(match.group("n")), row))
    return out


def task_names_from_traces(traces_path: Path, trace_ids: set[str]) -> dict[str, str]:
    """trace id -> task name, read from the store's episodes (only lines that mention a wanted id are parsed)."""
    found: dict[str, str] = {}
    if not trace_ids or not traces_path.is_file():
        return found
    needles = [tid.encode() for tid in trace_ids]
    with open(traces_path, "rb") as f:
        for line in f:
            if not any(needle in line for needle in needles):
                continue
            try:
                episode = json.loads(line)
            except json.JSONDecodeError:
                continue
            traces = episode.get("traces") if isinstance(episode, dict) else None
            for trace in traces if isinstance(traces, list) else [episode]:
                if not isinstance(trace, dict) or trace.get("id") not in trace_ids:
                    continue
                name = ((trace.get("task") or {}).get("data") or {}).get("name")
                if name:
                    found[trace["id"]] = str(name)
    return found


@dataclass
class Draw:
    draw: int
    status: str  # passed | failed | error | missing | cut
    reward: float | None = None
    stop_condition: str | None = None
    error: str | None = None
    trace_id: str | None = None
    rows: int = 0


@dataclass
class TaskScore:
    task: str
    source: str
    split: str
    n: int
    """Planned draws (the configured k)."""
    c: int = 0
    cut: int = 0
    """Draws the safety cap cut (excluded from pass@k)."""
    draws: list[Draw] = field(default_factory=list)

    @property
    def graded(self) -> int:
        """The draws that count: the planned ones minus the cut ones."""
        return self.n - self.cut

    def to_dict(self) -> dict:
        return {**{k: v for k, v in asdict(self).items() if k != "draws"}, "draws": [asdict(d) for d in self.draws]}


def score_rows(
    rows: list[dict],
    tasks: list[TaskRow],
    source: str,
    k: int,
    pass_threshold: float,
    traces_path: Path | None = None,
    cut: bool = False,
    calls: dict[tuple[str, int], CallState] | None = None,
    stopped_at: float | None = None,
) -> tuple[list[TaskScore], dict]:
    """Score one source's index rows against its expected tasks. Returns the per-task scores (every expected task,
    plus unexpected ones the store holds) and diagnostics (unmapped jobs, draws beyond k). ``cut``: the safety cap
    stopped this source's run (at ``stopped_at``), so its draws still running then (``calls``: the store's call
    records, ``cut_at_stop``) or never started are ``cut`` instead of error / missing."""
    names = [row.name for row in tasks]
    by_name = {row.name: row for row in tasks}
    slugs, ambiguous = slug_index(names)
    draws = draw_rows(rows)

    job_task: dict[str, str] = {}
    unresolved: dict[str, set[str]] = {}
    for job, _, row in draws:
        if job in job_task or job in unresolved:
            continue
        key = job_slug(job)
        if key is not None and key in slugs and key not in ambiguous:
            job_task[job] = slugs[key]
        else:
            unresolved[job] = set()
    if unresolved:
        for job, _, row in draws:
            if job in unresolved and row.get("id"):
                unresolved[job].add(str(row["id"]))
        from_traces = task_names_from_traces(traces_path, set().union(*unresolved.values())) if traces_path else {}
        for job, ids in unresolved.items():
            hits = {from_traces[i] for i in ids if i in from_traces}
            if len(hits) == 1:
                job_task[job] = hits.pop()

    per_task_draw: dict[tuple[str, int], list[dict]] = {}
    beyond_k = 0
    unmapped_jobs = sorted(job for job in unresolved if job not in job_task)
    for job, n, row in draws:
        task = job_task.get(job)
        if task is None:
            continue
        if n >= k:
            beyond_k += 1
            continue
        per_task_draw.setdefault((task, n), []).append(row)

    # Each task's draw call states (a task has one job; the latest record wins should it have more).
    task_calls: dict[tuple[str, int], CallState] = {}
    for (job, n), state in (calls or {}).items():
        task = job_task.get(job)
        if task is None:
            key = job_slug(job)
            task = slugs.get(key) if key is not None and key not in ambiguous else None
        if task is None:
            continue
        old = task_calls.get((task, n))
        if old is None or (state.updated or 0) >= (old.updated or 0):
            task_calls[(task, n)] = state

    scored_names = list(names) + sorted({t for (t, _) in per_task_draw} - set(names))
    scores: list[TaskScore] = []
    for name in scored_names:
        info = by_name.get(name)
        score = TaskScore(task=name, source=source, split=info.split if info else "other", n=k)
        for n in range(k):
            landed = per_task_draw.get((name, n), [])
            finished = [row for row in landed if is_finished(row)]
            if finished:
                row = finished[-1]
                reward = row.get("reward")
                passed = reward is not None and float(reward) >= pass_threshold
                draw = Draw(
                    draw=n,
                    status="passed" if passed else "failed",
                    reward=None if reward is None else float(reward),
                    stop_condition=row.get("stop_condition"),
                    trace_id=row.get("id"),
                    rows=len(landed),
                )
                score.c += passed
            elif landed:
                row = landed[-1]
                draw = Draw(
                    draw=n,
                    status="cut" if cut and cut_at_stop(task_calls.get((name, n)), True, stopped_at) else "error",
                    stop_condition=row.get("stop_condition"),
                    error=row.get("error"),
                    trace_id=row.get("id"),
                    rows=len(landed),
                )
            elif cut:
                state = task_calls.get((name, n))
                draw = Draw(draw=n, status="cut" if cut_at_stop(state, False, stopped_at) else "error")
                if draw.status == "error":
                    draw.error = f"call record {state.status} before the stop, no row landed"
            else:
                draw = Draw(draw=n, status="missing")
            score.cut += draw.status == "cut"
            score.draws.append(draw)
        scores.append(score)
    diagnostics = {
        "rows": len(rows),
        "draw_rows": len(draws),
        "unmapped_jobs": unmapped_jobs,
        "draw_rows_beyond_k": beyond_k,
        "unexpected_tasks": sorted({t for (t, _) in per_task_draw} - set(names)),
        "row_errors": dict(Counter(str(row.get("error")) for _, _, row in draws if row.get("error"))),
        "cut_by_cap": cut,
        "stopped_at": stopped_at,
        "call_records": None if calls is None else len(calls),
    }
    return scores, diagnostics


def summarize(scores: list[TaskScore]) -> dict:
    """pass@k and counts over a group of tasks (all with the same planned n). Cut draws are excluded: each task's
    pass@k uses its remaining draws, and only tasks with at least k of them enter pass@k's average."""
    if not scores:
        return {"tasks": 0}
    n = scores[0].n
    if any(score.n != n for score in scores):
        raise ValueError("tasks with different numbers of planned draws in one group")
    statuses = Counter(draw.status for score in scores for draw in score.draws)
    rewards = [draw.reward for score in scores for draw in score.draws if draw.reward is not None]
    stops = Counter(
        draw.stop_condition or "-" for score in scores for draw in score.draws if draw.status not in ("missing", "cut")
    )
    graded = statuses["passed"] + statuses["failed"]
    planned = len(scores) * n
    passk: dict[str, float | int | None] = {}
    for k in report_ks(n):
        eligible = [s for s in scores if s.graded >= k]
        passk[f"pass@{k}"] = sum(pass_at_k(s.graded, s.c, k) for s in eligible) / len(eligible) if eligible else None
        if statuses["cut"]:
            passk[f"pass@{k}_tasks"] = len(eligible)
    out = {
        "tasks": len(scores),
        "draws_per_task": n,
        "rollouts": planned,
        "passed": statuses["passed"],
        "failed": statuses["failed"],
        "errored": statuses["error"],
        "missing": statuses["missing"],
        "cut": statuses["cut"],
        **passk,
        "tasks_solved": sum(score.c > 0 for score in scores),
        "pass@1_graded_only": statuses["passed"] / graded if graded else None,
        "mean_reward_graded": sum(rewards) / len(rewards) if rewards else None,
        "stop_conditions": dict(sorted(stops.items())),
        "coverage": {
            "draws_counted": planned - statuses["cut"],
            "draws_planned": planned,
            "fraction": (planned - statuses["cut"]) / planned if planned else None,
            "tasks_with_all_draws": sum(s.cut == 0 for s in scores),
            "tasks_without_draws": sum(s.graded == 0 for s in scores),
            "note": "complete: every planned draw counts"
            if not statuses["cut"]
            else "the safety cap cut some draws: they are excluded from pass@k, which is over the remaining draws "
            "(pass@<k>_tasks tasks per k); not comparable to a complete run",
        },
    }
    return out


def summary_by_group(scores: list[TaskScore]) -> dict:
    """Overall, per split (val / train / other) and per source."""
    out = {"overall": summarize(scores), "split": {}, "source": {}}
    for split in sorted({s.split for s in scores}):
        out["split"][split] = summarize([s for s in scores if s.split == split])
    for source in sorted({s.source for s in scores}):
        out["source"][source] = summarize([s for s in scores if s.source == source])
    return out


def score_store(
    store: Path,
    tasks: list[TaskRow],
    source: str,
    k: int,
    pass_threshold: float,
    cut: bool = False,
    stopped_at: float | None = None,
) -> tuple[list[TaskScore], dict]:
    """Score a flow store (``<store>/traces/index.jsonl``, task names of unmapped jobs from ``<store>/traces.jsonl``).
    ``cut``: the safety cap stopped its run at ``stopped_at`` (its call records tell which draws were running)."""
    rows = read_jsonl(store / "traces" / "index.jsonl")
    calls = read_call_states(store) if cut else None
    return score_rows(
        rows,
        tasks,
        source,
        k,
        pass_threshold,
        traces_path=store / "traces.jsonl",
        cut=cut,
        calls=calls,
        stopped_at=stopped_at,
    )


def progress(store: Path) -> dict[str, int]:
    """A store's landed solver draws so far (the heartbeat): ``rollouts_done`` (draws with a finished row), ``rows``
    (draw rows, reruns and errors included)."""
    draws = draw_rows(read_jsonl(Path(store) / "traces" / "index.jsonl"))
    done = {(job, n) for job, n, row in draws if is_finished(row)}
    return {"rollouts_done": len(done), "rows": len(draws)}
