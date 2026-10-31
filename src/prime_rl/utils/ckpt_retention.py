"""Keep-3 checkpoint retention (see gb_settings.py): keep the last checkpoint and the best one per held-out
eval metric, plus every checkpoint whose eval is still pending; delete the rest during the run.

Two processes share the run's checkpoint directory:

- The orchestrator runs the online evals. It writes one record per (eval env, step) to
  ``<ckpt_dir>/eval_results/<env>/step_<N>.json`` (``EvalRecorder``): ``running`` when the eval is
  triggered, then ``done`` (with the metric means and the error rate) or ``failed``. A checkpoint step the
  env was not evaluated at gets ``skipped``. On startup it reconciles the records with the resume point
  (``EvalRecorder.reconcile``). The ``step_*`` glob of the checkpoint dir does not match ``eval_results``.
- The trainer master is the only deleter (``Keep3Retention``). It recomputes the keep set from the
  directory and the records on every step, so the policy survives restarts without its own state, and it
  never touches the step it saved last or anything above it.

The selection itself (``select_keep``) is pure. A step forked from another run (``[resume] dir``) is
external: it can hold a best role through its record in this run, but it is never deleted.

This module only uses the standard library, so it can be tested without the GPU stack.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

EVAL_RESULTS_DIR = "eval_results"
RETENTION_FILE = "retention.json"
WRITER_FILE = "writer.json"

# The file each component writes last when it saves a checkpoint step (mirrors utils/pathing.py).
CKPT_COMPLETE_MARKERS = {"trainer": ".metadata", "orchestrator": "progress.pt"}

RUNNING = "running"
DONE = "done"
FAILED = "failed"
LOST = "lost"
SKIPPED = "skipped"
TERMINAL = frozenset({DONE, FAILED, LOST, SKIPPED})
STATUSES = TERMINAL | {RUNNING}

_STEP_DIR_RE = re.compile(r"^step_(\d+)$")
_RECORD_RE = re.compile(r"^step_(\d+)\.json$")

_default_logger = logging.getLogger(__name__)


# ── checkpoint directory ───────────────────────────────────────────────────


@dataclass(frozen=True)
class CkptScan:
    present: tuple[int, ...]
    """Every ``step_N`` directory, complete or not."""

    complete: tuple[int, ...]
    """Steps with both the trainer and the orchestrator completion marker."""

    trainer_complete: tuple[int, ...]
    """Steps with the trainer completion marker (the orchestrator's may be missing)."""


def scan_ckpt_dir(ckpt_dir: Path) -> CkptScan:
    if not ckpt_dir.is_dir():
        return CkptScan((), (), ())
    present = sorted(int(m.group(1)) for p in ckpt_dir.iterdir() if (m := _STEP_DIR_RE.match(p.name)) and p.is_dir())

    def has(step: int, component: str) -> bool:
        return (ckpt_dir / f"step_{step}" / component / CKPT_COMPLETE_MARKERS[component]).exists()

    trainer_complete = [s for s in present if has(s, "trainer")]
    complete = [s for s in trainer_complete if has(s, "orchestrator")]
    return CkptScan(tuple(present), tuple(complete), tuple(trainer_complete))


def atomic_write_json(path: Path, obj) -> None:
    """Write JSON through a temp file and an atomic rename, so a reader never sees a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _utc(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


# ── eval records ───────────────────────────────────────────────────────────


def eval_results_dir(ckpt_dir: Path, env: str | None = None) -> Path:
    base = ckpt_dir / EVAL_RESULTS_DIR
    return base if env is None else base / env


def record_path(ckpt_dir: Path, env: str, step: int) -> Path:
    return eval_results_dir(ckpt_dir, env) / f"step_{step}.json"


def record_steps(ckpt_dir: Path, env: str) -> set[int]:
    """Steps with a record file for one env, readable or not."""
    d = eval_results_dir(ckpt_dir, env)
    if not d.is_dir():
        return set()
    return {int(m.group(1)) for p in d.iterdir() if (m := _RECORD_RE.match(p.name))}


def read_records(ckpt_dir: Path, env: str, log=None) -> dict[int, dict]:
    """``{step: record}`` for one env. Unreadable or mismatched files are skipped with a warning."""
    log = log or _default_logger
    out: dict[int, dict] = {}
    d = eval_results_dir(ckpt_dir, env)
    if not d.is_dir():
        return out
    for p in d.iterdir():
        m = _RECORD_RE.match(p.name)
        if not m:
            continue
        step = int(m.group(1))
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError) as e:
            log.warning(f"Skipping unreadable eval record {p}: {e}")
            continue
        if not isinstance(rec, dict) or rec.get("step") != step or rec.get("status") not in STATUSES:
            log.warning(f"Skipping malformed eval record {p}")
            continue
        out[step] = rec
    return out


def eval_error(metrics: Mapping[str, float | None], env: str) -> float | None:
    """Share of the eval epoch's attempts without a graded episode, or None when the error keys are missing.
    Same rule as genebench-env scripts/auto_export.py: has_error and cancelled share the episode denominator;
    dispatch_failure is over episodes + failures, so the two combine multiplicatively. dispatch_failure is
    logged only when there were attempts, so its absence counts as 0."""
    e = metrics.get(f"eval/{env}/all/has_error/mean")
    if e is None:
        e = metrics.get(f"eval/{env}/all/agent/has_error/mean")
    if e is None:
        return None
    e = min(1.0, float(e) + float(metrics.get(f"eval/{env}/all/cancelled/mean") or 0.0))
    return 1.0 - (1.0 - e) * (1.0 - float(metrics.get(f"eval/{env}/all/dispatch_failure/mean") or 0.0))


def _finite(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


class EvalRecorder:
    """Writes the orchestrator's eval records. Writes are synchronous and tiny (one small JSON per eval epoch
    or checkpoint step); a failed write is logged and never fails the run, since retention treats a step
    without a terminal record as pending (kept)."""

    def __init__(
        self,
        ckpt_dir: Path,
        *,
        ckpt_interval: int | None,
        log=None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.ckpt_dir = ckpt_dir
        self.ckpt_interval = ckpt_interval
        self.log = log or _default_logger
        self.now = now
        self.last_trigger_step: int | None = None

    def _write(self, rec: dict) -> None:
        rec["updated_at"] = _utc(self.now())
        try:
            atomic_write_json(record_path(self.ckpt_dir, rec["env"], rec["step"]), rec)
        except OSError as e:
            self.log.warning(f"Failed to write eval record {rec['env']} step {rec['step']}: {e}")

    def write_writer_info(self, info: Mapping) -> None:
        """Tell the trainer which process writes the records and with which settings (informational)."""
        try:
            atomic_write_json(eval_results_dir(self.ckpt_dir) / WRITER_FILE, {**info, "started_at": _utc(self.now())})
        except OSError as e:
            self.log.warning(f"Failed to write {WRITER_FILE}: {e}")

    def is_ckpt_step(self, step: int, *, is_final: bool) -> bool:
        return is_final or bool(self.ckpt_interval and step % self.ckpt_interval == 0)

    def on_trigger(
        self,
        step: int,
        *,
        fired: Iterable[str],
        env_names: Iterable[str],
        expected: Mapping[str, int],
        is_final: bool,
    ) -> None:
        """An eval trigger at ``step``: ``running`` for each fired env. At a checkpoint step, an env that did
        not fire gets ``skipped`` (a missing record would hold the checkpoint as pending forever); a leftover
        non-terminal record of an earlier attempt becomes ``lost``."""
        fired = list(fired)
        env_names = list(env_names)
        for env in fired:
            self._write({"env": env, "step": step, "status": RUNNING, "expected": int(expected.get(env, 0))})
        # Policy versions the watcher jumped over were never triggered (cannot happen with the blocking NCCL
        # handshake, but a checkpoint without a record would be held as pending for the rest of the run).
        if self.last_trigger_step is not None and self.ckpt_interval:
            for skipped in range(self.last_trigger_step + 1, step):
                if skipped % self.ckpt_interval == 0:
                    self._mark_not_evaluated(skipped, env_names, "policy version never applied by inference")
        self.last_trigger_step = max(step, self.last_trigger_step or step)
        if self.is_ckpt_step(step, is_final=is_final):
            self._mark_not_evaluated(step, [env for env in env_names if env not in fired], "not due at this step")

    def _mark_not_evaluated(self, step: int, env_names: Iterable[str], reason: str) -> None:
        for env in env_names:
            prev = self._read(env, step)
            if prev is None:
                self._write({"env": env, "step": step, "status": SKIPPED, "reason": reason})
            elif prev["status"] not in TERMINAL:
                self._write({**prev, "status": LOST, "reason": "eval of an earlier attempt, not re-run"})

    def on_done(
        self,
        env: str,
        step: int,
        *,
        metrics: Mapping[str, float | None],
        n_episodes: int,
        n_failures: int,
        n_cancelled: int,
        policy_version: int,
    ) -> dict:
        """A finished eval epoch. Keeps the ``all``-subset metrics (``eval/<env>/all/...``) and the
        policy version; ``eval_error`` is computed like auto_export's."""
        kept = {
            k: v
            for k, v in metrics.items()
            if (k.startswith(f"eval/{env}/all/") or k == f"eval/{env}/policy_version") and _finite(v) is not None
        }
        rec = {
            "env": env,
            "step": step,
            "status": DONE,
            "metrics": kept,
            "eval_error": eval_error(metrics, env),
            "n_episodes": int(n_episodes),
            "n_failures": int(n_failures),
            "n_cancelled": int(n_cancelled),
            "policy_version": int(policy_version),
        }
        self._write(rec)
        return rec

    def terminal_at(self, step: int, env_names: Iterable[str], statuses: Iterable[str] = TERMINAL) -> bool:
        """Every env has a record at ``step`` with one of ``statuses`` (default: any terminal one)."""
        env_names, statuses = list(env_names), set(statuses)
        return bool(env_names) and all(
            (rec := self._read(env, step)) is not None and rec["status"] in statuses for env in env_names
        )

    def on_failed(self, env: str, step: int, *, reason: str) -> None:
        self._write({"env": env, "step": step, "status": FAILED, "reason": reason})

    def _read(self, env: str, step: int) -> dict | None:
        try:
            rec = json.loads(record_path(self.ckpt_dir, env, step).read_text())
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            return {"env": env, "step": step, "status": RUNNING}
        return (
            rec
            if isinstance(rec, dict) and rec.get("status") in STATUSES
            else {"env": env, "step": step, "status": RUNNING}
        )

    def reconcile(
        self,
        *,
        env_names: Iterable[str],
        resume_step: int | None,
        complete_steps: Iterable[int],
        retrigger_on_resume: bool,
        external_steps: Iterable[int] = (),
    ) -> list[str]:
        """Align the records with the timeline this attempt resumes (from ``resume_step``, or from scratch).
        Returns log lines.

        - A record above the resume step belongs to a lost timeline: deleted (readable or not).
        - A non-terminal record below it can never finish (an online eval scores only the live policy): lost.
          So does one at the resume step when the eval is not re-triggered on resume.
        - A complete checkpoint step at or below the resume step without any record file gets ``lost``, except
          the resume step itself when its eval is re-triggered (the trigger writes its record). A record file
          that exists but cannot be read now (a transient FS error) is left alone: overwriting a ``done``
          record would cost that step its role, and the trainer treats an unreadable record as pending.
        """
        resume = resume_step or 0
        complete = set(complete_steps)
        external = set(external_steps)
        lines: list[str] = []
        for env in env_names:
            for step in sorted(s for s in record_steps(self.ckpt_dir, env) if s > resume):
                with contextlib.suppress(FileNotFoundError):
                    record_path(self.ckpt_dir, env, step).unlink()
                lines.append(f"dropped {env} eval record of step {step} (above resume step {resume}: lost timeline)")
            records = read_records(self.ckpt_dir, env, self.log)
            for step, rec in sorted(records.items()):
                if step > resume or rec["status"] in TERMINAL:
                    continue
                if step == resume and retrigger_on_resume:
                    continue
                self._write({**rec, "status": LOST, "reason": f"interrupted; not re-run after resuming at {resume}"})
                lines.append(f"marked {env} eval of step {step} lost (was {rec['status']})")
                if step in external:
                    self.log.warning(
                        f"The eval of the forked step {step} ({env}) was interrupted and cannot be re-run (its "
                        "weights are no longer served): the forked step cannot hold a best role in this run"
                    )
            for step in sorted(s for s in complete if s <= resume and s not in records):
                if step == resume and retrigger_on_resume:
                    continue
                if record_path(self.ckpt_dir, env, step).exists():
                    self.log.warning(f"Eval record of {env} step {step} exists but is unreadable; leaving it alone")
                    continue
                self._write({"env": env, "step": step, "status": LOST, "reason": "no eval record for this checkpoint"})
                lines.append(f"marked {env} eval of step {step} lost (checkpoint without a record)")
        return lines


# ── selection ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetentionConfig:
    metric_keys: Mapping[str, str]
    """``{metric name: full metric key}``; each name holds role ``best_<name>``. Higher is better."""

    max_eval_error: float = 0.05


@dataclass
class Selection:
    keep: set[int]
    roles: dict[str, int | None]
    """``last`` and ``best_<metric>`` -> step (None while there is no candidate)."""

    reasons: dict[int, list[str]]
    """Why each kept step is kept."""

    last: int | None = None
    pending: list[int] = field(default_factory=list)
    eligible: dict[int, dict[str, float]] = field(default_factory=dict)
    """Steps that can be best -> their metric values."""

    ineligible: dict[int, str] = field(default_factory=dict)
    """Steps with a record that cannot be best -> why."""

    def step_roles(self) -> dict[int, list[str]]:
        """``{step: [role, ...]}`` with roles in a stable order (best_* by metric order, then last)."""
        out: dict[int, list[str]] = {}
        for role, step in self.roles.items():
            if step is not None and role != "last":
                out.setdefault(step, []).append(role)
        if self.last is not None:
            out.setdefault(self.last, []).append("last")
        return out


def select_keep(
    complete_local_steps: Iterable[int],
    external_steps: Iterable[int],
    records: Mapping[int, Mapping],
    cfg: RetentionConfig,
    resume_point: int | None = None,
) -> Selection:
    """Keep = {last} U {best per metric} U {complete local steps whose eval is pending} U {resume_point}.

    - last = the newest complete local step (the resume point, unless ``complete_local_steps`` counts a step whose
      orchestrator checkpoint is not written yet; ``resume_point`` is then the newest step a restart can resume
      from, and it is kept whatever its roles).
    - A step can be best when its record is ``done``, its error rate is known and <= ``max_eval_error``, and
      it is a complete local step or an external (forked) step. best_<m> = argmax of metric m; ties go to
      the later step.
    - Pending = a complete local step whose record is missing or not terminal: it cannot be judged yet.
    External steps are never in ``keep`` (they are not this run's to delete), but they can hold a role.
    """
    complete = set(complete_local_steps)
    external = set(external_steps) - complete
    last = max(complete) if complete else None
    reasons: dict[int, list[str]] = {}
    keep: set[int] = set()
    if last is not None:
        keep.add(last)
        reasons.setdefault(last, []).append("last (newest complete checkpoint, the resume point)")

    pending = []
    for step in sorted(complete):
        rec = records.get(step)
        if rec is None or rec.get("status") not in TERMINAL:
            pending.append(step)
            keep.add(step)
            state = "no record yet" if rec is None else rec.get("status")
            reasons.setdefault(step, []).append(f"eval pending ({state})")

    eligible: dict[int, dict[str, float]] = {}
    ineligible: dict[int, str] = {}
    for step, rec in sorted(records.items()):
        if step not in complete and step not in external:
            continue
        status = rec.get("status")
        if status != DONE:
            if status in TERMINAL:
                ineligible[step] = f"eval {status}" + (f": {rec['reason']}" if rec.get("reason") else "")
            continue
        err = _finite(rec.get("eval_error"))
        if err is None:
            ineligible[step] = "eval error rate unknown"
            continue
        if err > cfg.max_eval_error:
            ineligible[step] = f"eval error rate {err:.3f} > {cfg.max_eval_error}"
            continue
        metrics = rec.get("metrics") or {}
        values = {name: v for name, key in cfg.metric_keys.items() if (v := _finite(metrics.get(key))) is not None}
        if not values:
            ineligible[step] = "none of the keep metrics in the record"
            continue
        eligible[step] = values

    roles: dict[str, int | None] = {}
    for name in cfg.metric_keys:
        candidates = [(vals[name], step) for step, vals in eligible.items() if name in vals]
        best = max(candidates)[1] if candidates else None
        roles[f"best_{name}"] = best
        if best is None:
            continue
        where = "external, not deleted by this run" if best in external else "local"
        reasons.setdefault(best, []).append(f"best_{name} = {eligible[best][name]:.4f} ({where})")
        if best in complete:
            keep.add(best)
    roles["last"] = last
    if resume_point is not None and resume_point not in keep:
        keep.add(resume_point)
        reasons.setdefault(resume_point, []).append(
            "resume point (the newest step with both completion markers; the last step's orchestrator "
            "checkpoint is not written yet)"
        )

    return Selection(
        keep=keep,
        roles=roles,
        reasons={s: r for s, r in sorted(reasons.items())},
        last=last,
        pending=pending,
        eligible=eligible,
        ineligible=ineligible,
    )


def fork_steps(resume_dir: Path | None, ckpt_dir: Path) -> tuple[set[int], set[int]]:
    """``(external, protected)`` steps of a ``[resume] dir`` fork. A step dir in another run is external (a
    best candidate this run never deletes); one inside this run's own checkpoint dir is protected instead."""
    if resume_dir is None:
        return set(), set()
    m = _STEP_DIR_RE.match(resume_dir.name)
    if not m:
        return set(), set()
    step = int(m.group(1))
    try:
        inside = resume_dir.resolve().parent == ckpt_dir.resolve()
    except OSError:
        inside = False
    return (set(), {step}) if inside else ({step}, set())


def plan_deletions(
    present_steps: Iterable[int], keep: Iterable[int], newest_saved: int | None, protected: Iterable[int] = ()
) -> list[int]:
    """Step dirs (complete or partial) strictly below the trainer's newest saved step that are not kept.
    Nothing is deleted before this attempt has saved (or resumed from) a step of its own."""
    if newest_saved is None:
        return []
    keep, protected = set(keep), set(protected)
    return sorted(s for s in present_steps if s < newest_saved and s not in keep and s not in protected)


# ── the trainer master's deleter ───────────────────────────────────────────


@dataclass
class RetentionPass:
    selection: Selection
    deleted: list[int]
    failed: dict[int, str]
    newest_saved: int | None


class Keep3Retention:
    """The trainer master's keep-3 deleter. ``run`` is cheap (a directory listing, a few stats and small
    JSON reads), so it is called on every step; it writes ``<ckpt_dir>/retention.json`` (informational)
    whenever the outcome changes."""

    def __init__(
        self,
        ckpt_dir: Path,
        *,
        cfg: RetentionConfig,
        eval_env: str,
        external_steps: Iterable[int] = (),
        protected_steps: Iterable[int] = (),
        log=None,
        rmtree: Callable[[Path], None] = shutil.rmtree,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.ckpt_dir = ckpt_dir
        self.cfg = cfg
        self.eval_env = eval_env
        self.external_steps = set(external_steps)
        self.protected_steps = set(protected_steps)
        self.log = log or _default_logger
        self.rmtree = rmtree
        self.now = now
        self._last_summary: dict | None = None
        self._warned_no_writer = False

    def select(
        self, assume_complete: Iterable[int] = (), newest_saved: int | None = None
    ) -> tuple[CkptScan, Selection]:
        """The current keep set. ``assume_complete`` adds steps whose trainer checkpoint is complete but whose
        orchestrator marker is not written yet (the final step before the orchestrator's teardown save).
        Steps above ``newest_saved`` are not this attempt's checkpoints yet (a lost timeline's leftovers)."""
        scan = scan_ckpt_dir(self.ckpt_dir)
        on_disk = set(scan.complete)
        complete = on_disk | (set(assume_complete) & set(scan.trainer_complete))
        if newest_saved is not None:
            on_disk = {s for s in on_disk if s <= newest_saved}
            complete = {s for s in complete if s <= newest_saved}
        records = read_records(self.ckpt_dir, self.eval_env, self.log)
        # What a restart would resume from (resolve_resume needs both markers) survives even when an assumed
        # step took its last role: deleting it would rewind a restart, or crash-loop it when no older step is left.
        resume_point = max(on_disk) if on_disk else None
        return scan, select_keep(complete, self.external_steps, records, self.cfg, resume_point=resume_point)

    def run_safely(self, newest_saved: int | None, assume_complete: Iterable[int] = ()) -> RetentionPass | None:
        """``run`` for the training loop: retention is housekeeping, so an unexpected error is logged and the
        pass is retried on the next step instead of crashing the trainer (a crash restarts the whole run)."""
        try:
            return self.run(newest_saved, assume_complete)
        except Exception as e:
            self.log.warning(f"Keep-3: retention pass failed ({type(e).__name__}: {e}); retrying next step")
            return None

    def run(self, newest_saved: int | None, assume_complete: Iterable[int] = ()) -> RetentionPass:
        if not self._warned_no_writer and not (eval_results_dir(self.ckpt_dir) / WRITER_FILE).exists():
            self._warned_no_writer = True
            self.log.warning(
                f"Keep-3: no {EVAL_RESULTS_DIR}/{WRITER_FILE} yet; until the orchestrator writes eval records "
                "(GB_KEEP3=1 in [orchestrator.env_vars]), every checkpoint counts as pending and nothing is deleted"
            )
        scan, sel = self.select(assume_complete, newest_saved)
        to_delete = plan_deletions(scan.present, sel.keep, newest_saved, self.external_steps | self.protected_steps)
        deleted, failed = [], {}
        for step in to_delete:
            path = self.ckpt_dir / f"step_{step}"
            try:
                self._remove(path)
            except FileNotFoundError:
                pass
            except OSError as e:
                failed[step] = str(e)
                self.log.warning(f"Keep-3: failed to remove {path}: {e}; retrying next step")
                continue
            deleted.append(step)
        if deleted:
            self.log.info(f"Keep-3: removed checkpoint step(s) {deleted}; keeping {sorted(sel.keep)} {sel.roles}")
        self._maybe_write(sel, deleted, failed, newest_saved)
        return RetentionPass(sel, deleted, failed, newest_saved)

    def _remove(self, path: Path) -> None:
        """Remove a step dir, its completion markers first: a removal killed halfway leaves a partial step that
        no restart resumes from (and the next pass removes), never one that looks complete with shards missing."""
        if path.is_symlink():
            raise OSError(f"{path} is a symlink; not following it")
        for component, marker in CKPT_COMPLETE_MARKERS.items():
            with contextlib.suppress(FileNotFoundError):
                (path / component / marker).unlink()
        self.rmtree(path)

    def _maybe_write(
        self, sel: Selection, deleted: list[int], failed: dict[int, str], newest_saved: int | None
    ) -> None:
        summary = {
            "keep": sorted(sel.keep),
            "roles": sel.roles,
            "reasons": {str(s): r for s, r in sel.reasons.items()},
            "pending": sel.pending,
            "eligible": {str(s): v for s, v in sorted(sel.eligible.items())},
            "ineligible": {str(s): r for s, r in sorted(sel.ineligible.items())},
            "external_steps": sorted(self.external_steps),
            "newest_saved": newest_saved,
            "eval_env": self.eval_env,
            "metric_keys": dict(self.cfg.metric_keys),
            "max_eval_error": self.cfg.max_eval_error,
        }
        if summary == self._last_summary and not deleted and not failed:
            return
        self._last_summary = summary
        try:
            atomic_write_json(
                self.ckpt_dir / RETENTION_FILE,
                {**summary, "deleted_now": deleted, "delete_failed": failed, "updated_at": _utc(self.now())},
            )
        except OSError as e:
            self.log.warning(f"Keep-3: failed to write {RETENTION_FILE}: {e}")
