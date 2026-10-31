"""GeneBench round-2 end-of-run export (GB_FINAL_EXPORT, see utils/gb_settings.py): the torch-free part.

After its final checkpoint, the RL trainer waits for the final held-out eval and for every eval still pending for a
kept checkpoint. It then settles the keep-3 set (utils/ckpt_retention.py) and exports the role holders (the best
checkpoint per keep metric, and the last one) to private Hugging Face repos, inside the same run.
trainer/rl/final_export.py drives trainer/sft/hf_export.py for the export itself. The files, under <output_dir>:

- ``final_checkpoints.json``: the final roles, steps, metrics, eligibility and repo ids, rewritten with each
  upload's commit.
- ``hf-export/plan.json``: what to export. It is fixed once computed, so a restarted trainer exports the same steps
  with the same roles.
- ``hf-export/state.json``: the verified uploads (a restarted trainer skips them) and the number of export
  attempts of this timeline.
- ``hf-export/step_N/``: the staged repo. Its weight shards are removed once the repo is verified; the README,
  config, tokenizer and export report stay.
- ``hf-export/DONE`` or ``hf-export/FAILED``: the outcome (JSON). The orchestrator waits for one of them before
  it marks the run finished, so the platform does not tear the run down mid-upload.
- ``hf-export/armed.json``: written at the trainer's startup; tells the orchestrator that an outcome will come.

A step forked from another run (``[resume] dir``) is external: it can hold a role, but it is never uploaded again
(its repo already exists); the plan only records it.

This module only uses the standard library, so it can be tested without the GPU stack.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import shutil
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from prime_rl.trainer.sft.rl_export_plan import HELDOUT_KEYS, expected_repo_id, is_valid_role
from prime_rl.utils.ckpt_retention import Selection, atomic_write_json, eval_error

EXPORT_DIR = "hf-export"
PLAN_FILE = "plan.json"
STATE_FILE = "state.json"
DONE_FILE = "DONE"
FAILED_FILE = "FAILED"
ARMED_FILE = "armed.json"
FINAL_CHECKPOINTS_FILE = "final_checkpoints.json"
REPORT_FILE = "export_report.json"

LIVE = "live"
DCP = "dcp"

EXPORT = "export"
SKIP = "skip"
REFRESH = "refresh"

_STEP_DIR_RE = re.compile(r"^step_(\d+)$")
_default_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExportPaths:
    output_dir: Path

    @property
    def root(self) -> Path:
        return self.output_dir / EXPORT_DIR

    @property
    def plan(self) -> Path:
        return self.root / PLAN_FILE

    @property
    def state(self) -> Path:
        return self.root / STATE_FILE

    @property
    def done(self) -> Path:
        return self.root / DONE_FILE

    @property
    def failed(self) -> Path:
        return self.root / FAILED_FILE

    @property
    def armed(self) -> Path:
        return self.root / ARMED_FILE

    @property
    def report(self) -> Path:
        return self.root / REPORT_FILE

    @property
    def final_checkpoints(self) -> Path:
        return self.output_dir / FINAL_CHECKPOINTS_FILE

    def staging(self, step: int) -> Path:
        return self.root / f"step_{step}"


def _utc(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def read_json(path: Path) -> dict | None:
    """The JSON object in ``path``, or None when it is missing or unreadable."""
    try:
        obj = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


# ── the eval wait ──────────────────────────────────────────────────────────


def wait_for(
    done: Callable[[], bool],
    *,
    timeout_s: float,
    poll_s: float,
    on_wait: Callable[[float], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> bool:
    """Poll ``done()`` until it is true (returns True) or ``timeout_s`` passes (returns False). ``on_wait`` gets
    the seconds waited so far, once per poll."""
    start = now()
    while True:
        if done():
            return True
        waited = now() - start
        if waited >= timeout_s:
            return False
        if on_wait is not None:
            on_wait(waited)
        sleep(min(poll_s, max(timeout_s - waited, 0.0)))


def heldout_from_record(rec: Mapping | None, env: str) -> dict[str, float | None] | None:
    """The model card's held-out eval values (rl_export_plan.HELDOUT_KEYS names) from an eval record, or None
    when the record is not ``done``."""
    if not rec or rec.get("status") != "done":
        return None
    metrics = rec.get("metrics") or {}
    prefix = f"eval/{env}/all/agent/"
    return {name: metrics.get(prefix + key) for name, key in HELDOUT_KEYS.items()}


# ── the plan ───────────────────────────────────────────────────────────────


def build_plan(
    sel: Selection,
    *,
    final_step: int,
    external_steps: Iterable[int],
    records: Mapping[int, Mapping],
    eval_env: str,
    prefix: str,
    run_id: str,
    ckpt_dir: Path,
    timed_out: bool,
    pending: Iterable[int] = (),
    now: Callable[[], float] = time.time,
) -> dict:
    """What to export, from the final keep-3 selection (computed with the final step assumed complete).

    Every role holder is listed with its combined role (e.g. ``best_train_reward+last``) and repo
    ``<prefix>-step<N>``. A local one is exported, the final step from the live model (``source: live``) and the
    others from their DCP checkpoints; the live one goes first, since loading another step replaces the live
    weights. An external (forked) one is only recorded. Kept steps without a role (evals still pending when the
    wait timed out) are listed as kept but not exported."""
    external = set(external_steps)
    step_roles = sel.step_roles()
    problems = []
    if sel.last != final_step:
        problems.append(f"the last checkpoint is step {sel.last}, not the final step {final_step}")
    exports, external_roles = [], []
    for step, roles in sorted(step_roles.items()):
        role = "+".join(roles)
        if not is_valid_role(role):
            problems.append(f"step {step}: invalid role {role!r}")
            continue
        rec = records.get(step)
        item = {
            "step": step,
            "roles": roles,
            "role": role,
            "repo": expected_repo_id(step, prefix),
            "eval": dict(rec) if rec else None,
            "heldout": heldout_from_record(rec, eval_env),
        }
        if step in external:
            external_roles.append(dict(item, note="forked from another run: its repo is not uploaded again"))
            continue
        item["source"] = LIVE if step == final_step else DCP
        item["dcp_dir"] = str(ckpt_dir / f"step_{step}" / "trainer")
        exports.append(item)
    exports.sort(key=lambda item: (item["source"] != LIVE, -item["step"]))
    exported = {item["step"] for item in exports}
    return {
        "final_step": final_step,
        "run_id": run_id,
        "repo_prefix": prefix,
        "eval_env": eval_env,
        "roles": sel.roles,
        "exports": exports,
        "external": external_roles,
        "kept_not_exported": {
            str(step): sel.reasons.get(step, []) for step in sorted(sel.keep - exported) if step not in external
        },
        "pending_evals": sorted(set(pending)),
        "eval_wait_timed_out": timed_out,
        "eligible": {str(s): v for s, v in sorted(sel.eligible.items())},
        "ineligible": {str(s): r for s, r in sorted(sel.ineligible.items())},
        "problems": problems,
        "created_at": _utc(now()),
    }


def plan_matches(plan: Mapping | None, final_step: int) -> bool:
    """A plan written by an earlier attempt for the same final step (reused as is, so that a restart exports the
    same steps under the same roles)."""
    return bool(plan) and plan.get("final_step") == final_step and isinstance(plan.get("exports"), list)


def plan_steps(plan: Mapping) -> set[int]:
    return {item["step"] for item in plan.get("exports", [])}


# ── upload state ───────────────────────────────────────────────────────────


def load_state(path: Path) -> dict:
    state = read_json(path) or {}
    uploads = state.get("uploads")
    state["uploads"] = uploads if isinstance(uploads, dict) else {}
    return state


def record_upload(state: dict, item: Mapping, commit: str, now: Callable[[], float] = time.time) -> dict:
    state["uploads"][str(item["step"])] = {
        "repo": item["repo"],
        "role": item["role"],
        "commit": commit,
        "verified_at": _utc(now()),
    }
    return state


def begin_export_attempt(state: dict, actions: list[tuple[dict, str]], max_attempts: int) -> str | None:
    """Count an attempt that has something to upload in ``state["export_attempts"]``. Returns why to give up when
    ``max_attempts`` attempts already started without finishing (each one crashed the trainer, e.g. a rank stuck in
    a collective until the NCCL watchdog fired; an export error that does not crash ends with FAILED), else None.
    Without a cap, every restart would resume at max_steps and crash in the same place again."""
    if all(action == SKIP for _, action in actions):
        return None
    attempts = int(state.get("export_attempts") or 0)
    if attempts >= max_attempts:
        return (
            f"{attempts} earlier export attempt(s) of this timeline ended without an outcome (the trainer died "
            f"mid-export); not trying again. Remove export_attempts from {STATE_FILE} to allow another attempt"
        )
    state["export_attempts"] = attempts + 1
    return None


def token_write_problem(who: Mapping, namespace: str) -> tuple[str | None, str | None]:
    """``(problem, warning)`` for an HF token (``HfApi().whoami()``) that should create repos under ``namespace``:
    a problem when it cannot (not in the namespace, a read token, a read-only org member), a warning when its
    scopes cannot be checked here (a fine-grained token without a visible repo.write scope)."""
    orgs = {org.get("name"): org for org in who.get("orgs") or [] if isinstance(org, Mapping)}
    names = {who.get("name"), *orgs}
    if namespace not in names:
        return (
            f"the HF token of {who.get('name')!r} cannot write to {namespace!r} (its namespaces: "
            f"{sorted(n for n in names if n)})",
            None,
        )
    token = (who.get("auth") or {}).get("accessToken") or {}
    role = token.get("role")
    if role == "read":
        return f"the HF token of {who.get('name')!r} is a read token; the export needs write access", None
    org_role = (orgs.get(namespace) or {}).get("roleInOrg")
    if org_role == "read":
        return f"{who.get('name')!r} has only read access to {namespace!r}", None
    if role == "fineGrained":
        fine = token.get("fineGrained") or {}
        perms = set(fine.get("global") or [])
        for scope in fine.get("scoped") or []:
            if ((scope.get("entity") or {}).get("name")) == namespace:
                perms |= set(scope.get("permissions") or [])
        if "repo.write" not in perms:
            return None, (
                f"the HF token is fine-grained and shows no repo.write scope for {namespace!r}; if it cannot create "
                "repos there, the export fails only at the end of training"
            )
    return None, None


def export_actions(plan: Mapping, state: Mapping) -> list[tuple[dict, str]]:
    """``(item, action)`` per plan export, in plan order. A step already verified in its repo is skipped, or gets
    only its README and report refreshed when its role changed (``refresh``); every other step is exported."""
    out = []
    uploads = state.get("uploads", {})
    for item in plan.get("exports", []):
        done = uploads.get(str(item["step"]))
        if done and done.get("repo") == item["repo"] and done.get("commit"):
            out.append((item, SKIP if done.get("role") == item["role"] else REFRESH))
        else:
            out.append((item, EXPORT))
    return out


def invalidate_stale(paths: ExportPaths, resume_step: int | None, max_steps: int, log=None) -> list[str]:
    """Run by the trainer master at startup when this attempt will train (it did not resume at ``max_steps``):
    the plan and the outcome of an earlier timeline no longer hold, and uploads and staged repos of steps after the
    resume point would be retrained. Uploads at or before the resume point stay valid (same weights). Returns
    log lines."""
    log = log or _default_logger
    lines: list[str] = []
    if resume_step is not None and resume_step >= max_steps:
        return lines
    resume = resume_step or 0
    for path in (paths.plan, paths.done, paths.failed, paths.final_checkpoints):
        if path.exists():
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
            lines.append(f"removed {path} (written before this attempt resumed at step {resume})")
    if paths.state.exists():
        state = load_state(paths.state)
        stale = sorted(int(s) for s in state["uploads"] if int(s) > resume)
        for step in stale:
            del state["uploads"][str(step)]
        if stale:
            lines.append(
                f"forgot the uploads of steps {stale} (after the resume step {resume}; they will be retrained)"
            )
        reset = bool(state.pop("export_attempts", None))
        if reset:
            lines.append("reset the export attempt count (this timeline has not exported yet)")
        if stale or reset:
            atomic_write_json(paths.state, state)
    if paths.root.is_dir():
        for child in sorted(paths.root.iterdir()):
            m = _STEP_DIR_RE.match(child.name)
            if m and int(m.group(1)) > resume and child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
                lines.append(f"removed the staged repo {child} (after the resume step {resume})")
    return lines


# ── outcome ────────────────────────────────────────────────────────────────


def write_outcome(paths: ExportPaths, ok: bool, payload: Mapping, now: Callable[[], float] = time.time) -> Path:
    """DONE (removing an older FAILED) or FAILED, as JSON."""
    path, other = (paths.done, paths.failed) if ok else (paths.failed, paths.done)
    atomic_write_json(path, {**payload, "ok": ok, "written_at": _utc(now())})
    with contextlib.suppress(FileNotFoundError):
        other.unlink()
    return path


def clear_outcome(paths: ExportPaths) -> list[str]:
    """Remove DONE / FAILED (an earlier attempt's outcome, before this one exports again). Returns log lines."""
    lines = []
    for path in (paths.done, paths.failed):
        if path.exists():
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
            lines.append(f"removed {path} (an earlier attempt's outcome; this attempt exports again)")
    return lines


def read_outcome(paths: ExportPaths, final_step: int | None = None) -> tuple[str | None, dict | None]:
    """``("done" | "failed", payload)`` when an outcome for ``final_step`` (any, when None) is written, else
    ``(None, None)``."""
    for status, path in (("done", paths.done), ("failed", paths.failed)):
        payload = read_json(path)
        if payload is not None and (final_step is None or payload.get("final_step") == final_step):
            return status, payload
    return None, None


def write_armed(paths: ExportPaths, info: Mapping, now: Callable[[], float] = time.time) -> None:
    atomic_write_json(paths.armed, {**info, "armed_at": _utc(now())})


def final_checkpoints_doc(plan: Mapping, state: Mapping, errors: Mapping[int, str] | None = None) -> dict:
    """``final_checkpoints.json``: the plan's roles and steps, with each upload's commit or error."""
    uploads = state.get("uploads", {})
    errors = errors or {}
    checkpoints = []
    for item in plan.get("exports", []):
        done = uploads.get(str(item["step"])) or {}
        checkpoints.append(
            {
                "step": item["step"],
                "roles": item["roles"],
                "repo": item["repo"],
                "source": item["source"],
                "dcp_dir": item["dcp_dir"],
                "heldout": item["heldout"],
                "eval_error": (item.get("eval") or {}).get("eval_error"),
                "commit": done.get("commit") if done.get("role") == item["role"] else None,
                "error": errors.get(item["step"]),
            }
        )
    for item in plan.get("external", []):
        checkpoints.append(
            {
                "step": item["step"],
                "roles": item["roles"],
                "repo": item["repo"],
                "source": "external",
                "heldout": item["heldout"],
                "eval_error": (item.get("eval") or {}).get("eval_error"),
                "repo_check": item.get("repo_check"),
                "note": item.get("note"),
            }
        )
    return {
        "final_step": plan.get("final_step"),
        "run_id": plan.get("run_id"),
        "roles": plan.get("roles"),
        "checkpoints": sorted(checkpoints, key=lambda c: c["step"]),
        "kept_not_exported": plan.get("kept_not_exported", {}),
        "pending_evals": plan.get("pending_evals", []),
        "eval_wait_timed_out": plan.get("eval_wait_timed_out"),
        "eligible": plan.get("eligible", {}),
        "ineligible": plan.get("ineligible", {}),
        "problems": plan.get("problems", []),
    }


def outcome_payload(plan: Mapping, state: Mapping, errors: Mapping[int, str]) -> dict:
    doc = final_checkpoints_doc(plan, state, errors)
    return {
        "final_step": plan.get("final_step"),
        "repos": {
            str(c["step"]): {"repo": c["repo"], "commit": c.get("commit"), "roles": c["roles"]}
            for c in doc["checkpoints"]
        },
        "errors": {str(step): error for step, error in sorted(errors.items())},
        "problems": plan.get("problems", []),
    }


def run_actions(
    actions: list[tuple[dict, str]],
    *,
    export: Callable[[dict], str],
    refresh: Callable[[dict], str],
    on_verified: Callable[[dict, str], None],
    log=None,
) -> dict[int, str]:
    """Run the plan's steps in order: ``export(item)`` or ``refresh(item)`` returns the repo's verified head
    commit, then ``on_verified(item, commit)`` records it. A step that raises is logged and recorded, and the next
    one still runs (it is exported from its own checkpoint). Returns ``{step: error}``.

    On the trainer, ``export`` and ``refresh`` are collectives that raise on every rank together, so every rank
    takes the same path through this loop."""
    log = log or _default_logger
    errors: dict[int, str] = {}
    for item, action in actions:
        step = item["step"]
        if action == SKIP:
            log.info(f"Final export: step {step} is already verified in {item['repo']}; skipping")
            continue
        try:
            commit = refresh(item) if action == REFRESH else export(item)
        except Exception as e:
            errors[step] = f"{type(e).__name__}: {e}"
            log.error(f"Final export: step {step} ({item['role']}) to {item['repo']} FAILED: {errors[step]}")
            continue
        on_verified(item, commit)
    return errors


def card_eval_desc(rec: Mapping | None, group_size: int | None) -> tuple[str, str]:
    """The model card's eval description and pass@k label, from the record and the eval group size."""
    if not rec or rec.get("status") != "done":
        return "not evaluated", "pass@k"
    err = rec.get("eval_error")
    if err is None:
        err = eval_error(rec.get("metrics") or {}, rec.get("env", ""))
    desc = f"{rec.get('n_episodes')} graded episodes" + (f", {group_size} per task" if group_size else "")
    if err is not None:
        desc += f", error rate {err:.3f}"
    return desc, f"pass@{group_size}" if group_size else "pass@k"
