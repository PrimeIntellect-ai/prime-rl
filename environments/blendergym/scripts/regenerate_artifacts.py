"""Regenerate ``trajectory.html`` for an existing BlenderGym work root.

Walks every ``<work-root>/<traj>/`` directory, rebuilds a synthetic
``Rollout`` from the persisted ``meta.json`` + ``trajectory.json``, and
re-emits the three artifacts via :func:`write_trajectory_artifacts`. Use
this after upgrading the writer (e.g. swapping markdown for HTML) to
backfill old rollout dirs without re-running training.

Example::

    uv run python environments/blendergym/scripts/regenerate_artifacts.py \\
        --work-root outputs/blendergym_v2/blendergym_work --limit 5
    uv run python environments/blendergym/scripts/regenerate_artifacts.py \\
        --work-root outputs/blendergym_v2/blendergym_work
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from blendergym.schema import Rollout, Task, TurnRecord
from blendergym.trajectory_writer import write_trajectory_artifacts

logger = logging.getLogger("blendergym.regenerate_artifacts")


def _build_task(traj: dict) -> Task:
    """Reconstruct a :class:`Task` from a persisted ``trajectory.json``.

    Schema v2 only persists ``task_id / task_type / blend_file`` — the
    other ``Task`` fields (goal/init image paths, start_code path) are not
    needed by :func:`write_trajectory_artifacts` (it reads images by
    relative path under ``work_dir``), so we use placeholders.
    """
    task = traj.get("task", {})
    blend_file = Path(task.get("blend_file", ""))
    return Task(
        task_id=task.get("task_id", "unknown"),
        task_type=task.get("task_type", "unknown"),
        blend_file=blend_file,
        goal_image=Path("inputs/goal.png"),
        init_image=Path("inputs/init.png"),
        start_code_path=Path("inputs/start.py"),
    )


def _build_rollout(work_dir: Path, traj: dict) -> Rollout:
    turns: list[TurnRecord] = []
    for step in traj.get("steps", []):
        record = TurnRecord(
            turn=step.get("turn", 0),
            exit_status=step.get("exit_status"),
            error_hint=step.get("error_hint"),
            action=step.get("action"),
            render_path=step.get("render_path"),
            code_path=step.get("code_path"),
            response_path=step.get("response_path") or "",
            log_path=step.get("log_path"),
            duration_s=step.get("duration_s"),
        )
        turns.append(record)

    return Rollout(
        task=_build_task(traj),
        trajectory_id=traj.get("trajectory_id", work_dir.name),
        work_dir=work_dir,
        gpu_id=int(traj.get("runtime", {}).get("gpu_id", 0) or 0),
        max_turns=int(traj.get("max_turns", len(turns))),
        turns=turns,
        final_reward=traj.get("final_reward"),
    )


def _iter_traj_dirs(work_root: Path) -> list[Path]:
    """Return sorted child directories that contain a ``meta.json``."""
    if not work_root.is_dir():
        raise FileNotFoundError(f"work-root does not exist: {work_root}")
    candidates = [d for d in work_root.iterdir() if d.is_dir()]
    return sorted(d for d in candidates if (d / "meta.json").is_file())


def _process_one(work_dir: Path, *, dry_run: bool, overwrite: bool) -> str:
    """Regenerate one trajectory dir; return a short status string."""
    traj_path = work_dir / "trajectory.json"
    if not traj_path.is_file():
        return "skipped (no trajectory.json)"

    html_path = work_dir / "trajectory.html"
    if html_path.exists() and not overwrite:
        return "skipped (html exists, no --overwrite)"

    if dry_run:
        return "dry-run"

    try:
        traj = json.loads(traj_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"failed (bad json: {exc})"

    rollout = _build_rollout(work_dir, traj)
    write_trajectory_artifacts(rollout, metrics=traj.get("metrics") or {})
    return "ok"


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python environments/blendergym/scripts/regenerate_artifacts.py",
        description=(
            "Re-emit meta.json / trajectory.json / trajectory.html for every "
            "rollout under a BlenderGym work root, using the current writer."
        ),
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        required=True,
        help="Path to <output>/blendergym_work containing per-trajectory dirs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N dirs (sorted by name). Useful for spot checks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the dirs that would be processed without writing anything.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help=(
            "Overwrite existing trajectory.html (default: True). "
            "Pass --no-overwrite to skip dirs already converted."
        ),
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Skip dirs that already have trajectory.html.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only log totals, not per-dir status lines.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )

    dirs = _iter_traj_dirs(args.work_root)
    if args.limit is not None:
        dirs = dirs[: args.limit]
    logger.info("scanning %d trajectory dirs under %s", len(dirs), args.work_root)

    counts: dict[str, int] = {}
    for d in dirs:
        status = _process_one(d, dry_run=args.dry_run, overwrite=args.overwrite)
        counts[status.split(" (", 1)[0]] = counts.get(status.split(" (", 1)[0], 0) + 1
        if not args.quiet:
            logger.info("[%s] %s", status, d.name)

    logger.warning("done: %s", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0 if counts.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
