"""BlenderGym artifact lifecycle manager."""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .render import RenderResult
    from .schema import Rollout, TurnRecord

logger = logging.getLogger(__name__)


# ---- filename constants (真正的常量，UPPER_CASE) ----------------------------

CODE_FILENAME = "code.py"
RENDER_FILENAME = "render1.png"
LOG_FILENAME = "blender.log"
RESPONSE_FILENAME = "response.txt"
META_JSON_FILENAME = "meta.json"
TRAJECTORY_JSON_FILENAME = "trajectory.json"
TRAJECTORY_HTML_FILENAME = "trajectory.html"
TURN_DIR_FMT = "turn_{}"
INPUTS_DIRNAME = "inputs"
BLENDER_USER_DIRNAME = "blender_user"

INPUT_SYMLINKS: tuple[tuple[str, str], ...] = (
    ("goal_image", "goal.png"),
    ("init_image", "init.png"),
    ("start_code_path", "start.py"),
)


# ---- policy ----------------------------------------------------------------


@dataclass
class ArtifactPolicy:
    """Declarative knobs for artifact management.

    All flags default to "keep everything" so existing configs
    that omit new keys behave identically.
    """
    save_blender_log: bool = True
    save_response_txt: bool = True
    save_meta_json: bool = True
    save_trajectory_html: bool = True
    keep_failed_only: bool = False
    remove_intermediate_turns: bool = False
    max_rollouts_per_example: int = 0  # 0 = unlimited


# ---- path bundles ----------------------------------------------------------


@dataclass(frozen=True)
class TurnPaths:
    """Immutable bundle of resolved per-turn artifact paths."""
    turn_dir: Path
    code: Path
    render: Path
    log: Path | None       # None = skip writing blender.log
    response: Path
    blender_user: Path


@dataclass(frozen=True)
class RolloutPaths:
    """Immutable bundle of resolved per-rollout artifact paths."""
    meta_json: Path | None
    trajectory_json: Path
    trajectory_html: Path | None


# ---- manager ---------------------------------------------------------------


class ArtifactManager:
    """Single entry-point for artifact path resolution, I/O, and retention.

    Instantiated once per env worker in BlenderGymEnv.__init__.
    """

    def __init__(self, work_root: Path, policy: ArtifactPolicy) -> None:
        self.work_root = work_root
        self.policy = policy

    # ---- path resolution (pure, no side effects) ----------------------------

    def rollout_dir(
        self, *, traj_id: str, task_id: str,
        split: str | None = None, example_id: object | None = None,
    ) -> Path:
        if split is not None and isinstance(example_id, int):
            return (
                self.work_root / split
                / f"example_{example_id:04d}__{task_id}"
                / traj_id[:8]
            )
        return self.work_root / f"{task_id}__{traj_id[:8]}"

    def turn_dir(self, work_dir: Path, turn: int) -> Path:
        return work_dir / TURN_DIR_FMT.format(turn)

    def blender_user_dir(self, rollout_work_dir: Path) -> Path:
        """Per-rollout shared blender_user directory."""
        return rollout_work_dir / BLENDER_USER_DIRNAME

    def turn_paths(self, rollout_work_dir: Path, turn_dir: Path) -> TurnPaths:
        p = self.policy
        return TurnPaths(
            turn_dir=turn_dir,
            code=turn_dir / CODE_FILENAME,
            render=turn_dir / RENDER_FILENAME,
            log=turn_dir / LOG_FILENAME if p.save_blender_log else None,
            response=turn_dir / RESPONSE_FILENAME,
            blender_user=self.blender_user_dir(rollout_work_dir),
        )

    def rollout_paths(self, work_dir: Path) -> RolloutPaths:
        p = self.policy
        return RolloutPaths(
            meta_json=work_dir / META_JSON_FILENAME if p.save_meta_json else None,
            trajectory_json=work_dir / TRAJECTORY_JSON_FILENAME,
            trajectory_html=work_dir / TRAJECTORY_HTML_FILENAME if p.save_trajectory_html else None,
        )

    def rel_code(self, turn: int) -> str:
        return f"{TURN_DIR_FMT.format(turn)}/{CODE_FILENAME}"

    def rel_render(self, turn: int) -> str:
        return f"{TURN_DIR_FMT.format(turn)}/{RENDER_FILENAME}"

    def rel_log(self, turn: int) -> str:
        return f"{TURN_DIR_FMT.format(turn)}/{LOG_FILENAME}"

    def rel_response(self, turn: int) -> str:
        return f"{TURN_DIR_FMT.format(turn)}/{RESPONSE_FILENAME}"

    def last_render_path(self, rollout: "Rollout") -> Path | None:
        t = rollout.last_turn
        if t is None or not t.render_success:
            return None
        p = rollout.work_dir / self.rel_render(t.turn)
        return p if p.is_file() else None

    # ---- setup I/O ----------------------------------------------------------

    def make_rollout_dir(
        self, *, traj_id: str, task_id: str,
        split: str | None = None, example_id: object | None = None,
    ) -> Path:
        d = self.rollout_dir(
            traj_id=traj_id, task_id=task_id,
            split=split, example_id=example_id,
        )
        d.mkdir(parents=True, exist_ok=True)
        return d

    def populate_input_symlinks(self, rollout: "Rollout") -> None:
        inputs_dir = rollout.work_dir / INPUTS_DIRNAME
        inputs_dir.mkdir(exist_ok=True)
        for attr, link_name in INPUT_SYMLINKS:
            src = getattr(rollout.task, attr)
            link = inputs_dir / link_name
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(os.path.abspath(src), link)

    # ---- per-turn I/O -------------------------------------------------------

    def begin_turn(self, rollout_work_dir: Path, turn: int) -> TurnPaths:
        td = self.turn_dir(rollout_work_dir, turn)
        td.mkdir(parents=True, exist_ok=True)
        paths = self.turn_paths(rollout_work_dir, td)
        paths.blender_user.mkdir(parents=True, exist_ok=True)
        return paths

    def write_response(self, paths: TurnPaths, text: str) -> None:
        if self.policy.save_response_txt:
            paths.response.write_text(text, encoding="utf-8")

    # ---- record construction ------------------------------------------------

    def init_record(self, turn: int) -> "TurnRecord":
        from .schema import TurnRecord
        return TurnRecord.for_turn(
            turn, response_path=self.rel_response(turn),
        )

    def fill_record(self, record: "TurnRecord", result: "RenderResult") -> None:
        """Fill both execution-state and path fields on record."""
        t = record.turn
        record.fill_from_render(
            result,
            rel_code=self.rel_code(t),
            rel_log=self.rel_log(t) if self.policy.save_blender_log else None,
            rel_render=self.rel_render(t),
        )

    # ---- per-rollout I/O ----------------------------------------------------

    def save_trajectory(
        self, rollout: "Rollout", *, metrics: dict | None = None,
    ) -> None:
        from .trajectory_writer import write_trajectory_artifacts
        paths = self.rollout_paths(rollout.work_dir)
        write_trajectory_artifacts(rollout, metrics=metrics, paths=paths)

    # ---- retention (best-effort, no locking) --------------------------------

    def cleanup_rollout(self, rollout: "Rollout") -> None:
        work_dir = rollout.work_dir
        if not work_dir.is_dir():
            return
        if self.policy.keep_failed_only and rollout.xml_parsed and rollout.render_success:
            shutil.rmtree(work_dir, ignore_errors=True)
            return
        if self.policy.remove_intermediate_turns:
            turn_dirs = sorted(work_dir.glob(TURN_DIR_FMT.format("*")), key=lambda d: d.name)
            if len(turn_dirs) > 1:
                for td in turn_dirs[:-1]:
                    shutil.rmtree(td, ignore_errors=True)

    def prune_old_rollouts(self, rollout: "Rollout") -> None:
        """Best-effort pruning. No locking — concurrent workers may
        overshoot or undershoot the retention limit by a small margin.
        """
        limit = self.policy.max_rollouts_per_example
        if limit <= 0:
            return
        example_dir = rollout.work_dir.parent
        if not example_dir.is_dir():
            return
        traj_dirs = [
            d for d in example_dir.iterdir()
            if d.is_dir() and d.name != INPUTS_DIRNAME
        ]
        if len(traj_dirs) <= limit:
            return
        traj_dirs.sort(key=lambda d: d.stat().st_mtime)
        for td in traj_dirs[: len(traj_dirs) - limit]:
            logger.debug("prune_old_rollouts: removing %s", td)
            shutil.rmtree(td, ignore_errors=True)
