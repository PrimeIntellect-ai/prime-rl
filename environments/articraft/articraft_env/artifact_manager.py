"""Articraft artifact lifecycle manager.

All filesystem paths are resolved here; ``env.py`` never hardcodes paths.
Follows the BlenderGym ArtifactManager pattern but without render/image logic.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.workspace_docs import VirtualWorkspace

    from .schema import Rollout

logger = logging.getLogger(__name__)

MODEL_FILENAME = "model.py"
META_JSON_FILENAME = "meta.json"
TRAJECTORY_JSON_FILENAME = "trajectory.json"
CHECKPOINT_URDF_FILENAME = "checkpoint.urdf"

from .schema import SCHEMA_VERSION


@dataclass
class ArtifactPolicy:
    save_meta_json: bool = True
    save_trajectory_json: bool = True
    keep_failed_only: bool = False
    max_rollouts_per_example: int = 0


class ArticraftArtifactManager:
    """Single entry-point for artifact path resolution, I/O, and retention."""

    def __init__(
        self,
        work_root: Path,
        policy: ArtifactPolicy,
        *,
        articraft_root: Path,
        sdk_package: str = "sdk",
    ) -> None:
        self.work_root = work_root
        self.policy = policy
        self.articraft_root = articraft_root
        self.sdk_package = sdk_package

    # ---- path resolution (pure) ----

    def rollout_dir(
        self,
        *,
        traj_id: str,
        record_id: str,
        split: str | None = None,
        example_id: int | None = None,
    ) -> Path:
        if split is not None and example_id is not None:
            return (
                self.work_root / split
                / f"example_{example_id:04d}__{record_id[:40]}"
                / traj_id[:12]
            )
        return self.work_root / f"{record_id[:40]}__{traj_id[:12]}"

    def script_path(self, work_dir: Path) -> Path:
        return work_dir / MODEL_FILENAME

    def checkpoint_urdf_path(self, work_dir: Path) -> Path:
        return work_dir / CHECKPOINT_URDF_FILENAME

    # ---- workspace construction ----

    def build_workspace(self, script_path: Path) -> VirtualWorkspace:
        from agent.workspace_docs import build_virtual_workspace

        return build_virtual_workspace(
            self.articraft_root,
            model_file_path=script_path,
            sdk_package=self.sdk_package,
        )

    # ---- setup I/O ----

    def make_rollout_dir(
        self,
        *,
        traj_id: str,
        record_id: str,
        split: str | None = None,
        example_id: int | None = None,
    ) -> Path:
        d = self.rollout_dir(
            traj_id=traj_id,
            record_id=record_id,
            split=split,
            example_id=example_id,
        )
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- per-rollout I/O ----

    def save_trajectory(
        self,
        rollout: Rollout,
        *,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        work_dir = rollout.work_dir

        if self.policy.save_meta_json:
            meta: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "record_id": rollout.task.record_id,
                "trajectory_id": rollout.trajectory_id,
                "category_slug": rollout.task.category_slug,
                "final_reward": rollout.final_reward,
                "turns_used": len(rollout.turns),
                "max_turns": rollout.max_turns,
                "compile_attempted": any(
                    t.compile_attempted for t in rollout.turns
                ),
                "code_is_fresh": rollout.code_is_fresh(),
            }
            if metrics:
                meta["metrics"] = metrics
            if rollout.metadata:
                meta["metadata"] = rollout.metadata
            (work_dir / META_JSON_FILENAME).write_text(
                json.dumps(meta, indent=2, default=str), encoding="utf-8"
            )

        if self.policy.save_trajectory_json:
            traj: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "task": {
                    "record_id": rollout.task.record_id,
                    "prompt_text": rollout.task.prompt_text,
                    "category_slug": rollout.task.category_slug,
                    "sdk_package": rollout.task.sdk_package,
                },
                "trajectory_id": rollout.trajectory_id,
                "final_reward": rollout.final_reward,
                "turns": [
                    {
                        "turn": t.turn,
                        "tool_calls": t.tool_calls,
                        "compile_attempted": t.compile_attempted,
                        "compile_success": t.compile_success,
                        "compile_signals": t.compile_signals,
                    }
                    for t in rollout.turns
                ],
            }
            (work_dir / TRAJECTORY_JSON_FILENAME).write_text(
                json.dumps(traj, indent=2, default=str), encoding="utf-8"
            )

    # ---- retention ----

    def cleanup_rollout(self, rollout: Rollout) -> None:
        work_dir = rollout.work_dir
        if not work_dir.is_dir():
            return
        if self.policy.keep_failed_only and rollout.code_is_fresh():
            shutil.rmtree(work_dir, ignore_errors=True)

    def prune_old_rollouts(self, rollout: Rollout) -> None:
        limit = self.policy.max_rollouts_per_example
        if limit <= 0:
            return
        example_dir = rollout.work_dir.parent
        if not example_dir.is_dir():
            return
        traj_dirs = [d for d in example_dir.iterdir() if d.is_dir()]
        if len(traj_dirs) <= limit:
            return
        traj_dirs.sort(key=lambda d: d.stat().st_mtime)
        for td in traj_dirs[: len(traj_dirs) - limit]:
            logger.debug("prune_old_rollouts: removing %s", td)
            shutil.rmtree(td, ignore_errors=True)
