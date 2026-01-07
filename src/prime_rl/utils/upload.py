"""
Checkpoint upload utility for uploading training checkpoints to HuggingFace Hub.

Usage:
    # Upload all checkpoint types for all common steps (oldest first)
    uv run upload-ckpt /shared/outputs/int4

    # Upload specific type only
    uv run upload-ckpt /shared/outputs/int4 --type weights
    uv run upload-ckpt /shared/outputs/int4 --type trainer
    uv run upload-ckpt /shared/outputs/int4 --type orchestrator

    # Upload specific steps (in given order)
    uv run upload-ckpt /shared/outputs/int4 --steps 100 200 300

    # Combine: specific type + specific steps
    uv run upload-ckpt /shared/outputs/int4 --type weights --steps 450

    # Custom org/prefix
    uv run upload-ckpt /shared/outputs/int4 --org MyOrg --repo-prefix MyModel

    # Public repo (default is private)
    uv run upload-ckpt /shared/outputs/int4 --public

    # Dry run - show what would be uploaded
    uv run upload-ckpt /shared/outputs/int4 --dry-run
"""

import os
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Annotated

UPLOADING_MARKER = ".uploading"

from huggingface_hub import HfApi
from pydantic import Field
from pydantic_settings import BaseSettings, CliApp, CliPositionalArg

from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import (
    get_all_ckpt_steps,
    get_ckpt_dir,
    get_common_ckpt_steps,
    get_step_path,
    get_weights_dir,
)


class CheckpointType(str, Enum):
    TRAINER = "trainer"
    WEIGHTS = "weights"
    ORCHESTRATOR = "orchestrator"
    ALL = "all"


@contextmanager
def offline_disabled():
    """Temporarily disable HF offline mode for uploads."""
    offline_vars = ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]
    saved = {k: os.environ.pop(k, None) for k in offline_vars}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def clear_uploading_markers(output_dir: Path) -> int:
    """Remove all .uploading markers from checkpoint directories. Returns count of markers removed."""
    logger = get_logger()
    count = 0
    for marker in output_dir.rglob(UPLOADING_MARKER):
        logger.debug(f"Removing stale marker: {marker}")
        marker.unlink()
        count += 1
    if count > 0:
        logger.info(f"Cleared {count} stale {UPLOADING_MARKER} marker(s)")
    return count


class CheckpointUploader:
    """Handles uploading checkpoints to HuggingFace Hub."""

    def __init__(
        self,
        output_dir: Path,
        org: str = "PrimeIntellect",
        repo_prefix: str = "INTELLECT-4",
        num_workers: int = 8,
        public: bool = False,
        dry_run: bool = False,
    ):
        self.output_dir = output_dir
        self.org = org
        self.repo_prefix = repo_prefix
        self.num_workers = num_workers
        self.public = public
        self.dry_run = dry_run
        self.api = HfApi()
        self.logger = get_logger()

    def _weights_repo_id(self, step: int) -> str:
        """Repo ID for weights: {org}/{prefix}-{step}"""
        return f"{self.org}/{self.repo_prefix}-{step}"

    def _ckpt_repo_id(self, step: int) -> str:
        """Repo ID for checkpoints: {org}/{prefix}-{step}-Ckpt"""
        return f"{self.org}/{self.repo_prefix}-{step}-Ckpt"

    def _ensure_repo(self, repo_id: str) -> None:
        """Create repo if it doesn't exist."""
        if self.dry_run:
            self.logger.info(f"[DRY-RUN] Would create repo: {repo_id}")
            return
        try:
            self.api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=not self.public,
                exist_ok=True,
            )
            self.logger.info(f"Ensured repo exists: {repo_id}")
        except Exception as e:
            self.logger.error(f"Failed to create repo {repo_id}: {e}")
            raise

    def _upload_folder(self, folder_path: Path, repo_id: str) -> None:
        """Upload a folder using upload_large_folder."""
        if not folder_path.exists():
            self.logger.warning(f"Folder does not exist, skipping: {folder_path}")
            return

        if self.dry_run:
            self.logger.info(f"[DRY-RUN] Would upload {folder_path} -> {repo_id}")
            return

        # Create marker to prevent trainer from deleting during upload
        marker = folder_path / UPLOADING_MARKER
        try:
            marker.touch()
            self.logger.info(f"Uploading {folder_path} -> {repo_id}")

            # Enable high performance mode for Xet backend
            os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

            self.api.upload_large_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=str(folder_path),
                num_workers=self.num_workers,
            )
            self.logger.info(f"Completed upload: {folder_path} -> {repo_id}")
        finally:
            marker.unlink(missing_ok=True)

    def upload_weights(self, step: int) -> None:
        """Upload weights checkpoint for a step."""
        weights_dir = get_weights_dir(self.output_dir)
        step_dir = get_step_path(weights_dir, step)
        repo_id = self._weights_repo_id(step)

        with offline_disabled():
            self._ensure_repo(repo_id)
            self._upload_folder(step_dir, repo_id)

    def upload_trainer(self, step: int) -> None:
        """Upload trainer checkpoint for a step."""
        ckpt_dir = get_ckpt_dir(self.output_dir)
        step_dir = get_step_path(ckpt_dir, step)
        trainer_dir = step_dir / "trainer"
        repo_id = self._ckpt_repo_id(step)

        with offline_disabled():
            self._ensure_repo(repo_id)
            self._upload_folder(trainer_dir, repo_id)

    def upload_orchestrator(self, step: int) -> None:
        """Upload orchestrator checkpoint for a step."""
        # Find orchestrator dir - could be run_0, run_1, etc.
        orch_dirs = list(self.output_dir.glob("run_*"))
        if not orch_dirs:
            self.logger.warning(f"No orchestrator directories found in {self.output_dir}")
            return

        # Use first orchestrator dir (single-tenant assumption for uploads)
        orch_dir = sorted(orch_dirs)[0]
        ckpt_dir = get_ckpt_dir(orch_dir)
        step_dir = get_step_path(ckpt_dir, step)
        orch_ckpt_dir = step_dir / "orchestrator"
        repo_id = self._ckpt_repo_id(step)

        with offline_disabled():
            self._ensure_repo(repo_id)
            self._upload_folder(orch_ckpt_dir, repo_id)

    def upload_step(self, step: int, ckpt_type: CheckpointType) -> None:
        """Upload checkpoint(s) for a specific step."""
        self.logger.info(f"Uploading step {step}, type={ckpt_type.value}")

        if ckpt_type == CheckpointType.WEIGHTS:
            self.upload_weights(step)
        elif ckpt_type == CheckpointType.TRAINER:
            self.upload_trainer(step)
        elif ckpt_type == CheckpointType.ORCHESTRATOR:
            self.upload_orchestrator(step)
        elif ckpt_type == CheckpointType.ALL:
            self.upload_weights(step)
            self.upload_trainer(step)
            self.upload_orchestrator(step)

    def get_uploadable_steps(self, ckpt_type: CheckpointType) -> list[int]:
        """Get steps that can be uploaded based on checkpoint type. Returns sorted list."""
        if ckpt_type == CheckpointType.WEIGHTS:
            return get_all_ckpt_steps(get_weights_dir(self.output_dir))
        elif ckpt_type == CheckpointType.TRAINER:
            return get_all_ckpt_steps(get_ckpt_dir(self.output_dir))
        elif ckpt_type == CheckpointType.ORCHESTRATOR:
            orch_dirs = list(self.output_dir.glob("run_*"))
            if not orch_dirs:
                return []
            return get_all_ckpt_steps(get_ckpt_dir(sorted(orch_dirs)[0]))
        else:  # ALL - use common steps
            dirs = [get_weights_dir(self.output_dir), get_ckpt_dir(self.output_dir)]
            orch_dirs = list(self.output_dir.glob("run_*"))
            if orch_dirs:
                dirs.append(get_ckpt_dir(sorted(orch_dirs)[0]))
            return get_common_ckpt_steps(dirs)

    def upload_all(self, ckpt_type: CheckpointType, steps: list[int] | None = None) -> None:
        """Upload all (or specified) steps.

        If steps is None, auto-discover and sort ascending (oldest first).
        If steps is provided, use the order as-is.
        """
        if steps is None:
            steps = self.get_uploadable_steps(ckpt_type)
            # Auto-discovered steps are already sorted by get_all_ckpt_steps/get_common_ckpt_steps

        if not steps:
            self.logger.warning(f"No steps to upload for type={ckpt_type.value}")
            return

        self.logger.info(f"Uploading {len(steps)} steps: {steps}")

        for step in steps:
            self.upload_step(step, ckpt_type)


class UploadConfig(BaseSettings):
    """Configuration for checkpoint upload CLI."""

    output_dir: Annotated[Path, CliPositionalArg] = Field(description="Output directory containing checkpoints")
    org: str = Field(default="PrimeIntellect", description="HuggingFace organization")
    repo_prefix: str = Field(default="INTELLECT-4", description="Repository name prefix")
    type: CheckpointType = Field(default=CheckpointType.ALL, description="Checkpoint type to upload")
    steps: list[int] | None = Field(default=None, description="Specific steps to upload in given order")
    num_workers: int = Field(default=8, description="Number of upload workers")
    public: bool = Field(default=False, description="Create public repositories")
    dry_run: bool = Field(default=False, description="Show what would be uploaded without actually uploading")


def main():
    """CLI entry point for checkpoint upload."""
    config = CliApp.run(UploadConfig)

    logger = get_logger()
    logger.info(f"Upload config: {config}")

    # Clear any stale markers from previous runs
    clear_uploading_markers(config.output_dir)

    uploader = CheckpointUploader(
        output_dir=config.output_dir,
        org=config.org,
        repo_prefix=config.repo_prefix,
        num_workers=config.num_workers,
        public=config.public,
        dry_run=config.dry_run,
    )

    uploader.upload_all(config.type, config.steps)
    logger.info("Upload complete!")


if __name__ == "__main__":
    main()
