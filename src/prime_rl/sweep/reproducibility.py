"""Checksum and git metadata helpers for sweep manifests."""

import hashlib
import subprocess
from pathlib import Path
from typing import TypedDict


class GitMetadata(TypedDict):
    sha: str | None
    dirty: bool | None


def file_checksum(path: Path) -> str:
    """SHA-256 hex digest of a file's contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_metadata(cwd: Path | None = None) -> GitMetadata:
    """Capture git commit SHA and dirty flag at study creation time.

    Returns ``{"sha": None, "dirty": None}`` when not running inside a git
    work tree or when git is unavailable. The absence is recorded explicitly
    in the manifest rather than silently dropped.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"sha": sha, "dirty": status.strip() != ""}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}
