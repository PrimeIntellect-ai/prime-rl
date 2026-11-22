#!/usr/bin/env python3
"""Check release files for credentials, machine paths, and notebook outputs.

Reports file names and finding types only, never the matched secret. Git history
needs a separate scan; this check covers tracked and unignored working files.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DENIED_SUFFIXES = {".sbatch", ".pbs"}
DENIED_PATTERNS = {
    "absolute user home": re.compile(r"/(?:home|Users)/[^/\s]+/"),
    "Slurm directive": re.compile(r"^\s*#SBATCH\b", re.MULTILINE),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "provider token": re.compile(
        r"\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"
        r"|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
        r"|hf_[A-Za-z0-9]{24,}|AKIA[A-Z0-9]{16})\b"
    ),
    "credentials in URL": re.compile(r"https?://[^/\s:@]+:[^/\s@]+@"),
}


def tracked_files() -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return [ROOT / item.decode() for item in output.split(b"\0") if item]


def check_file(path: Path) -> list[str]:
    """Return finding labels without exposing file contents."""
    if not path.is_file():
        return []
    if path.suffix in DENIED_SUFFIXES:
        return ["scheduler-specific file"]
    if (
        path.name == ".env"
        or (path.name.startswith(".env.") and path.name != ".env.example")
        or path.name in {"id_rsa", "id_ed25519", "credentials.json"}
    ):
        return ["credential file"]
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    failures = [label for label, pattern in DENIED_PATTERNS.items() if pattern.search(content)]
    if path.suffix == ".ipynb":
        notebook = json.loads(content)
        if any(
            cell.get("outputs") or cell.get("execution_count") is not None
            for cell in notebook.get("cells", [])
        ):
            failures.append("notebook contains execution state or outputs")
    return failures


def main() -> int:
    failures: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT)
        try:
            failures.extend(f"{relative}: {label}" for label in check_file(path))
        except (OSError, ValueError) as exc:
            failures.append(f"{relative}: could not check file ({type(exc).__name__})")
    if failures:
        print("Release hygiene check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Release hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
