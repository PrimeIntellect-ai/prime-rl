import shutil
from pathlib import Path

from prime_rl.orchestrator.utils import min_fresh_version
from prime_rl.utils.logger import get_logger


class PayloadGC:
    """Owns the payload root. Each per-policy-version directory (``v<k>``, written by
    the inference servers for requests salted with version ``k``) is deleted once no
    batch the trainer has yet to read can reference it.

    Trainer step ``s`` reads batch ``s`` and then broadcasts policy ``v{s}``. So when
    inference applies ``v{step}``, the trainer has read every batch up to ``step``. The
    unread batches are ``step + 1`` and later, and batch ``b`` only holds rollouts
    dispatched at ``min_fresh_version(b)`` or newer (the train sink's stale sweep). That
    makes ``min_fresh_version(step + 1)`` the oldest version still referenced.

    A fresh run wipes the root. A resumed run keeps it, because a trainer that outlived
    the orchestrator may still read batches that point into it. The startup sync
    (``v{resume_step}``) then collects by version like any later update."""

    def __init__(self, root: Path, max_off_policy_steps: int, resume: bool):
        self.root = root
        self.max_off_policy_steps = max_off_policy_steps
        if not resume:
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)

    def collect(self, step: int) -> None:
        oldest_kept = min_fresh_version(step + 1, self.max_off_policy_steps)
        for path in self.root.iterdir():
            version = path.name.removeprefix("v")
            if path.is_dir() and version.isdigit() and int(version) < oldest_kept:
                shutil.rmtree(path, ignore_errors=True)
                get_logger().debug(f"Deleted payload directory {path}")
