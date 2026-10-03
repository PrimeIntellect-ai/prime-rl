import shutil
from pathlib import Path

from prime_rl.orchestrator.utils import min_fresh_version
from prime_rl.utils.logger import get_logger


class PayloadGC:
    """Owns the payload root: wiped at startup, then each per-policy-version directory
    (``v<k>``, written by the inference servers for requests salted with version ``k``)
    is deleted once no batch the trainer has yet to read can reference it.

    When inference applies policy ``step``, the trainer has read every batch before
    ``step``; later batches only hold rollouts dispatched at ``min_fresh_version(step)``
    or newer. One extra version of margin covers the off-by-one between batch and
    checkpoint numbering."""

    def __init__(self, root: Path, max_off_policy_steps: int):
        self.root = root
        self.max_off_policy_steps = max_off_policy_steps
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)

    def collect(self, step: int) -> None:
        oldest_kept = min_fresh_version(step, self.max_off_policy_steps) - 1
        for path in self.root.iterdir():
            version = path.name.removeprefix("v")
            if path.is_dir() and version.isdigit() and int(version) < oldest_kept:
                shutil.rmtree(path, ignore_errors=True)
                get_logger().debug(f"Deleted payload directory {path}")
