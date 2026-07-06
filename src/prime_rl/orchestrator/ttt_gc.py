"""GC for TTT adapter checkpoints — the replay artifacts the TTT service leaves on disk.

Every compaction of a TTT rollout writes a versioned adapter checkpoint
(`outputs/ttt/<rollout_id>/v<k>/`), which the trainer needs exactly once: to replay the
rollout's branches in the training step that ships them. After that step is consumed the
checkpoints are dead weight, and a rollout that never ships (errored, filtered, dropped
group) leaves them orphaned. The orchestrator drives this GC because it alone knows both
lifecycles:

- ``track_batch(step, batch)``: called at ship time. Rollout dirs referenced by shipped
  samples are deferred until the trainer finishes that step; dirs of the batch's dropped
  rollouts are deleted immediately (nothing will ever read them).
- ``on_new_version(version)``: the weight watcher observed the trainer's broadcast for
  ``version`` — every step ≤ ``version`` is consumed, so its deferred dirs are deleted.

Deletion is best-effort (`ignore_errors`): a vanished dir (e.g. the TTT service configured
with ``keep_checkpoints=false``) is not an error.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout, TrainBatch


def rollout_ckpt_dirs(rollout: "Rollout") -> set[Path]:
    """The rollout's adapter checkpoint dir(s), derived from its recorded updates
    (``trace.info["ttt"]``) — one per rollout in practice (`<ttt_root>/<rollout_id>`)."""
    dirs: set[Path] = set()
    for update in rollout.info.get("ttt", {}).get("updates", []):
        path = update.get("ckpt_path")
        if path:
            dirs.add(Path(path).parent)
    return dirs


class TTTCheckpointGC:
    """Tracks shipped batches' adapter dirs and deletes them once consumed."""

    def __init__(self) -> None:
        self._deferred: dict[int, set[Path]] = {}

    def track_batch(self, step: int, batch: "TrainBatch") -> None:
        """Register a shipped batch: defer shipped rollouts' dirs to the trainer's
        consumption of ``step``; drop unshipped rollouts' dirs immediately."""
        shipped_paths = {s.ttt_adapter_path for s in batch.samples if s.ttt_adapter_path}
        shipped_dirs = {Path(p).parent for p in shipped_paths}
        dropped_dirs: set[Path] = set()
        for rollout in batch.rollouts:
            dropped_dirs |= rollout_ckpt_dirs(rollout)
        dropped_dirs -= shipped_dirs
        if dropped_dirs:
            self._delete(dropped_dirs, reason=f"dropped rollouts of step {step}")
        if shipped_dirs:
            self._deferred.setdefault(step, set()).update(shipped_dirs)

    def on_new_version(self, version: int) -> None:
        """The trainer broadcast weights for ``version`` — every step ≤ it is consumed."""
        due = [step for step in self._deferred if step <= version]
        for step in due:
            self._delete(self._deferred.pop(step), reason=f"consumed step {step}")

    def _delete(self, dirs: set[Path], reason: str) -> None:
        for path in sorted(dirs):
            shutil.rmtree(path, ignore_errors=True)
        get_logger().debug(f"TTT ckpt GC: removed {len(dirs)} adapter dir(s) ({reason})")
