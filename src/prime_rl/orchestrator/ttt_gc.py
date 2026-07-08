"""GC for TTT adapter checkpoints — the replay artifacts the TTT service leaves on disk.

Every compaction of a TTT rollout writes a versioned adapter checkpoint
(`outputs/ttt/<rollout_id>/v<k>/`), which the trainer needs exactly once: to replay the
rollout's branches in the training step that ships them. After that step is consumed the
checkpoints are dead weight, and a rollout that never ships (errored, filtered, dropped
group) leaves them orphaned. The orchestrator drives this GC because it alone knows both
lifecycles:

- ``track_batch(step, batch)``: called at ship time. Rollout dirs referenced by shipped
  samples are deferred until the trainer finishes that step. ``batch.rollouts`` is the
  ARRIVAL window, not the ship cohort — a rollout can arrive in step N's window and ship
  its samples in step N+1 (batch overflow; a group finalizing after the cut) — so only
  conclusively dead rollouts (errored: never tokenized into samples; filtered: excluded
  from shipping) are deleted immediately. Everything else is carried until its dirs show
  up in a later step's shipped samples.
- ``on_new_version(version)``: the weight watcher observed the trainer's broadcast for
  ``version`` — every step ≤ ``version`` is consumed, so its deferred dirs are deleted.

Rollouts that are silently dropped without a filter/error mark (a group-scored group with
a partial failure) stay in the carry set for the run's lifetime — a bounded leak on disk,
never a premature delete. Deletion is best-effort (`ignore_errors`): a vanished dir (e.g.
the TTT service configured with ``keep_checkpoints=false``) is not an error.
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


def dispose_eval_rollout_ckpts(rollout: "Rollout") -> None:
    """Best-effort delete of an eval rollout's adapter checkpoint dirs. Eval rollouts do
    TTT at inference time but their adapters are never replayed by the trainer — dismiss
    immediately (the write-then-delete inefficiency is accepted; skipping the write needs
    a wire-protocol change in the verifiers submodule)."""
    for path in rollout_ckpt_dirs(rollout):
        shutil.rmtree(path, ignore_errors=True)


class TTTCheckpointGC:
    """Tracks shipped batches' adapter dirs and deletes them once consumed."""

    def __init__(self) -> None:
        self._deferred: dict[int, set[Path]] = {}
        # Dirs of window rollouts that neither shipped nor conclusively died yet — they may
        # ship in a LATER step (batch overflow; group finalized after the cut). Moved to
        # ``_deferred`` when their paths appear in a later step's shipped samples.
        self._carry: set[Path] = set()

    def track_batch(self, step: int, batch: "TrainBatch") -> None:
        """Register a shipped batch: defer shipped rollouts' dirs (including carried ones
        that ship now) to the trainer's consumption of ``step``; delete conclusively dead
        rollouts' dirs; carry the undecided rest."""
        shipped_paths = {s.ttt_adapter_path for s in batch.samples if s.ttt_adapter_path}
        shipped_dirs = {Path(p).parent for p in shipped_paths}
        dead_dirs: set[Path] = set()
        undecided_dirs: set[Path] = set()
        for rollout in batch.rollouts:
            dirs = rollout_ckpt_dirs(rollout)
            # Errored rollouts never tokenize into samples; filtered ones never ship.
            # Anything else in the window may still ship in a later step.
            if rollout.has_error or rollout.is_filtered:
                dead_dirs |= dirs
            else:
                undecided_dirs |= dirs
        dead_dirs -= shipped_dirs
        if dead_dirs:
            self._delete(dead_dirs, reason=f"dead rollouts of step {step}")
        self._carry |= undecided_dirs - shipped_dirs
        self._carry -= shipped_dirs
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
