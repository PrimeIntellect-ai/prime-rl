"""Filesystem-marker handshake between trainer broadcast backends and the orchestrator.

In-band weight transports (NCCL, NIXL) coordinate each sync through two
markers in the run's broadcast step directory:

* ``STABLE`` — created by the trainer master to tell the orchestrator a new
  weight version is ready to be pushed.
* ``NCCL_READY`` — created by the orchestrator (see
  ``prime_rl.utils.client.update_weights``) after it has paused all inference
  engines, telling every trainer rank it is safe to start the transfer.
"""

from pathlib import Path

from prime_rl.utils.pathing import sync_wait_for_path
from prime_rl.utils.utils import get_broadcast_dir, get_step_path

READY_MARKER = "NCCL_READY"


class OrchestratorMarkers:
    """Mixin for broadcast backends; requires ``logger``, ``world`` and
    ``multi_run_manager`` attributes on the host class."""

    def _compute_notified_runs(self) -> list[tuple[int, Path]]:
        """Derive the list of (run_idx, save_dir) pairs that need broadcasting.

        Pure function of `multi_run_manager` state, which is replicated across
        trainer ranks (SPMD). Returns the same list on every rank so master and
        non-master ranks agree on which ready markers to wait for.
        """
        notified_runs: list[tuple[int, Path]] = []
        for idx in self.multi_run_manager.used_idxs:
            if not self.multi_run_manager.ready_to_update[idx]:
                continue
            try:
                save_dir = get_step_path(
                    get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                    self.multi_run_manager.progress[idx].step,
                )
                notified_runs.append((idx, save_dir))
            except FileNotFoundError:
                self.logger.warning(f"Run {idx} is deleted, skipping")
            except Exception as e:
                self.logger.error(f"Error resolving broadcast dir for run {idx}: {e}")
        return notified_runs

    def _notify_orchestrator(self, notified_runs: list[tuple[int, Path]]) -> None:
        """Create STABLE markers for each notified run and clear their ready flags.

        Master-only side effects (filesystem writes + state mutation). Called
        after `_compute_notified_runs`; non-master ranks skip this entirely.
        """
        for idx, save_dir in notified_runs:
            try:
                save_dir.mkdir(parents=True, exist_ok=True)
                stable_file = save_dir / "STABLE"
                stable_file.touch()
            except FileNotFoundError:
                self.logger.warning(f"Run {idx} is deleted, skipping")
            except Exception as e:
                self.logger.error(f"Error broadcasting weights for run {idx}: {e}")
            finally:
                self.multi_run_manager.ready_to_update[idx] = False

    def _wait_for_ready_marker(self, notified_runs: list[tuple[int, Path]]) -> None:
        """Wait for the orchestrator to signal that inference engines are paused."""
        for idx, save_dir in notified_runs:
            ready_file = save_dir / READY_MARKER
            self.logger.debug(f"Waiting for {READY_MARKER} marker at {ready_file}")
            sync_wait_for_path(ready_file, interval=0.1, log_interval=10)
            self.logger.debug(f"Inference engines paused, transfer may start (run {idx})")
