from dataclasses import dataclass
from pathlib import Path
from typing import Callable


# TODO: Delete the one in ckpt.py?
@dataclass
class Progress:
    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0


class Runs:
    """This class stores information about the runs in the system."""

    def __init__(self, output_dir: Path, max_runs: int):
        self.output_dir = output_dir
        self.max_runs = max_runs

        self.idx_2_id: dict[int, str] = {}
        self.id_2_idx: dict[str, int] = {}
        self.unused_idxs = {i for i in range(self.max_runs)}

        self.progress: dict[int, Progress] = {}
        self.ready_to_update = [False] * max_runs

        self._deletion_hooks: list[Callable[[int, str], None]] = []
        self._creation_hooks: list[Callable[[int, str], None]] = []

    def check_for_changes(self) -> None:
        run_ids = {run_path.stem for run_path in self.output_dir.glob("run_*")}
        deleted_runs = self.id_2_idx.keys() - run_ids
        new_runs = run_ids - self.id_2_idx.keys()

        for deleted_run in deleted_runs:
            deleted_idx = self.id_2_idx[deleted_run]
            del self.progress[deleted_idx]
            self.ready_to_update[deleted_idx] = False

            # Process mappings
            self.unused_idxs.add(deleted_idx)
            del self.idx_2_id[deleted_idx]
            del self.id_2_idx[deleted_run]

            # Call deletion hooks
            for hook in self._deletion_hooks:
                hook(deleted_idx, deleted_run)

        for new_run in new_runs:
            try:
                # Process mappings
                new_id = next(iter(self.unused_idxs))
                self.id_2_idx[new_run] = new_id
                self.unused_idxs.remove(new_id)
                self.idx_2_id[new_id] = new_run

                # Process start args
                self.progress[new_id] = Progress()

                prev_ckpt_steps = [
                    int(i.stem.split("_")[-1]) for i in (self.get_run_dir(new_id) / "checkpoints").glob("step_*")
                ]
                self.progress[new_id].step = max(prev_ckpt_steps) if prev_ckpt_steps else 0

                # Call creation hooks
                for hook in self._creation_hooks:
                    hook(new_id, new_run)
            except StopIteration:
                continue

    @property
    def used_idxs(self):
        return self.idx_2_id.keys()

    def run_dirs(self) -> list[Path]:
        return [self.output_dir / run_id for run_id in self.id_2_idx.keys()]

    def get_run_dir(self, idx: int) -> Path:
        return self.output_dir / self.idx_2_id[idx]

    def register_deletion_hook(self, hook: Callable[[int, str], None]) -> None:
        """Register a hook to be called when a run is deleted.

        Args:
            hook: A callable that takes (idx: int, run_id: str) as arguments.
                  Called when a run is deleted from the system.
        """
        self._deletion_hooks.append(hook)

    def register_creation_hook(self, hook: Callable[[int, str], None]) -> None:
        """Register a hook to be called when a new run is created.

        Args:
            hook: A callable that takes (idx: int, run_id: str) as arguments.
                  Called when a new run is added to the system.
        """
        self._creation_hooks.append(hook)

    def __repr__(self):
        return f"Runs(max={self.max_runs})[{self.idx_2_id.keys()}]"


# Singleton instance of Tenants
_RUNS: Runs | None = None


def get_runs() -> Runs:
    """Returns the World. If not initialized, it will initialize."""
    global _RUNS
    if _RUNS is None:
        raise RuntimeError("Runs not initialized. Please call `setup_runs` first.")
    return _RUNS


def setup_runs(output_dir: Path, max_runs: int):
    global _RUNS
    _RUNS = Runs(output_dir, max_runs)
