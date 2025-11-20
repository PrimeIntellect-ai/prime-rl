from pathlib import Path


class Runs:
    """This class stores information about the runs in the system."""

    def __init__(self, output_dir: Path, max_runs: int):
        self.output_dir = output_dir
        self.max_runs = max_runs

        self.idx_2_id = {}
        self.id_2_idx = {}
        self._unused_idxs = {i for i in range(self.max_runs)}

    def check_for_changes(self) -> None:
        run_ids = {run_path.stem for run_path in self.output_dir.glob("run_*")}
        deleted_runs = self.id_2_idx.keys() - run_ids
        new_runs = run_ids - self.id_2_idx.keys()

        for deleted_run in deleted_runs:
            # TODO: Support hooks?
            self._unused_idxs.add(self.id_2_idx[deleted_run])
            del self.idx_2_id[self.id_2_idx[deleted_run]]
            del self.id_2_idx[deleted_run]

        for new_run in new_runs:
            try:
                self.id_2_idx[new_run] = next(iter(self._unused_idxs))
                self._unused_idxs.remove(self.id_2_idx[new_run])
                self.idx_2_id[self.id_2_idx[new_run]] = new_run
                # TODO: Support hooks?
            except StopIteration:
                continue

    def run_dirs(self) -> list[Path]:
        return [self.output_dir / run_id for run_id in self.id_2_idx.keys()]

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
