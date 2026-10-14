from collections import defaultdict

from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader

from prime_rl.trainer.world import get_world


class StatefulIterableDataset(Stateful, IterableDataset):
    """SFT dataset are iterable (infinite) and stateful (can be checkpointed)."""

    def __init__(self, non_dp_size: int = 1):
        self.step, self.epoch = 0, 0
        self.num_samples = defaultdict(int)
        self.num_tokens = defaultdict(int)
        self.fast_forward = False
        self.non_dp_size = non_dp_size
        self._setup_world_info()

    def state_dict(self) -> dict:
        return {"step": self.step, "epoch": self.epoch}

    def load_state_dict(self, state_dict: dict):
        assert "step" in state_dict and "epoch" in state_dict
        self.fast_forward = True
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]

    def _setup_world_info(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id, num_workers = 0, 1
        world = get_world()
        assert world.world_size % self.non_dp_size == 0, "world_size must be divisible by non_dp_size"
        self.data_rank = world.rank // self.non_dp_size * num_workers + worker_id
        self.data_world_size = world.world_size // self.non_dp_size * num_workers


def get_dataset_state(dataloader: StatefulDataLoader) -> dict:
    """Dataset position per worker for the startup log, parsed from ``StatefulDataLoader.state_dict()``.

    The loader is the only source that is correct in every case: after a resume's
    ``load_state_dict`` the restored position exists solely in the loader's stashed
    state (it reaches the dataset copies inside workers when the iterator forks them;
    the main-process dataset object stays at position zero). The keys are torchdata's
    private worker-snapshot layout."""
    snapshots = dataloader.state_dict()["_snapshot"]["_worker_snapshots"]
    return {wid: snap["dataset_state"]["dataset"] for wid, snap in sorted(snapshots.items())}


def get_dataset_progress(dataloader: StatefulDataLoader) -> dict:
    """Dataset position and aggregate counters from dataloader workers."""
    snapshot = dataloader.state_dict()["_snapshot"]
    worker_snapshots = snapshot["_worker_snapshots"]
    positions = [worker_snapshot["dataset_state"]["dataset"] for worker_snapshot in worker_snapshots.values()]
    furthest = max(positions, key=lambda position: position["step"])
    num_samples = defaultdict(int)
    num_tokens = defaultdict(int)
    for worker_snapshot in worker_snapshots.values():
        progress = worker_snapshot["dataset_state"].get("progress", {})
        for name, count in progress.get("num_samples", {}).items():
            num_samples[name] += count
        for name, count in progress.get("num_tokens", {}).items():
            num_tokens[name] += count
    return {
        **furthest,
        "num_samples": dict(num_samples),
        "num_tokens": dict(num_tokens),
    }
