import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save

from prime_rl.trainer.ckpt import AppState
from prime_rl.trainer.utils import get_dcp_group
from prime_rl.trainer.world import get_world

FINISHED_MARKER = ".finished"


def snapshot_dir(run_dir: Path) -> Path:
    return run_dir / "snapshots"


def finished_snapshot_steps(run_dir: Path) -> list[int]:
    """Steps of the complete snapshots, ascending."""
    paths = snapshot_dir(run_dir).glob(f"step_*/{FINISHED_MARKER}")
    return sorted(int(path.parent.name.removeprefix("step_")) for path in paths)


def delete_snapshots_up_to(run_dir: Path, step: int) -> None:
    for path in snapshot_dir(run_dir).glob("step_*"):
        if int(path.name.removeprefix("step_")) <= step:
            shutil.rmtree(path, ignore_errors=True)


def snapshot_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """A copy of one model tensor for the snapshot, in bf16 (the dtype the forward computes in, so scoring is
    unchanged and the snapshot is half the size), except the router (gate weights and selection bias), which
    runs in fp32 and decides the expert selection."""
    if tensor.is_floating_point() and "router" not in name:
        return tensor.to(torch.bfloat16, copy=True)
    return tensor.clone()


class SnapshotWriter:
    """Model-only DCP snapshots for the eval worker (`prime_rl.trainer.pretrain.eval_worker`), in the layout of a
    model-only checkpoint (`{"app": AppState}`), in bf16 except the router (`snapshot_tensor`); loading casts back
    to the model's dtypes. The trainer only waits for a copy of its weight shards on the GPU;
    a background thread writes them. A snapshot is complete once its `.finished` marker exists; the eval worker
    deletes it after scoring it."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending: Future | None = None
        # Created here, on every rank at the same point; the writer thread only uses it.
        self.group = get_dcp_group()

    def save(self, model: nn.Module, step: int) -> None:
        self.wait()
        state = AppState(model, [], None, None).state_dict()
        state["model"] = {name: snapshot_tensor(name, tensor) for name, tensor in state["model"].items()}
        self.pending = self.executor.submit(self.write, state, snapshot_dir(self.run_dir) / f"step_{step}")

    def write(self, state: dict, path: Path) -> None:
        dcp_save({"app": state}, checkpoint_id=path, process_group=self.group)
        if get_world().is_master:
            (path / FINISHED_MARKER).touch()

    def wait(self) -> None:
        """Block until the previous snapshot is on disk."""
        if self.pending is not None:
            self.pending.result()
            self.pending = None
