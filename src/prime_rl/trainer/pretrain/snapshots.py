import shutil
from pathlib import Path

import torch.distributed.checkpoint as dcp
from torch import nn

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


class SnapshotWriter:
    """Model-only DCP snapshots for the eval worker (`prime_rl.trainer.pretrain.eval_worker`), written in the
    background: the trainer only waits for the copy of its shards to CPU memory. A snapshot is complete once its
    `.finished` marker exists; the eval worker deletes it after scoring it."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.pending = None

    def save(self, model: nn.Module, step: int) -> None:
        self.wait()
        path = snapshot_dir(self.run_dir) / f"step_{step}"
        future = dcp.async_save(
            {"app": AppState(model, [], None, None)}, checkpoint_id=path, process_group=get_dcp_group()
        )
        if get_world().is_master:
            # The coordinator's save finishes after every rank's shards and the metadata are written.
            future.add_done_callback(lambda done: mark_finished(path) if done.exception() is None else None)
        self.pending = (future, path)

    def wait(self) -> None:
        """Block until the last snapshot is on disk (at the end of training)."""
        if self.pending is not None:
            future, path = self.pending
            future.result()
            if get_world().is_master:
                mark_finished(path)
            self.pending = None


def mark_finished(path: Path) -> None:
    (path / FINISHED_MARKER).touch()
