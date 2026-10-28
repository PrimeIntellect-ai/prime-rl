"""CPU tests of [val] keep_best: which validation wins, what stays on disk, exact restore, resume, and a two-rank run
where every rank has its own "node" directory (the volume staged per node), so no rank can see another's files."""

import json
import math
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from prime_rl.trainer.sft.best_ckpt import BEST_JSON, BestCheckpoint, is_improvement, local_tensors
from prime_rl.trainer.world import reset_world

# Validation losses of the first TB4 1M run (eivawxtikvt75p1ietxg4lok) at steps 36..288, plus a NaN pass.
VAL_LOSSES = [0.5969, 0.5909, 0.5949, 0.6056, float("nan"), 0.6202, 0.6350, 0.6438, 0.6468]


def test_is_improvement():
    assert is_improvement(0.7, None)
    assert is_improvement(0.5, 0.6)
    assert not is_improvement(0.6, 0.6)  # ties keep the earlier checkpoint
    assert not is_improvement(0.61, 0.6)
    assert not is_improvement(float("nan"), None)
    assert not is_improvement(float("inf"), 0.6)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 3)
        self.register_buffer("bias_correction", torch.zeros(3))
        # Derived caches are not snapshotted (their shape may change between save and restore).
        self.register_buffer("rope_cache", torch.zeros(2), persistent=False)


def train_step(model: nn.Module, step: int) -> None:
    with torch.no_grad():
        for tensor in local_tensors(model).values():
            tensor.add_(0.01 * step)


def snapshot_files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_keeps_only_the_best_and_restores_it_exactly(tmp_path):
    torch.manual_seed(0)
    model = Tiny()
    keeper = BestCheckpoint(tmp_path / "best-weights")
    expected = None
    for i, loss in enumerate(VAL_LOSSES, start=1):
        step = 36 * i
        train_step(model, step)
        saved = keeper.consider(step, loss, model, val_logged_step=step)
        assert saved == (step in (36, 72))
        if step == 72:
            expected = {k: v.clone() for k, v in local_tensors(model).items()}
        # Only the current best's snapshot is on disk at any time.
        assert snapshot_files(tmp_path / "best-weights") == [
            BEST_JSON,
            f"step_{keeper.best_step}/rank_00000-of-00001.pt",
        ]

    assert keeper.best_step == 72 and keeper.best_loss == 0.5909
    record = json.loads((tmp_path / "best-weights" / BEST_JSON).read_text())
    assert record["best"]["step"] == 72 and record["best"]["val_loss"] == 0.5909
    assert record["world_size"] == 1
    assert len(record["history"]) == len(VAL_LOSSES) and math.isnan(record["history"]["180"])

    # The model has moved on (the last weights); restoring gives back step 72's bytes.
    assert not torch.equal(model.lin.weight, expected["lin.weight"])
    assert set(expected) == {"lin.weight", "lin.bias", "bias_correction"}
    model.rope_cache = torch.zeros(5)
    assert keeper.restore(model)
    for name, tensor in local_tensors(model).items():
        assert torch.equal(tensor, expected[name]), name


def test_resume_picks_up_the_best(tmp_path):
    model = Tiny()
    first = BestCheckpoint(tmp_path)
    first.consider(20, 0.6, model, val_logged_step=20)
    first.consider(40, 0.7, model, val_logged_step=40)

    resumed = BestCheckpoint(tmp_path)
    resumed.load_state()
    assert resumed.best_step == 20 and resumed.history == {20: 0.6, 40: 0.7}
    # A later, worse loss does not replace it; a better one does and removes step 20.
    assert not resumed.consider(60, 0.65, model, val_logged_step=60)
    assert resumed.consider(80, 0.55, model, val_logged_step=80)
    assert snapshot_files(tmp_path) == [BEST_JSON, "step_80/rank_00000-of-00001.pt"]


def test_resume_with_other_world_size_starts_over(tmp_path):
    model = Tiny()
    BestCheckpoint(tmp_path).consider(20, 0.6, model, val_logged_step=20)
    record = json.loads((tmp_path / BEST_JSON).read_text())
    record["world_size"] = 32
    (tmp_path / BEST_JSON).write_text(json.dumps(record))
    resumed = BestCheckpoint(tmp_path)
    resumed.load_state()
    assert resumed.best is None and resumed.history == {20: 0.6}


def test_failed_save_keeps_the_previous_best(tmp_path, monkeypatch):
    model = Tiny()
    keeper = BestCheckpoint(tmp_path)
    keeper.consider(20, 0.6, model, val_logged_step=20)

    def broken_save(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(torch, "save", broken_save)
    assert not keeper.consider(40, 0.5, model, val_logged_step=40)
    assert keeper.best_step == 20
    assert snapshot_files(tmp_path) == [BEST_JSON, "step_20/rank_00000-of-00001.pt"]


def _two_node_worker(rank: int, world_size: int, port: int, base: str, queue) -> None:
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK="0",
        LOCAL_WORLD_SIZE="1",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    reset_world()
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

        mesh = init_device_mesh("cpu", (world_size,))
        torch.manual_seed(0)
        model = Tiny()
        model.lin.weight = nn.Parameter(distribute_tensor(model.lin.weight.data, mesh, [Shard(0)]))
        # Same path on every rank, but each "node" has its own staged copy of the volume.
        root = Path(base) / f"node{rank}" / "best-weights"
        keeper = BestCheckpoint(root)
        expected = None
        for step, loss in [(20, 0.62), (40, 0.59), (60, 0.60)]:
            train_step(model, step)
            keeper.consider(step, loss, model, val_logged_step=step)
            if step == 40:
                expected = {k: v.clone() for k, v in local_tensors(model).items()}
        train_step(model, 80)
        ok = keeper.restore(model)
        same = all(torch.equal(t, expected[n]) for n, t in local_tensors(model).items())
        queue.put((rank, keeper.best_step, ok, same, snapshot_files(root)))
    finally:
        dist.destroy_process_group()


def test_two_ranks_on_separate_nodes(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    mp.spawn(_two_node_worker, args=(2, port, str(tmp_path), queue), nprocs=2, join=True)
    results = sorted(queue.get() for _ in range(2))
    assert results[0] == (0, 40, True, True, [BEST_JSON, "step_40/rank_00000-of-00002.pt"])
    # Rank 1's node holds only its own shard: best.json lives on the master's node only.
    assert results[1] == (1, 40, True, True, ["step_40/rank_00001-of-00002.pt"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
