import weakref

import torch
from torch.distributed.checkpoint import load, save

from prime_rl.trainer.checkpoint_writer import CheckpointFileWriter


def test_checkpoint_releases_staged_tensors_between_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(CheckpointFileWriter, "_shard_bytes", 1024)
    base = torch.arange(6 * 128, dtype=torch.float32)
    state = {f"tensor_{index}": tensor for index, tensor in enumerate(base.split(128))}
    staged_tensors = []
    live_counts = []
    torch_save = torch.save

    def track_tensors(value, *args, **kwargs):
        torch_save(value, *args, **kwargs)
        if isinstance(value, torch.Tensor):
            staged_tensors.append(weakref.ref(value))
            live_counts.append(sum(reference() is not None for reference in staged_tensors))

    monkeypatch.setattr(torch, "save", track_tensors)

    save(state, storage_writer=CheckpointFileWriter(tmp_path), no_dist=True)
    restored = {key: torch.empty_like(value) for key, value in state.items()}
    load(restored, checkpoint_id=tmp_path, no_dist=True)

    assert max(live_counts) == 2
    assert len(list(tmp_path.glob("*.distcp"))) == 3
    assert all(torch.equal(restored[key], value) for key, value in state.items())
