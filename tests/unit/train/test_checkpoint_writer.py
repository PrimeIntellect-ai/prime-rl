"""DCP's file writer copies a CPU tensor that is a view into a larger storage, which full offload's slab-backed masters
and split fused parameters are, and keeps every tensor it wrote to a file until the file is done. `CheckpointWriter`
bounds the files, so the copies of a save are freed file by file instead of piling up for the whole rank shard."""

import weakref
from unittest.mock import patch

import torch
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint import load as dcp_load
from torch.distributed.checkpoint import save as dcp_save
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, MetadataIndex, TensorProperties
from torch.distributed.checkpoint.planner import TensorWriteData, WriteItem, WriteItemType

from prime_rl.trainer.ckpt import CheckpointWriter, _write_item_bytes

NUM_VIEWS = 6
VIEW_NUMEL = 1024


def slab_views() -> dict[str, torch.Tensor]:
    slab = torch.randn(NUM_VIEWS * VIEW_NUMEL + 64)
    return {f"w{i}": slab.narrow(0, i * VIEW_NUMEL, VIEW_NUMEL).view(32, 32) for i in range(NUM_VIEWS)}


def max_live_copies(state_dict: dict[str, torch.Tensor], writer: FileSystemWriter) -> int:
    """The most copies DCP's writer holds at once while it saves `state_dict`."""
    copies: list[weakref.ref] = []
    peak = 0
    clone = torch.Tensor.clone

    def tracked_clone(tensor, *args, **kwargs):
        nonlocal peak
        copy = clone(tensor, *args, **kwargs)
        copies.append(weakref.ref(copy))
        peak = max(peak, sum(ref() is not None for ref in copies))
        return copy

    with patch.object(torch.Tensor, "clone", tracked_clone):
        dcp_save(state_dict, storage_writer=writer, no_dist=True)
    return peak


def test_default_writer_holds_a_copy_of_every_view(tmp_path):
    assert max_live_copies(slab_views(), FileSystemWriter(tmp_path)) == NUM_VIEWS


def test_bounded_files_hold_the_copies_of_at_most_two_files(tmp_path):
    # One view per file. The writer's copy-ahead loader starts the next file while the last one's copies are still
    # referenced, so up to two files' worth are live at once.
    views = slab_views()
    assert max_live_copies(views, CheckpointWriter(tmp_path, max_file_bytes=VIEW_NUMEL * 4)) <= 2

    loaded = {name: torch.zeros_like(view) for name, view in views.items()}
    dcp_load(loaded, checkpoint_id=tmp_path, no_dist=True)
    for name, view in views.items():
        assert torch.equal(loaded[name], view), name


def test_files_are_sized_by_the_shard_a_rank_writes():
    # A rank of an 8-way FSDP save writes 1/8 of a (1024, 1024) bf16 weight.
    shard = TensorWriteData(
        chunk=ChunkStorageMetadata(offsets=torch.Size([128, 0]), sizes=torch.Size([128, 1024])),
        properties=TensorProperties(dtype=torch.bfloat16),
        size=torch.Size([1024, 1024]),
    )
    item = WriteItem(index=MetadataIndex("w"), type=WriteItemType.SHARD, tensor_data=shard)
    assert _write_item_bytes(item) == 128 * 1024 * 2
