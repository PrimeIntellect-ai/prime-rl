import os
import threading
import time
from pathlib import Path

import torch
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, MetadataIndex, TensorProperties
from torch.distributed.checkpoint.planner import TensorWriteData, WriteItem, WriteItemType
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save

from prime_rl.trainer.ckpt import ChunkedFileSystemWriter, chunk_write_items

MiB = 1024**2


def slab_views(numel: int, count: int, fill: float) -> dict[str, torch.Tensor]:
    # Like full optimizer offload: every state tensor is a view into one CPU slab.
    slab = torch.arange(numel, dtype=torch.float32) + fill
    step = numel // count
    return {f"t{i}": slab.narrow(0, i * step, step).view(step // 64, 64) for i in range(count)}


def rss_bytes() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def test_chunked_writer_round_trips_and_splits_files(tmp_path: Path) -> None:
    state = slab_views(numel=4 * MiB, count=32, fill=0.5)  # 16 MiB of float32
    writer = ChunkedFileSystemWriter(tmp_path, chunk_bytes=2 * MiB, thread_count=2)
    dcp_save({"state": state}, storage_writer=writer, no_dist=True)

    # 16 MiB in chunks of at most 2 MiB (4 views of 512 KiB each), two files per chunk, none of them empty.
    files = list(tmp_path.glob("*.distcp"))
    assert len(files) == 16
    assert all(path.stat().st_size > 0 for path in files)

    loaded = {name: torch.zeros_like(tensor) for name, tensor in state.items()}
    dcp_load({"state": loaded}, checkpoint_id=tmp_path, no_dist=True)
    for name, tensor in state.items():
        assert torch.equal(loaded[name], tensor), name


def test_chunked_writer_bounds_extra_host_memory(tmp_path: Path) -> None:
    numel = 128 * MiB  # 512 MiB of float32 views
    state = slab_views(numel=numel, count=64, fill=1.0)
    baseline = rss_bytes()
    peak = baseline
    done = False

    def watch() -> None:
        nonlocal peak
        while not done:
            peak = max(peak, rss_bytes())
            time.sleep(0.002)

    watcher = threading.Thread(target=watch)
    watcher.start()
    try:
        writer = ChunkedFileSystemWriter(tmp_path, chunk_bytes=32 * MiB, thread_count=1)
        dcp_save({"state": state}, storage_writer=writer, no_dist=True)
    finally:
        done = True
        watcher.join()

    # torch's writer clones every view of a file and keeps the clones until the file is written; one file per rank
    # meant a full second copy (+512 MiB here). Chunked, only about one chunk of clones is alive at a time.
    assert peak - baseline < 128 * MiB, f"save grew RSS by {(peak - baseline) / MiB:.0f} MiB"


def shard_item(name: str, global_rows: int, local_rows: int) -> WriteItem:
    # A rank's shard of a [global_rows, 1024] float32 tensor, as DCP plans it for a DTensor.
    return WriteItem(
        index=MetadataIndex(name, torch.Size([0, 0])),
        type=WriteItemType.SHARD,
        tensor_data=TensorWriteData(
            chunk=ChunkStorageMetadata(offsets=torch.Size([0, 0]), sizes=torch.Size([local_rows, 1024])),
            properties=TensorProperties(dtype=torch.float32),
            size=torch.Size([global_rows, 1024]),
        ),
    )


def test_chunks_are_sized_by_the_local_shard() -> None:
    # Each item writes 1 MiB locally but is a 32 MiB tensor globally: chunks must count the 1 MiB.
    items = [shard_item(f"w{i}", global_rows=8192, local_rows=256) for i in range(10)]
    chunks = chunk_write_items(items, chunk_bytes=4 * MiB)
    assert [len(chunk) for chunk in chunks] == [4, 4, 2]
    assert [item for chunk in chunks for item in chunk] == items
