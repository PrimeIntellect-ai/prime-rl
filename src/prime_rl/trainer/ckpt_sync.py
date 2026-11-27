"""Mirrors trainer checkpoints to an object store from a process of its own, one per node, so no trainer rank uploads,
downloads or coordinates transfers. The trainer's local rank 0 drives it over stdin, one command per line, and reads
its answers from stdout. It runs the commands in order:

``upload <step>`` uploads the files this node's ranks wrote for a step on disk. Node 0 also uploads the step's model
files (config.json, tokenizer) and, once every node's files are in the remote, ``.metadata`` last.
``pull <step>`` downloads this node's share of a step's files, without ``.metadata``, and answers ``pulled <step>``.
``commit <step>`` (node 0, once every node has pulled) downloads ``.metadata`` and answers ``committed <step>``.
``delete-remote <step>`` removes a step from the remote, ``.metadata`` first.
``delete-local <step>`` removes a step from the disk. Queued after the step's upload, it never removes a step from
disk before it is in the remote.
``exit`` finishes the queued commands and exits.

The trainer decides what to delete (``CheckpointManager.maybe_clean``); this process only executes it. A step is in
the remote exactly when its ``.metadata`` is. EOF before ``exit`` means the trainer died: the process exits at once,
mid-transfer, so it never commits a step while a restarted trainer's sync removes uncommitted ones. A failed transfer
exits it non-zero, which the trainer raises on its next command."""

import argparse
import asyncio
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

import obstore
from obstore.exceptions import BaseError
from obstore.store import ObjectStore, from_url
from torch.distributed.checkpoint import FileSystemReader

from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.pathing import get_step_path, try_rmtree

METADATA_FILE = ".metadata"
# R2 needs equal-size multipart parts. 64 MB parts with this many in flight reach the link rate
TRANSFER_PART_BYTES = 64 << 20
UPLOAD_PARTS_PER_FILE = 32
DOWNLOAD_PARTS_IN_FLIGHT = 256
# Other nodes' uploads are only visible in the remote, so node 0 lists the step until they are in
UPLOADS_POLL_SECONDS = 1
UPLOADS_LOG_SECONDS = 60
STEP_KEY = re.compile(r"step_(\d+)/")
SHARD_RANK = re.compile(r"__(\d+)_\d+\.distcp")


class CheckpointSync:
    def __init__(self, remote: ObjectStore, ckpt_dir: Path, node_rank: int, num_nodes: int, ranks_per_node: int):
        self.remote = remote
        self.ckpt_dir = ckpt_dir
        self.node_rank = node_rank
        self.num_nodes = num_nodes
        self.ranks = range(node_rank * ranks_per_node, (node_rank + 1) * ranks_per_node)
        self.world = range(num_nodes * ranks_per_node)
        self.logger = get_logger()

    def commit(self, step: int) -> str:
        """Download ``.metadata`` after every node pulled its share, so a cut-off pull never reads as complete."""
        path = self._get_path(step)
        partial = path / f"{METADATA_FILE}.partial"
        partial.write_bytes(obstore.get(self.remote, _get_metadata_key(step)).bytes())
        partial.replace(path / METADATA_FILE)
        return f"committed {step}"

    def delete_local(self, step: int) -> None:
        try_rmtree(get_step_path(self.ckpt_dir, step))

    def delete_remote(self, step: int) -> None:
        remove_remote_step(self.remote, step)

    def pull(self, step: int) -> str:
        """Download this node's share of a step (every ``num_nodes``-th file, as concurrent byte ranges): the trainer
        checkpoint and the model files next to it."""
        objects = sorted(
            (
                meta
                for meta in obstore.list(self.remote, prefix=f"step_{step}").collect()
                if meta["path"] != _get_metadata_key(step)
            ),
            key=lambda meta: meta["path"],
        )
        start_time = time.perf_counter()
        asyncio.run(self._get(objects[self.node_rank :: self.num_nodes]))
        self.logger.info(f"Pulled step {step} from the remote in {time.perf_counter() - start_time:.1f}s")
        return f"pulled {step}"

    def remove_uncommitted(self) -> None:
        """Remove steps a crash left without ``.metadata``: nothing else would ever read or remove them."""
        steps, committed = get_remote_steps(self.remote)
        for step in steps - committed:
            remove_remote_step(self.remote, step)

    def upload(self, step: int) -> None:
        """Upload the files this node's ranks wrote. Node 0 then commits the step once every node's files are in the
        remote."""
        path = self._get_path(step)
        start_time = time.perf_counter()
        asyncio.run(self._put(_get_rank_files(path, self.ranks)))
        if self.node_rank != 0:
            return

        model_files = [file for file in path.parent.iterdir() if file.is_file()]
        asyncio.run(self._put([*model_files, *path.glob("mapping.json")]))
        self._wait_for_uploads(step, _get_rank_files(path, self.world))
        asyncio.run(self._put([path / METADATA_FILE]))
        self.logger.info(f"Uploaded step {step} to the remote in {time.perf_counter() - start_time:.1f}s")

    async def _get(self, objects: list[dict]) -> None:
        in_flight = asyncio.Semaphore(DOWNLOAD_PARTS_IN_FLIGHT)

        async def get_part(key: str, start: int, end: int, fd: int) -> None:
            async with in_flight:
                part = await obstore.get_range_async(self.remote, key, start=start, end=end)
            os.pwrite(fd, memoryview(part), start)

        async def get_object(meta: dict) -> None:
            file = self.ckpt_dir / meta["path"]
            file.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(file, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
            try:
                size = meta["size"]
                await asyncio.gather(
                    *[
                        get_part(meta["path"], start, min(start + TRANSFER_PART_BYTES, size), fd)
                        for start in range(0, size, TRANSFER_PART_BYTES)
                    ]
                )
            finally:
                os.close(fd)

        await asyncio.gather(*[get_object(meta) for meta in objects])

    def _get_path(self, step: int) -> Path:
        return get_step_path(self.ckpt_dir, step) / "trainer"

    def _get_key(self, file: Path) -> str:
        return file.relative_to(self.ckpt_dir).as_posix()

    async def _put(self, files: list[Path]) -> None:
        await asyncio.gather(
            *[
                obstore.put_async(
                    self.remote,
                    self._get_key(file),
                    file,
                    chunk_size=TRANSFER_PART_BYTES,
                    max_concurrency=UPLOAD_PARTS_PER_FILE,
                )
                for file in files
            ]
        )

    def _wait_for_uploads(self, step: int, files: list[Path]) -> None:
        """Wait until every rank's files are in the remote: the other nodes upload theirs."""
        expected = {self._get_key(file) for file in files}
        start_time = last_log_time = time.perf_counter()
        while missing := expected - {
            meta["path"] for meta in obstore.list(self.remote, prefix=f"step_{step}").collect()
        }:
            if time.perf_counter() - last_log_time >= UPLOADS_LOG_SECONDS:
                last_log_time = time.perf_counter()
                self.logger.warning(
                    f"Step {step} waits {last_log_time - start_time:.0f}s for {len(missing)} files from other nodes,"
                    f" e.g. {sorted(missing)[0]}"
                )
            time.sleep(UPLOADS_POLL_SECONDS)


def get_remote_steps(remote: ObjectStore) -> tuple[set[int], set[int]]:
    """Every step in the remote, and the committed ones: those whose ``.metadata`` was uploaded. Keys outside a
    ``step_<N>/`` are not steps."""
    keys = {meta["path"] for meta in obstore.list(remote).collect()}
    steps = {int(match.group(1)) for key in keys if (match := STEP_KEY.match(key))}
    return steps, {step for step in steps if _get_metadata_key(step) in keys}


def remove_remote_step(remote: ObjectStore, step: int) -> None:
    """``.metadata`` goes first, so a partly removed step never reads as complete. Logs and skips on failure: an
    uncommitted rest is removed by the next start."""
    try:
        keys = [meta["path"] for meta in obstore.list(remote, prefix=f"step_{step}").collect()]
        if _get_metadata_key(step) in keys:
            obstore.delete(remote, _get_metadata_key(step))
        obstore.delete(remote, [key for key in keys if key != _get_metadata_key(step)])
    except BaseError as error:
        get_logger().warning(f"Failed to remove step {step} from the remote: {error}, skipping cleanup")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("remote_dir")
    parser.add_argument("ckpt_dir", type=Path)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--ranks-per-node", type=int, required=True)
    args = parser.parse_args()
    # Answers keep stdout to themselves: anything else written to it, logs included, goes to stderr
    answers = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    setup_logger(tag="ckpt-sync")
    sync = CheckpointSync(
        remote=from_url(args.remote_dir),
        ckpt_dir=args.ckpt_dir,
        node_rank=args.node_rank,
        num_nodes=args.num_nodes,
        ranks_per_node=args.ranks_per_node,
    )
    command_to_handler = {
        "commit": sync.commit,
        "delete-local": sync.delete_local,
        "delete-remote": sync.delete_remote,
        "pull": sync.pull,
        "upload": sync.upload,
    }

    commands: queue.Queue[str] = queue.Queue()
    threading.Thread(target=_read_commands, args=(commands,), daemon=True).start()

    if args.node_rank == 0:
        sync.remove_uncommitted()
    while (line := commands.get()) != "exit":
        command, step = line.split()
        answer = command_to_handler[command](int(step))
        if answer is not None:
            print(answer, file=answers, flush=True)


def _get_metadata_key(step: int) -> str:
    return f"step_{step}/trainer/{METADATA_FILE}"


def _get_rank_files(path: Path, ranks: range) -> list[Path]:
    """The files the save on disk at ``path`` has from ``ranks``: their DCP shards, as its ``.metadata`` lists them,
    and their dataloader states. A shard of an earlier save of the step by more ranks or threads is not listed, so it is
    neither uploaded nor waited for."""
    storage = FileSystemReader(path).read_metadata().storage_data
    shards = sorted({info.relative_path for info in storage.values()})
    dataloader_files = [path / "dataloader" / f"rank_{rank}.pt" for rank in ranks]
    return [
        *(path / shard for shard in shards if int(SHARD_RANK.fullmatch(shard).group(1)) in ranks),
        *(file for file in dataloader_files if file.exists()),
    ]


def _read_commands(commands: queue.Queue[str]) -> None:
    """Queue the trainer's commands. Stdin closing before ``exit`` means the trainer died, so exit at once."""
    for line in sys.stdin:
        commands.put(line.strip())
        if line.strip() == "exit":
            return
    get_logger().error("The trainer exited without closing the checkpoint sync, stopping it")
    os._exit(1)


if __name__ == "__main__":
    main()
