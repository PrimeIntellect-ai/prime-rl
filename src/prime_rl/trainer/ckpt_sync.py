"""Mirrors trainer checkpoints to an object store from a process of its own, one per node, so no trainer rank uploads,
downloads or coordinates transfers. The trainer's local rank 0 drives it over stdin, one command per line, and reads
its answers from stdout:

``pull <step>`` downloads this node's share of a step's files, without ``.metadata``, and answers ``pulled <step>``.
``commit <step>`` (node 0, once every node has pulled) downloads ``.metadata`` and answers ``committed <step>``.
``upload <step>`` uploads the files this node's ranks wrote for a step on disk. Node 0 then waits until every node's
files are in the remote, uploads ``.metadata`` last, and applies ``keep_last``/``keep_interval`` to the remote and the
disk, so a step is never removed from disk before it is uploaded.

``exit`` finishes the queued commands and exits.

A step is in the remote exactly when its ``.metadata`` is. EOF before ``exit`` means the trainer died: the process
exits at once, mid-transfer, so it never commits a step while a restarted trainer's sync removes uncommitted ones. A
failed transfer exits it non-zero, which the trainer raises on its next command."""

import argparse
import asyncio
import os
import queue
import sys
import threading
import time
from pathlib import Path

import obstore
from obstore.exceptions import BaseError
from obstore.store import ObjectStore, from_url

from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.pathing import get_step_path, get_steps_to_keep, try_rmtree

METADATA_FILE = ".metadata"
# R2 needs equal-size multipart parts. 64 MB parts with this many in flight reach the link rate
TRANSFER_PART_BYTES = 64 << 20
UPLOAD_PARTS_PER_FILE = 32
DOWNLOAD_PARTS_IN_FLIGHT = 256
# Other nodes' uploads are only visible in the remote, so node 0 lists it until they are in
UPLOADS_POLL_SECONDS = 1


class CheckpointSync:
    def __init__(
        self,
        remote: ObjectStore,
        ckpt_dir: Path,
        node_rank: int,
        num_nodes: int,
        ranks_per_node: int,
        keep_last: int | None,
        keep_interval: int | None,
    ):
        self.remote = remote
        self.ckpt_dir = ckpt_dir
        self.node_rank = node_rank
        self.num_nodes = num_nodes
        self.ranks = range(node_rank * ranks_per_node, (node_rank + 1) * ranks_per_node)
        self.keep_last = keep_last
        self.keep_interval = keep_interval
        self.logger = get_logger()

    def commit(self, step: int) -> str:
        """Download ``.metadata`` after every node pulled its share, so a cut-off pull never reads as complete."""
        path = self._get_path(step)
        partial = path / f"{METADATA_FILE}.partial"
        partial.write_bytes(obstore.get(self.remote, f"{_get_key(step)}/{METADATA_FILE}").bytes())
        partial.replace(path / METADATA_FILE)
        return f"committed {step}"

    def pull(self, step: int) -> str:
        """Download this node's share of a step: every ``num_nodes``-th file, as concurrent byte ranges."""
        key = _get_key(step)
        objects = sorted(
            (
                meta
                for meta in obstore.list(self.remote, prefix=key).collect()
                if meta["path"] != f"{key}/{METADATA_FILE}"
            ),
            key=lambda meta: meta["path"],
        )
        start_time = time.perf_counter()
        asyncio.run(self._get(key, self._get_path(step), objects[self.node_rank :: self.num_nodes]))
        self.logger.info(f"Pulled step {step} from the remote in {time.perf_counter() - start_time:.1f}s")
        return f"pulled {step}"

    def remove_uncommitted(self) -> None:
        """Remove steps a crash left without ``.metadata``: nothing else would ever read or remove them."""
        steps, committed = get_remote_steps(self.remote)
        for step in steps - committed:
            self._remove_remote(step)

    def upload(self, step: int) -> None:
        """Upload the files this node's ranks wrote (DCP names its shards ``__<rank>_<n>.distcp``). Node 0 then
        commits the step once every node's files are in the remote."""
        path = self._get_path(step)
        files = [file for rank in self.ranks for file in _get_rank_files(path, rank)]
        start_time = time.perf_counter()
        asyncio.run(self._put(path, step, files))
        if self.node_rank != 0:
            return

        asyncio.run(self._put(path, step, list(path.glob("mapping.json"))))
        self._wait_for_uploads(path, step)
        asyncio.run(self._put(path, step, [path / METADATA_FILE]))
        self.logger.info(f"Uploaded step {step} to the remote in {time.perf_counter() - start_time:.1f}s")
        self._clean()

    def _clean(self) -> None:
        """Apply ``keep_last``/``keep_interval`` to the committed steps, on the remote and on disk."""
        _, committed = get_remote_steps(self.remote)
        steps_to_keep = get_steps_to_keep(sorted(committed), self.keep_last, self.keep_interval)
        for step in sorted(committed - steps_to_keep):
            self._remove_remote(step)
            try_rmtree(get_step_path(self.ckpt_dir, step))

    async def _get(self, key: str, path: Path, objects: list[dict]) -> None:
        in_flight = asyncio.Semaphore(DOWNLOAD_PARTS_IN_FLIGHT)

        async def get_part(object_key: str, start: int, end: int, fd: int) -> None:
            async with in_flight:
                part = await obstore.get_range_async(self.remote, object_key, start=start, end=end)
            os.pwrite(fd, memoryview(part), start)

        async def get_object(meta: dict) -> None:
            file = path / meta["path"].removeprefix(f"{key}/")
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

    async def _put(self, path: Path, step: int, files: list[Path]) -> None:
        await asyncio.gather(
            *[
                obstore.put_async(
                    self.remote,
                    f"{_get_key(step)}/{file.relative_to(path).as_posix()}",
                    file,
                    chunk_size=TRANSFER_PART_BYTES,
                    max_concurrency=UPLOAD_PARTS_PER_FILE,
                )
                for file in files
            ]
        )

    def _remove_remote(self, step: int) -> None:
        """``.metadata`` goes first, so a partly removed step never reads as complete. Logs and skips on failure:
        what is left is removed again by the next start."""
        metadata_key = f"{_get_key(step)}/{METADATA_FILE}"
        try:
            keys = [meta["path"] for meta in obstore.list(self.remote, prefix=f"step_{step}/").collect()]
            if metadata_key in keys:
                obstore.delete(self.remote, metadata_key)
            obstore.delete(self.remote, [key for key in keys if key != metadata_key])
        except BaseError as error:
            self.logger.warning(f"Failed to remove step {step} from the remote: {error}, skipping cleanup")

    def _wait_for_uploads(self, path: Path, step: int) -> None:
        """Wait until every rank's files on the shared disk are in the remote: the other nodes upload theirs."""
        key = _get_key(step)
        expected = {f"{key}/{file.relative_to(path).as_posix()}" for file in _get_rank_files(path, rank="*")}
        while not expected <= {meta["path"] for meta in obstore.list(self.remote, prefix=key).collect()}:
            time.sleep(UPLOADS_POLL_SECONDS)


def get_remote_steps(remote: ObjectStore) -> tuple[set[int], set[int]]:
    """Every step in the remote, and the committed ones: those whose ``.metadata`` was uploaded."""
    keys = {meta["path"] for meta in obstore.list(remote).collect()}
    steps = {int(key.split("/")[0].removeprefix("step_")) for key in keys}
    return steps, {step for step in steps if f"{_get_key(step)}/{METADATA_FILE}" in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("remote_dir")
    parser.add_argument("ckpt_dir", type=Path)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--ranks-per-node", type=int, required=True)
    parser.add_argument("--keep-last", type=int)
    parser.add_argument("--keep-interval", type=int)
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
        keep_last=args.keep_last,
        keep_interval=args.keep_interval,
    )
    command_to_handler = {"commit": sync.commit, "pull": sync.pull, "upload": sync.upload}

    commands: queue.Queue[str] = queue.Queue()
    threading.Thread(target=_read_commands, args=(commands,), daemon=True).start()

    if args.node_rank == 0:
        sync.remove_uncommitted()
    while (line := commands.get()) != "exit":
        command, step = line.split()
        answer = command_to_handler[command](int(step))
        if answer is not None:
            print(answer, file=answers, flush=True)


def _get_key(step: int) -> str:
    return f"step_{step}/trainer"


def _get_rank_files(path: Path, rank: int | str) -> list[Path]:
    return [*path.glob(f"__{rank}_*.distcp"), *path.glob(f"dataloader/rank_{rank}.pt")]


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
