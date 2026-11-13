import subprocess
import sys
from pathlib import Path

import obstore
import pytest
from obstore.store import LocalStore

from prime_rl.trainer.ckpt_sync import CheckpointSync, get_remote_steps
from prime_rl.utils.pathing import get_step_path

# Two ranks' DCP shards and dataloader states, the mapping, and the .metadata DCP writes last
CHECKPOINT_FILES = {
    "__0_0.distcp": b"rank 0 shard",
    "__1_0.distcp": b"rank 1 shard",
    "dataloader/rank_0.pt": b"rank 0 dataloader",
    "dataloader/rank_1.pt": b"rank 1 dataloader",
    "mapping.json": b"{}",
    ".metadata": b"metadata",
}


@pytest.fixture
def remote(tmp_path: Path) -> LocalStore:
    (tmp_path / "remote").mkdir()
    return LocalStore(tmp_path / "remote")


def make_sync(remote: LocalStore, ckpt_dir: Path, node_rank: int = 0, num_nodes: int = 1, **keep) -> CheckpointSync:
    return CheckpointSync(
        remote=remote,
        ckpt_dir=ckpt_dir,
        node_rank=node_rank,
        num_nodes=num_nodes,
        ranks_per_node=2 // num_nodes,
        keep_last=keep.get("keep_last"),
        keep_interval=keep.get("keep_interval"),
    )


def read_step(ckpt_dir: Path, step: int) -> dict[str, bytes]:
    path = get_step_path(ckpt_dir, step) / "trainer"
    return {file.relative_to(path).as_posix(): file.read_bytes() for file in path.rglob("*") if file.is_file()}


def write_step(ckpt_dir: Path, step: int) -> None:
    for name, data in CHECKPOINT_FILES.items():
        file = get_step_path(ckpt_dir, step) / "trainer" / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)


def test_pull_on_every_node_then_commit_restores_the_step(tmp_path: Path, remote: LocalStore) -> None:
    write_step(tmp_path / "run", 3)
    make_sync(remote, tmp_path / "run").upload(3)
    nodes = [make_sync(remote, tmp_path / "fresh", node_rank=n, num_nodes=2) for n in range(2)]

    answers = [node.pull(3) for node in nodes]
    pulled = read_step(tmp_path / "fresh", 3)
    answers.append(nodes[0].commit(3))

    assert (answers, ".metadata" in pulled, read_step(tmp_path / "fresh", 3)) == (
        ["pulled 3", "pulled 3", "committed 3"],
        False,
        CHECKPOINT_FILES,
    )


def test_step_commits_only_once_every_node_uploaded(tmp_path: Path, remote: LocalStore) -> None:
    write_step(tmp_path / "run", 3)
    nodes = [make_sync(remote, tmp_path / "run", node_rank=n, num_nodes=2) for n in range(2)]

    nodes[1].upload(3)
    committed_after_node_1 = get_remote_steps(remote)[1]
    nodes[0].upload(3)

    assert (committed_after_node_1, get_remote_steps(remote)[1]) == (set(), {3})


def test_upload_applies_keep_last_to_remote_and_disk(tmp_path: Path, remote: LocalStore) -> None:
    sync = make_sync(remote, tmp_path / "run", keep_last=1)
    for step in (1, 2):
        write_step(tmp_path / "run", step)

    for step in (1, 2):
        sync.upload(step)

    assert (get_remote_steps(remote), sorted(p.name for p in (tmp_path / "run").iterdir())) == (({2}, {2}), ["step_2"])


def test_remove_uncommitted_drops_steps_without_metadata(tmp_path: Path, remote: LocalStore) -> None:
    write_step(tmp_path / "run", 3)
    make_sync(remote, tmp_path / "run").upload(3)
    obstore.delete(remote, "step_3/trainer/.metadata")

    make_sync(remote, tmp_path / "run").remove_uncommitted()

    assert obstore.list(remote).collect() == []


@pytest.mark.parametrize(
    ("commands", "expected_stdout", "expected_exit_code"),
    [
        ("upload 3\npull 3\nexit\n", "pulled 3\n", 0),
        # step 4 is not on disk, so its .metadata cannot be uploaded
        ("upload 4\nexit\n", "", 1),
        # the trainer died: stdin closed before exit
        ("upload 3\n", "", 1),
    ],
)
def test_process_answers_commands_and_exits(
    tmp_path: Path, remote: LocalStore, commands: str, expected_stdout: str, expected_exit_code: int
) -> None:
    write_step(tmp_path / "run", 3)
    args = [sys.executable, "-m", "prime_rl.trainer.ckpt_sync", f"file://{tmp_path}/remote", str(tmp_path / "run")]

    result = subprocess.run(
        [*args, "--node-rank=0", "--num-nodes=1", "--ranks-per-node=2"],
        input=commands,
        capture_output=True,
        text=True,
    )

    assert (result.stdout, result.returncode) == (expected_stdout, expected_exit_code), result.stderr
