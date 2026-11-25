import pickle
import subprocess
import sys
from pathlib import Path

import obstore
import pytest
from obstore.store import LocalStore
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex

from prime_rl.trainer.ckpt_sync import CheckpointSync, get_remote_steps
from prime_rl.utils.pathing import get_step_path

# Two ranks' DCP shards and dataloader states, and the mapping. DCP writes .metadata last
CHECKPOINT_FILES = {
    "__0_0.distcp": b"rank 0 shard",
    "__1_0.distcp": b"rank 1 shard",
    "dataloader/rank_0.pt": b"rank 0 dataloader",
    "dataloader/rank_1.pt": b"rank 1 dataloader",
    "mapping.json": b"{}",
}
METADATA = pickle.dumps(
    Metadata(
        state_dict_metadata={},
        storage_data={MetadataIndex(f"w{rank}"): _StorageInfo(f"__{rank}_0.distcp", 0, 1) for rank in range(2)},
    )
)
# The ModelFiles next to the trainer checkpoint
MODEL_FILES = {"config.json": b"{}"}


@pytest.fixture
def remote(tmp_path: Path) -> LocalStore:
    (tmp_path / "remote").mkdir()
    return LocalStore(tmp_path / "remote")


def make_sync(remote: LocalStore, ckpt_dir: Path, node_rank: int = 0, num_nodes: int = 1) -> CheckpointSync:
    return CheckpointSync(remote, ckpt_dir, node_rank=node_rank, num_nodes=num_nodes, ranks_per_node=2 // num_nodes)


def read_step(ckpt_dir: Path, step: int) -> dict[str, bytes]:
    path = get_step_path(ckpt_dir, step)
    return {file.relative_to(path).as_posix(): file.read_bytes() for file in path.rglob("*") if file.is_file()}


def write_step(ckpt_dir: Path, step: int) -> None:
    files = {**{f"trainer/{name}": data for name, data in CHECKPOINT_FILES.items()}, **MODEL_FILES}
    for name, data in {**files, "trainer/.metadata": METADATA}.items():
        file = get_step_path(ckpt_dir, step) / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)


def test_pull_on_every_node_then_commit_restores_the_step(tmp_path: Path, remote: LocalStore) -> None:
    write_step(tmp_path / "run", 3)
    make_sync(remote, tmp_path / "run").upload(3)
    nodes = [make_sync(remote, tmp_path / "fresh", node_rank=n, num_nodes=2) for n in range(2)]

    answers = [node.pull(3) for node in nodes]
    pulled = read_step(tmp_path / "fresh", 3)
    answers.append(nodes[0].commit(3))

    assert (answers, "trainer/.metadata" in pulled, read_step(tmp_path / "fresh", 3)) == (
        ["pulled 3", "pulled 3", "committed 3"],
        False,
        read_step(tmp_path / "run", 3),
    )


def test_step_commits_only_once_every_node_uploaded(tmp_path: Path, remote: LocalStore) -> None:
    write_step(tmp_path / "run", 3)
    nodes = [make_sync(remote, tmp_path / "run", node_rank=n, num_nodes=2) for n in range(2)]

    nodes[1].upload(3)
    committed_after_node_1 = get_remote_steps(remote)[1]
    nodes[0].upload(3)

    assert (committed_after_node_1, get_remote_steps(remote)[1]) == (set(), {3})


def test_upload_skips_shards_its_metadata_does_not_list(tmp_path: Path, remote: LocalStore) -> None:
    # Left by a cut-off save of the step with four ranks: no node of this world uploads it, so node 0 must not wait
    write_step(tmp_path / "run", 3)
    (get_step_path(tmp_path / "run", 3) / "trainer/__3_0.distcp").write_bytes(b"stale shard")

    make_sync(remote, tmp_path / "run").upload(3)

    assert sorted(meta["path"] for meta in obstore.list(remote).collect()) == sorted(
        f"step_3/{name}" for name in read_step(tmp_path / "run", 3) if name != "trainer/__3_0.distcp"
    )


def test_delete_commands_remove_a_step_from_remote_or_disk(tmp_path: Path, remote: LocalStore) -> None:
    sync = make_sync(remote, tmp_path / "run")
    for step in (1, 2):
        write_step(tmp_path / "run", step)
        sync.upload(step)

    sync.delete_remote(1)
    sync.delete_local(2)

    assert (get_remote_steps(remote), [p.name for p in (tmp_path / "run").iterdir()]) == (({2}, {2}), ["step_1"])


def test_remote_steps_ignore_keys_outside_step_dirs(tmp_path: Path, remote: LocalStore) -> None:
    write_step(tmp_path / "run", 3)
    make_sync(remote, tmp_path / "run").upload(3)
    obstore.put(remote, "README.md", b"checkpoints")

    assert get_remote_steps(remote) == ({3}, {3})


def test_remove_uncommitted_drops_steps_without_metadata(tmp_path: Path, remote: LocalStore) -> None:
    write_step(tmp_path / "run", 3)
    make_sync(remote, tmp_path / "run").upload(3)
    obstore.delete(remote, "step_3/trainer/.metadata")

    make_sync(remote, tmp_path / "run").remove_uncommitted()

    assert obstore.list(remote).collect() == []


@pytest.mark.parametrize(
    ("commands", "expected_stdout", "expected_exit_code", "expected_steps"),
    [
        ("upload 3\npull 3\nexit\n", "pulled 3\n", 0, {3}),
        # deletions run after the upload queued before them
        ("upload 3\ndelete-remote 3\nexit\n", "", 0, set()),
        # step 4 is not on disk, so its .metadata cannot be uploaded
        ("upload 4\nexit\n", "", 1, set()),
        # the trainer died: stdin closed before exit
        ("upload 3\n", "", 1, None),
    ],
)
def test_process_answers_commands_and_exits(
    tmp_path: Path,
    remote: LocalStore,
    commands: str,
    expected_stdout: str,
    expected_exit_code: int,
    expected_steps: set[int] | None,
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
    if expected_steps is not None:
        assert get_remote_steps(remote)[1] == expected_steps
