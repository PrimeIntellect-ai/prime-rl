import socket
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from obstore.store import LocalStore
from torch import nn

from prime_rl.configs.shared import ResumeConfig
from prime_rl.configs.trainer import CheckpointConfig, RemoteCheckpointConfig
from prime_rl.entrypoints.rl import validate_remote_resume
from prime_rl.trainer.ckpt import CheckpointManager, ModelFiles, Progress
from prime_rl.trainer.ckpt_sync import get_remote_steps
from prime_rl.trainer.utils import get_dcp_group


@pytest.fixture(autouse=True)
def single_rank_group():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    yield
    get_dcp_group.cache_clear()
    dist.destroy_process_group()


@pytest.fixture
def remote_dir(tmp_path: Path) -> Path:
    (tmp_path / "remote").mkdir()
    return tmp_path / "remote"


@pytest.fixture
def config(remote_dir: Path) -> CheckpointConfig:
    return CheckpointConfig(remote=RemoteCheckpointConfig(dir=f"file://{remote_dir}"))


def make_manager(output_dir: Path, config: CheckpointConfig, resume: ResumeConfig | None = None) -> CheckpointManager:
    return CheckpointManager(output_dir, config, Mock(spec=ModelFiles, config=None), resume)


def make_trained(seed: int) -> tuple[nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(seed)
    model = nn.Linear(8, 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model(torch.randn(4, 8)).sum().backward()
    optimizer.step()
    return model, optimizer


def save(manager: CheckpointManager, *steps: int, clean: bool = False) -> None:
    model, optimizer = make_trained(seed=0)
    for step in steps:
        manager.save(step, model, [optimizer], Progress(step=step))
        manager.wait()
        if clean:
            manager.maybe_clean()
    manager.close()


def test_resumes_from_remote_when_disk_is_empty(tmp_path: Path, config: CheckpointConfig) -> None:
    model, optimizer = make_trained(seed=0)
    save(make_manager(tmp_path / "run", config), 3)
    manager = make_manager(tmp_path / "fresh", config, ResumeConfig())
    resumed_model, resumed_optimizer = make_trained(seed=1)

    step = manager.resolve_latest_step(None)
    manager.prefetch(step)
    manager.load(step, resumed_model, [resumed_optimizer], Progress())
    manager.close()

    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), resumed_model.state_dict().values()))


def test_save_raises_once_the_sync_process_died(tmp_path: Path, config: CheckpointConfig) -> None:
    manager = make_manager(tmp_path / "run", config)
    manager.sync.kill()
    manager.sync.wait()

    with pytest.raises(RuntimeError, match="sync process exited"):
        save(manager, 3)


def test_fresh_run_refuses_a_remote_with_steps(tmp_path: Path, config: CheckpointConfig) -> None:
    save(make_manager(tmp_path / "run", config), 3)

    with pytest.raises(FileExistsError, match="--resume"):
        make_manager(tmp_path / "other", config)


def test_disk_and_remote_keep_their_own_last_steps(tmp_path: Path, config: CheckpointConfig, remote_dir: Path) -> None:
    # Step 10 was saved before the remote was turned on, so it is only on disk
    save(make_manager(tmp_path / "run", CheckpointConfig()), 10)
    config.keep_last, config.remote.keep_last = 1, 2

    save(make_manager(tmp_path / "run", config, ResumeConfig()), 20, 30, 40, clean=True)

    disk_steps = sorted(path.name for path in (tmp_path / "run/checkpoints").iterdir())
    assert (disk_steps, get_remote_steps(LocalStore(remote_dir))[1]) == (["step_40"], {30, 40})


def test_remote_steps_past_the_resume_step_take_no_keep_last_slot(
    tmp_path: Path, config: CheckpointConfig, remote_dir: Path
) -> None:
    save(make_manager(tmp_path / "run", config), 3, 5)
    config.remote.keep_last = 1

    save(make_manager(tmp_path / "run", config, ResumeConfig(step=3)), 4, clean=True)

    assert get_remote_steps(LocalStore(remote_dir))[1] == {4, 5}


def test_resume_takes_the_later_of_the_disk_step_and_the_remote(tmp_path: Path, config: CheckpointConfig) -> None:
    save(make_manager(tmp_path / "run", config), 3)
    manager = make_manager(tmp_path / "run", config, ResumeConfig())

    steps = (manager.resolve_latest_step(None), manager.resolve_latest_step(2), manager.resolve_latest_step(8))
    manager.close()

    assert steps == (3, 3, 8)


@pytest.mark.parametrize(
    ("disk_step", "expectation"),
    [(None, pytest.raises(ValueError)), (2, pytest.raises(ValueError)), (3, nullcontext())],
)
def test_rl_resume_fails_fast_when_remote_is_ahead_of_disk(
    tmp_path: Path, config: CheckpointConfig, disk_step: int | None, expectation
) -> None:
    save(make_manager(tmp_path / "run", config), 3)

    with expectation:
        validate_remote_resume(config, disk_step)
