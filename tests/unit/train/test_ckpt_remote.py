import socket
from contextlib import nullcontext
from pathlib import Path

import obstore
import pytest
import torch
import torch.distributed as dist
from obstore.store import from_url
from torch import nn

from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.entrypoints.rl import validate_remote_resume
from prime_rl.trainer.ckpt import CheckpointManager, Progress
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
def config(tmp_path: Path) -> CheckpointConfig:
    (tmp_path / "remote").mkdir()
    return CheckpointConfig(remote_dir=f"file://{tmp_path}/remote")


def make_trained(seed: int) -> tuple[nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(seed)
    model = nn.Linear(8, 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model(torch.randn(4, 8)).sum().backward()
    optimizer.step()
    return model, optimizer


def save(manager: CheckpointManager, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    manager.save(3, model, [optimizer], torch.optim.lr_scheduler.ConstantLR(optimizer), Progress(step=3))
    manager.wait()


def test_resumes_from_remote_when_disk_is_empty(tmp_path: Path, config: CheckpointConfig) -> None:
    model, optimizer = make_trained(seed=0)
    save(CheckpointManager(tmp_path / "run", config), model, optimizer)

    manager = CheckpointManager(tmp_path / "fresh", config)
    resumed_model, resumed_optimizer = make_trained(seed=1)
    manager.load(manager.resolve_latest_step(), resumed_model, [resumed_optimizer], None, Progress())

    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), resumed_model.state_dict().values()))


def test_startup_removes_uncommitted_remote_steps(tmp_path: Path, config: CheckpointConfig) -> None:
    model, optimizer = make_trained(seed=0)
    save(CheckpointManager(tmp_path / "run", config), model, optimizer)
    store = from_url(config.remote_dir)
    obstore.delete(store, "step_3/trainer/.metadata")

    manager = CheckpointManager(tmp_path / "fresh", config)

    assert (manager.resolve_latest_step(), obstore.list(store).collect()) == (None, [])


def test_failed_upload_raises_and_does_not_commit(tmp_path: Path, config: CheckpointConfig, monkeypatch) -> None:
    async def fail(store, key: str, *args, **kwargs) -> None:
        raise PermissionError(f"403 uploading {key}")

    model, optimizer = make_trained(seed=0)
    manager = CheckpointManager(tmp_path / "run", config)
    monkeypatch.setattr(obstore, "put_async", fail)

    with pytest.raises(PermissionError):
        save(manager, model, optimizer)

    assert CheckpointManager(tmp_path / "fresh", config).resolve_latest_step() is None


def test_upload_failed_on_another_rank_raises(tmp_path: Path, config: CheckpointConfig, monkeypatch) -> None:
    all_reduce = dist.all_reduce

    def another_rank_failed(tensor: torch.Tensor, *args, **kwargs) -> None:
        all_reduce(tensor, *args, **kwargs)
        tensor += 1

    model, optimizer = make_trained(seed=0)
    manager = CheckpointManager(tmp_path / "run", config)
    monkeypatch.setattr(dist, "all_reduce", another_rank_failed)

    with pytest.raises(RuntimeError, match="failed on 1 rank"):
        save(manager, model, optimizer)

    assert (manager.ckpt_steps, CheckpointManager(tmp_path / "fresh", config).resolve_latest_step()) == ([], None)


@pytest.mark.parametrize(
    ("disk_step", "expectation"),
    [(None, pytest.raises(ValueError)), (2, pytest.raises(ValueError)), (3, nullcontext())],
)
def test_rl_resume_fails_fast_when_remote_is_ahead_of_disk(
    tmp_path: Path, config: CheckpointConfig, disk_step: int | None, expectation
) -> None:
    model, optimizer = make_trained(seed=0)
    save(CheckpointManager(tmp_path / "run", config), model, optimizer)

    with expectation:
        validate_remote_resume(config, disk_step)


def test_trainer_only_resume_skips_steps_without_a_complete_trainer_save(
    tmp_path: Path, config: CheckpointConfig
) -> None:
    model, optimizer = make_trained(seed=0)
    manager = CheckpointManager(tmp_path / "run", config)
    save(manager, model, optimizer)
    (manager.ckpt_dir / "step_8/orchestrator").mkdir(parents=True)
    (manager.ckpt_dir / "step_12/trainer").mkdir(parents=True)

    manager = CheckpointManager(tmp_path / "run", config)

    assert (manager.resolve_latest_step(), manager.resolve_latest_step(trainer_only=True)) == (12, 3)
