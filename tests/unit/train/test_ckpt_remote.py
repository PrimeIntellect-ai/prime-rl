import socket
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from torch import nn

from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.entrypoints.rl import validate_remote_resume
from prime_rl.trainer.ckpt import CheckpointManager, ModelFiles, Progress
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
    manager.save(3, model, [optimizer], Progress(step=3))
    manager.close()


def test_resumes_from_remote_when_disk_is_empty(tmp_path: Path, config: CheckpointConfig) -> None:
    model, optimizer = make_trained(seed=0)
    save(CheckpointManager(tmp_path / "run", config, Mock(spec=ModelFiles)), model, optimizer)
    manager = CheckpointManager(tmp_path / "fresh", config, Mock(spec=ModelFiles))
    resumed_model, resumed_optimizer = make_trained(seed=1)

    step = manager.resolve_latest_step(None)
    manager.prefetch(step)
    manager.load(step, resumed_model, [resumed_optimizer], Progress())
    manager.close()

    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), resumed_model.state_dict().values()))


def test_save_raises_once_the_sync_process_died(tmp_path: Path, config: CheckpointConfig) -> None:
    model, optimizer = make_trained(seed=0)
    manager = CheckpointManager(tmp_path / "run", config, Mock(spec=ModelFiles))
    manager.sync.kill()
    manager.sync.wait()

    with pytest.raises(RuntimeError, match="sync process exited"):
        save(manager, model, optimizer)


def test_resume_takes_the_later_of_the_disk_step_and_the_remote(tmp_path: Path, config: CheckpointConfig) -> None:
    model, optimizer = make_trained(seed=0)
    save(CheckpointManager(tmp_path / "run", config, Mock(spec=ModelFiles)), model, optimizer)
    manager = CheckpointManager(tmp_path / "run", config, Mock(spec=ModelFiles))

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
    model, optimizer = make_trained(seed=0)
    save(CheckpointManager(tmp_path / "run", config, Mock(spec=ModelFiles)), model, optimizer)

    with expectation:
        validate_remote_resume(config, disk_step)
