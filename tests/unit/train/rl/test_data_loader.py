from pathlib import Path
from typing import Generator

import pytest
import tomli_w
import torch
import torch.distributed as dist

import prime_rl.trainer.runs as runs
from prime_rl.trainer.rl.data import DataLoader
from prime_rl.trainer.runs import setup_multi_run_manager
from prime_rl.trainer.world import reset_world
from prime_rl.transport.config import FileSystemTransportConfig


@pytest.fixture(autouse=True, scope="module")
def init_process_group() -> Generator[None, None, None]:
    dist.init_process_group(backend="gloo", init_method="tcp://localhost:12358", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


class DummyPacker:
    def pack(self) -> None:
        return


class DummyReceiver:
    def wait(self) -> None:
        return

    def receive(self):
        return []


def _write_root_orch_config(output_dir: Path, config: dict) -> None:
    control_dir = output_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    with open(control_dir / "orch.toml", "wb") as f:
        tomli_w.dump(config, f)


def _base_orch_config() -> dict:
    return {
        "model": {"name": "test-model"},
        "env": [{"id": "test-env"}],
        "sampling": {"temperature": 1.0},
    }


def _setup_single_run_manager(tmp_path: Path) -> None:
    reset_world()
    runs._MULTI_RUN_MANAGER = None
    setup_multi_run_manager(output_dir=tmp_path, max_runs=1, device=torch.device("cpu"))


def _build_dataloader(tmp_path: Path) -> DataLoader:
    return DataLoader(
        output_dir=tmp_path,
        start_step=0,
        dp_world_size=1,
        seq_len=32,
        pad_to_multiple_of=1,
        tokenizer=None,
        config=FileSystemTransportConfig(),
    )


def test_single_run_dataloader_wires_rollout_batch_target_from_control_file(tmp_path: Path, monkeypatch) -> None:
    _setup_single_run_manager(tmp_path)
    orch_config = _base_orch_config()
    orch_config["batch_size"] = 6
    orch_config["rollouts_per_example"] = 2
    _write_root_orch_config(tmp_path, orch_config)

    captured: dict = {}
    monkeypatch.setattr(
        "prime_rl.trainer.rl.data.setup_packer",
        lambda **kwargs: captured.update(kwargs) or DummyPacker(),
    )
    monkeypatch.setattr(
        "prime_rl.trainer.rl.data.setup_micro_batch_receiver", lambda *_args, **_kwargs: DummyReceiver()
    )

    _build_dataloader(tmp_path)

    assert captured["token_batch_size"] is None
    assert captured["rollout_batch_size"] == 6


def test_single_run_dataloader_wires_token_batch_target_from_control_file(tmp_path: Path, monkeypatch) -> None:
    _setup_single_run_manager(tmp_path)
    orch_config = _base_orch_config()
    orch_config["token_batch_size"] = 1024
    orch_config["max_inflight_rollouts"] = 8
    _write_root_orch_config(tmp_path, orch_config)

    captured: dict = {}
    monkeypatch.setattr(
        "prime_rl.trainer.rl.data.setup_packer",
        lambda **kwargs: captured.update(kwargs) or DummyPacker(),
    )
    monkeypatch.setattr(
        "prime_rl.trainer.rl.data.setup_micro_batch_receiver", lambda *_args, **_kwargs: DummyReceiver()
    )

    _build_dataloader(tmp_path)

    assert captured["token_batch_size"] == 1024
    assert captured["rollout_batch_size"] is None


def test_single_run_dataloader_keeps_fallback_when_control_file_missing(tmp_path: Path, monkeypatch) -> None:
    _setup_single_run_manager(tmp_path)

    captured: dict = {}
    monkeypatch.setattr(
        "prime_rl.trainer.rl.data.setup_packer",
        lambda **kwargs: captured.update(kwargs) or DummyPacker(),
    )
    monkeypatch.setattr(
        "prime_rl.trainer.rl.data.setup_micro_batch_receiver", lambda *_args, **_kwargs: DummyReceiver()
    )

    _build_dataloader(tmp_path)

    assert captured["token_batch_size"] is None
    assert captured["rollout_batch_size"] is None
