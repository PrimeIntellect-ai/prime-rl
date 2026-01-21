from pathlib import Path
from typing import Generator

import pytest
import tomli_w
import torch
import torch.distributed as dist

import prime_rl.trainer.runs as runs
from prime_rl.trainer.rl.packer import MultiPacker
from prime_rl.trainer.runs import setup_multi_run_manager
from prime_rl.trainer.world import reset_world
from prime_rl.transport.config import FileSystemTransportConfig
from prime_rl.transport.types import TrainingSample


@pytest.fixture(autouse=True, scope="module")
def init_process_group() -> Generator[None, None, None]:
    dist.init_process_group(backend="gloo", init_method="tcp://localhost:12356", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def create_run_with_config(output_dir: Path, run_name: str) -> Path:
    run_dir = output_dir / run_name
    run_dir.mkdir()
    config_dir = run_dir / "configs"
    config_dir.mkdir()
    config = {
        "model": {"name": "test-model"},
        "batch_size": 2,
        "rollouts_per_example": 1,
        "env": [{"id": "test-env"}],
    }
    with open(config_dir / "orch.toml", "wb") as f:
        tomli_w.dump(config, f)
    return run_dir


def make_training_sample() -> TrainingSample:
    return TrainingSample(
        prompt_ids=[1],
        prompt_mask=[False],
        completion_ids=[2],
        completion_mask=[True],
        completion_logprobs=[-0.1],
    )


def test_packer_progress_updates_once_per_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reset_world()
    runs._MULTI_RUN_MANAGER = None
    manager = setup_multi_run_manager(output_dir=tmp_path, max_runs=1, device=torch.device("cpu"))

    create_run_with_config(tmp_path, "run_test123")
    manager.discover_runs()
    run_idx = manager.id_2_idx["run_test123"]

    class DummyReceiver:
        def receive(self):
            return []

        def reset_run(self, idx: int) -> None:
            pass

    class DummySender:
        def __init__(self):
            self.sent = []

        def send(self, micro_batch_grid):
            self.sent.append(micro_batch_grid)

    sender_holder: dict[str, DummySender] = {}

    def fake_receiver(_config):
        return DummyReceiver()

    def fake_sender(_output_dir, _data_world_size, _current_step, _config):
        sender = DummySender()
        sender_holder["sender"] = sender
        return sender

    monkeypatch.setattr("prime_rl.trainer.rl.packer.setup_training_batch_receiver", fake_receiver)
    monkeypatch.setattr("prime_rl.trainer.rl.packer.setup_micro_batch_sender", fake_sender)

    packer = MultiPacker(
        dp_world_size=1,
        seq_len=4,
        pad_to_multiple_of=1,
        tokenizer=None,
        config=FileSystemTransportConfig(),
        start_step=0,
    )

    packer.buffers[run_idx].append((make_training_sample(), 0))
    packer.buffers[run_idx].append((make_training_sample(), 0))

    packer.pack()

    progress = manager.progress[run_idx]
    assert progress.total_samples == 2
    assert progress.total_tokens == 4
    assert progress.step == 1

    sender = sender_holder["sender"]
    assert len(sender.sent) == 1
    assert len(sender.sent[0][0]) == 1
