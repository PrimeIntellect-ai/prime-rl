from pathlib import Path
from typing import Generator

import pytest
import tomli_w
import torch
import torch.distributed as dist

import prime_rl.trainer.runs as runs
from prime_rl.trainer.rl.packer import MultiPacker, SinglePacker
from prime_rl.trainer.runs import setup_multi_run_manager
from prime_rl.trainer.world import reset_world
from prime_rl.transport.config import FileSystemTransportConfig
from prime_rl.transport.types import TrainingSample


@pytest.fixture(autouse=True, scope="module")
def init_process_group() -> Generator[None, None, None]:
    dist.init_process_group(backend="gloo", init_method="tcp://localhost:12356", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


class DummyReceiver:
    def __init__(self):
        self.start_steps: list[tuple[int, int]] = []

    def receive(self):
        return []

    def reset_run(self, idx: int) -> None:
        pass

    def set_start_step(self, idx: int, step: int) -> None:
        self.start_steps.append((idx, step))


class DummySender:
    def __init__(self):
        self.sent: list = []

    def send(self, micro_batch_grid):
        self.sent.append(micro_batch_grid)


def create_run_with_config(output_dir: Path, run_name: str, config: dict | None = None) -> Path:
    run_dir = output_dir / run_name
    run_dir.mkdir()
    control_dir = run_dir / "control"
    control_dir.mkdir()
    if config is None:
        config = {
            "model": {"name": "test-model"},
            "max_inflight_rollouts": 2,
            "rollouts_per_example": 1,
            "env": [{"id": "test-env"}],
            "sampling": {"temperature": 1.0},
        }
    with open(control_dir / "orch.toml", "wb") as f:
        tomli_w.dump(config, f)
    return run_dir


def make_training_sample() -> TrainingSample:
    return TrainingSample(
        prompt_ids=[1],
        prompt_mask=[False],
        completion_ids=[2],
        completion_mask=[True],
        completion_logprobs=[-0.1],
        completion_temperatures=[1.0],
    )


@pytest.fixture
def packer_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Set up a single-run manager with dummy transport for packer tests."""
    reset_world()
    runs._MULTI_RUN_MANAGER = None
    manager = setup_multi_run_manager(output_dir=tmp_path, max_runs=1, device=torch.device("cpu"))
    create_run_with_config(tmp_path, "run_test123")
    manager.discover_runs()
    run_idx = manager.id_2_idx["run_test123"]

    receiver = DummyReceiver()
    sender = DummySender()

    monkeypatch.setattr("prime_rl.trainer.rl.packer.setup_training_batch_receiver", lambda _config: receiver)
    monkeypatch.setattr(
        "prime_rl.trainer.rl.packer.setup_micro_batch_sender",
        lambda _output_dir, _data_world_size, _current_step, _config: sender,
    )

    return manager, run_idx, receiver, sender


@pytest.fixture
def multi_packer_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Set up a multi-run manager with dummy transport for packer tests."""
    reset_world()
    runs._MULTI_RUN_MANAGER = None
    manager = setup_multi_run_manager(output_dir=tmp_path, max_runs=2, device=torch.device("cpu"))
    create_run_with_config(tmp_path, "run_a")
    manager.discover_runs()
    run_idx = manager.id_2_idx["run_a"]

    receiver = DummyReceiver()
    sender = DummySender()

    monkeypatch.setattr("prime_rl.trainer.rl.packer.setup_training_batch_receiver", lambda _config: receiver)
    monkeypatch.setattr(
        "prime_rl.trainer.rl.packer.setup_micro_batch_sender",
        lambda _output_dir, _data_world_size, _current_step, _config: sender,
    )

    return manager, run_idx, receiver, sender


def test_packer_progress_updates_once_per_run(packer_env) -> None:
    manager, run_idx, receiver, sender = packer_env

    packer = MultiPacker(
        dp_world_size=1,
        seq_len=4,
        pad_to_multiple_of=1,
        tokenizer=None,
        config=FileSystemTransportConfig(),
        token_batch_size=4,
        start_step=0,
    )

    packer.buffers[run_idx].append((make_training_sample(), 0))
    packer.buffers[run_idx].append((make_training_sample(), 0))
    packer.pack()

    progress = manager.progress[run_idx]
    assert progress.total_samples == 2
    assert progress.total_tokens == 4
    assert progress.step == 1

    assert len(sender.sent) == 1
    assert len(sender.sent[0][0]) == 1


def test_multi_packer_token_budget_threshold(packer_env) -> None:
    manager, run_idx, receiver, sender = packer_env

    packer = MultiPacker(
        dp_world_size=1,
        seq_len=8,
        pad_to_multiple_of=1,
        tokenizer=None,
        config=FileSystemTransportConfig(),
        token_batch_size=5,
        start_step=0,
    )

    packer.buffers[run_idx].append((make_training_sample(), 0))  # 2 tokens
    packer.buffers[run_idx].append((make_training_sample(), 0))  # 2 tokens
    assert not packer._has_enough_batch_units()
    packer.buffers[run_idx].append((make_training_sample(), 0))  # +2 = 6 tokens
    assert packer._has_enough_batch_units()


def test_multi_packer_keeps_first_oversized_sample(packer_env) -> None:
    manager, run_idx, receiver, sender = packer_env

    packer = MultiPacker(
        dp_world_size=1,
        seq_len=16,
        pad_to_multiple_of=1,
        tokenizer=None,
        config=FileSystemTransportConfig(),
        token_batch_size=4,
        start_step=0,
    )

    oversized = TrainingSample(
        prompt_ids=[1, 2, 3],
        prompt_mask=[False, False, False],
        completion_ids=[4, 5],
        completion_mask=[True, True],
        completion_logprobs=[-0.1, -0.2],
        completion_temperatures=[1.0, 1.0],
    )
    packer.buffers[run_idx].append((oversized, 0))

    selected = packer._select_samples_round_robin()
    assert len(selected) == 1
    assert selected[0][0] == run_idx
    assert selected[0][1] is oversized


def test_multi_packer_rollout_budget_threshold(packer_env) -> None:
    manager, run_idx, receiver, sender = packer_env

    packer = MultiPacker(
        dp_world_size=1,
        seq_len=8,
        pad_to_multiple_of=1,
        tokenizer=None,
        config=FileSystemTransportConfig(),
        rollout_batch_size=3,
        start_step=0,
    )

    packer.buffers[run_idx].append((make_training_sample(), 0))
    packer.buffers[run_idx].append((make_training_sample(), 0))
    assert not packer._has_enough_batch_units()
    packer.buffers[run_idx].append((make_training_sample(), 0))
    assert packer._has_enough_batch_units()


def test_single_packer_sets_receiver_start_step_to_zero(packer_env) -> None:
    manager, run_idx, receiver, sender = packer_env

    SinglePacker(
        dp_world_size=1,
        seq_len=8,
        pad_to_multiple_of=1,
        tokenizer=None,
        config=FileSystemTransportConfig(),
        token_batch_size=4,
        start_step=0,
    )

    assert receiver.start_steps == [(0, 0)]


def test_multi_packer_sets_receiver_start_step_to_zero_for_existing_and_new_runs(
    multi_packer_env, tmp_path: Path
) -> None:
    manager, run_idx, receiver, sender = multi_packer_env

    MultiPacker(
        dp_world_size=1,
        seq_len=8,
        pad_to_multiple_of=1,
        tokenizer=None,
        config=FileSystemTransportConfig(),
        token_batch_size=4,
        start_step=0,
    )

    assert (run_idx, 0) in receiver.start_steps
    existing_steps = list(receiver.start_steps)

    create_run_with_config(
        tmp_path,
        "run_b",
        config={
            "model": {"name": "test-model"},
            "token_batch_size": 4,
            "max_inflight_rollouts": 2,
            "rollouts_per_example": 1,
            "env": [{"id": "test-env"}],
            "sampling": {"temperature": 1.0},
        },
    )
    manager.discover_runs()

    new_run_idx = manager.id_2_idx["run_b"]
    assert (new_run_idx, 0) in receiver.start_steps
    assert len(receiver.start_steps) == len(existing_steps) + 1
