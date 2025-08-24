from pathlib import Path
from typing import Callable

import pytest
import torch

from prime_rl.orchestrator.batch import BatchSample
from prime_rl.trainer.rl.data import MicroBatch
from prime_rl.utils.utils import get_rollout_dir
from tests import Command, Environment, ProcessResult

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

ENV = {"CUDA_VISIBLE_DEVICES": "1"}
CMD = ["uv", "run", "trainer", "@", "configs/debug/train.toml"]


def create_sample(seq_len: int) -> BatchSample:
    return {
        "input_ids": torch.randint(0, 100, (seq_len,)).long(),
        "position_ids": torch.zeros(seq_len).long(),
        "advantages": torch.randn(seq_len).float(),
        "loss_mask": torch.ones(seq_len).bool(),
        "logprobs": torch.randn(seq_len).float(),
        "total_tokens": seq_len,
    }


def create_dummy_batch(batch_size: int, seq_len: int) -> MicroBatch:
    micro_batch = {}
    samples = [create_sample(seq_len) for _ in range(batch_size)]
    for key in ["input_ids", "advantages", "loss_mask", "logprobs", "position_ids"]:
        micro_batch[key] = torch.cat([sample[key] for sample in samples]).unsqueeze(0)
    micro_batch["temperature"] = 1.0
    micro_batch["total_tokens"] = batch_size * seq_len
    return micro_batch


@pytest.fixture(scope="module")
def fake_rollout_dir(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[list[int], int, int, int], Path]:
    """Create a temporary directory with dummy batches."""
    outputs_dir = tmp_path_factory.mktemp("outputs")

    def write_dummy_batches(
        steps: list[int] = [1],
        batch_size: int = 1,
        micro_batch_size: int = 1,
        seq_len: int = 10,
    ) -> Path:
        for step in steps:
            step_path = get_rollout_dir(outputs_dir) / f"step_{step}"
            step_path.mkdir(parents=True, exist_ok=True)
            batch_path = step_path / "rank_0.pt"
            tmp_path = batch_path.with_suffix(".tmp")
            batches = []
            assert batch_size % micro_batch_size == 0, "Batch size must be divisible by micro batch size"
            for _ in range(batch_size // micro_batch_size):
                micro_batch = create_dummy_batch(micro_batch_size, seq_len)
                batches.append(micro_batch)
            torch.save(batches, tmp_path)
            tmp_path.rename(batch_path)

        return outputs_dir

    return write_dummy_batches


@pytest.fixture(scope="module")
def train_process(
    run_process: Callable[[Command, Environment], ProcessResult],
    fake_rollout_dir: Callable[[list[int], int, int, int], Path],
):
    outputs_dir = fake_rollout_dir(list(range(5)), 16, 8, 16)
    return run_process(
        CMD + ["--outputs-dir", outputs_dir.as_posix(), "--data.fake", "None", "--log.level", "debug"], ENV
    )


def test_no_error(train_process: ProcessResult):
    assert train_process.returncode == 0, f"Train process failed with return code {train_process.returncode}"
