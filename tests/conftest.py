import os
import subprocess
from pathlib import Path
from typing import Callable

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch.distributed as dist
from huggingface_hub import HfApi
from pyarrow import Table

from zeroband.training.data import STABLE_FILE
from zeroband.utils.logger import reset_logger
from zeroband.utils.models import AttnImpl, ModelName
from zeroband.utils.parquet import pa_schema
from zeroband.utils.world_info import reset_world_info


@pytest.fixture(autouse=True)
def global_setup_and_cleanup():
    """
    Fixture to reset environment variables and singletons after each test.
    """
    original_env = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(original_env)
    reset_world_info()
    reset_logger("TRAIN")
    reset_logger("INFER")


@pytest.fixture(params=["eager", "sdpa", "flash_attention_2"])
def attn_impl(request) -> AttnImpl:
    """
    Fixture to test different attention implementations.
    """
    try:
        # ruff: noqa: F401
        import flash_attn
    except ImportError:
        pytest.skip("Flash Attention not available")
    return request.param


@pytest.fixture(scope="session")
def model_name() -> ModelName:
    """Main model to use for tests."""
    return "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="session")
def hf_api() -> HfApi:
    """Hugging Face API to use for tests."""
    return HfApi()


@pytest.fixture(scope="session")
def llm(model_name: ModelName) -> "LLM":
    """
    vLLM LLM instance to use for tests. Incurs significant startup time, hence reused across tests.
    """
    from vllm import LLM

    yield LLM(model=model_name, enforce_eager=True, disable_async_output_proc=True, dtype="bfloat16")

    if dist.is_initialized():
        dist.destroy_process_group()


def create_dummy_parquet_table(batch_size: int, seq_len: int) -> Table:
    """
    Create a dummy parquet table with the inference schema.

    Args:
        batch_size: Number of samples in the batch
        seq_len: Length of the sequence

    Returns:
        PyArrow table with the inference schema
    """
    # Create data dictionary with typed arrays
    data = {
        "input_tokens": pa.array([[1] * seq_len for _ in range(batch_size)], type=pa.list_(pa.int32())),
        "output_tokens": pa.array([[1] * seq_len for _ in range(batch_size)], type=pa.list_(pa.int32())),
        "advantages": pa.array([1] * batch_size, type=pa.float32()),
        "rewards": pa.array([1] * batch_size, type=pa.float32()),
        "task_rewards": pa.array([0] * batch_size, type=pa.float32()),
        "length_penalties": pa.array([0] * batch_size, type=pa.float32()),
        "proofs": pa.array([b"I am toploc proof, handcrafted by jack"] * batch_size, type=pa.binary()),
        "step": pa.array([0] * batch_size, type=pa.int32()),
        "target_lengths": pa.array([seq_len] * batch_size, type=pa.int32()),
    }

    # Create table directly from dictionary
    return Table.from_pydict(data, schema=pa_schema)


@pytest.fixture(scope="module")
def fake_rollout_files_dir(tmp_path_factory: pytest.TempPathFactory) -> Callable[[list[int], int, int, int], Path]:
    """
    Create a temporary directory with dummy parquet files with inference output for testing

    Args:
        tmp_path: Automatically created temporary path by pytest

    Returns:
        A function that can be called to write dummy parquet files to the temporary directory
    """
    path = tmp_path_factory.mktemp("fake_rollout_files")

    def write_dummy_parquet_files(steps: list[int] = [0], num_files: int = 1, batch_size: int = 1, seq_len: int = 10) -> Path:
        for step in steps:
            step_path = path / f"step_{step}"
            os.makedirs(step_path, exist_ok=True)
            for file_idx in range(num_files):
                table = create_dummy_parquet_table(batch_size, seq_len)
                pq.write_table(table, f"{step_path}/{file_idx}.parquet")

            stable_file = step_path / STABLE_FILE
            stable_file.touch()

        return path

    return write_dummy_parquet_files
