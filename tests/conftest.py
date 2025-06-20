import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch.distributed as dist
from huggingface_hub import HfApi
from pyarrow import Table

TIMEOUT = 120


Environment = dict[str, str]
Command = list[str]


from loguru import logger

from zeroband.training.data import STABLE_FILE
from zeroband.training.world_info import reset_world_info
from zeroband.utils.logger import reset_logger, set_logger
from zeroband.utils.models import AttnImpl
from zeroband.utils.parquet import pa_schema


@pytest.fixture(autouse=True)
def setup_logger():
    """
    Fixture to set and reset the logger after each test.
    """
    set_logger(logger)  # Use the default loguru.logger
    yield
    reset_logger()


@pytest.fixture(autouse=True)
def setup_env():
    """
    Fixture to reset environment variables after each test.
    """
    original_env = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def setup_world_info():
    """
    Fixture to reset the world info after each test.
    """
    yield
    reset_world_info()


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
def model_name() -> str:
    """Main model to use for tests."""
    return "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="session")
def hf_api() -> HfApi:
    """Hugging Face API to use for tests."""
    return HfApi()


@pytest.fixture(scope="module")
def llm(model_name: str) -> "LLM":
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
        "input_logprobs": pa.array([[0.1] * seq_len for _ in range(batch_size)], type=pa.list_(pa.float32())),
        "output_logprobs": pa.array([[0.1] * seq_len for _ in range(batch_size)], type=pa.list_(pa.float32())),
        "input_mask": pa.array([[True] * seq_len for _ in range(batch_size)], type=pa.list_(pa.bool_())),
        "output_mask": pa.array([[True] * seq_len for _ in range(batch_size)], type=pa.list_(pa.bool_())),
        "prompt": pa.array(["prompt" for _ in range(batch_size)], type=pa.string()),
        "completion": pa.array(["completion" for _ in range(batch_size)], type=pa.string()),
        "advantages": pa.array([1] * batch_size, type=pa.float32()),
        "rewards": pa.array([1] * batch_size, type=pa.float32()),
        "task_rewards": pa.array([0] * batch_size, type=pa.float32()),
        "length_penalties": pa.array([0] * batch_size, type=pa.float32()),
        "proofs": pa.array([b"I am toploc proof, handcrafted by jack"] * batch_size, type=pa.binary()),
        "step": pa.array([0] * batch_size, type=pa.int32()),
        "target_lengths": pa.array([seq_len] * batch_size, type=pa.int32()),
        "task_type": pa.array(["test_task"] * batch_size, type=pa.string()),
        "problem_id": pa.array(["0"] * batch_size, type=pa.string()),
        "seed": pa.array([42] * batch_size, type=pa.int64()),
        "temperature": pa.array([1.0] * batch_size, type=pa.float32()),
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


class ProcessResult:
    def __init__(self, returncode: int, pid: int):
        self.returncode = returncode
        self.pid = pid


def run_subprocess(command: Command, env: Environment, timeout: int = TIMEOUT) -> ProcessResult:
    """Run a subprocess with given command and environment with a timeout"""
    try:
        process = subprocess.Popen(command, env={**os.environ, **env})
        process.wait(timeout=timeout)
        return ProcessResult(process.returncode, process.pid)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)  # Give it 10 seconds to terminate gracefully
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    except Exception as e:
        raise e


def run_subprocesses_in_parallel(commands: list[Command], envs: list[Environment], timeout: int = TIMEOUT) -> list[ProcessResult]:
    """Start multiple processes in parallel using ProcessPoolExecutor and wait for completion."""
    assert len(commands) == len(envs), "Should have an environment for each command"
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(commands)) as executor:
        futures = [executor.submit(run_subprocess, cmd, env, timeout) for cmd, env in zip(commands, envs)]
        results = []
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Process {i} did not complete within {timeout} seconds")

    return results


@pytest.fixture(scope="module")
def run_process() -> Callable[[Command, Environment], ProcessResult]:
    """Factory fixture for running a single process."""
    return run_subprocess


@pytest.fixture(scope="module")
def run_processes() -> Callable[[list[Command], list[Environment]], list[ProcessResult]]:
    """Factory fixture for running multiple processes in parallel. Used for parallel inference tests and RL training tests."""
    return run_subprocesses_in_parallel
