import pyarrow as pa
import pyarrow.parquet as pq
from zeroband.utils.parquet import pa_schema
from pathlib import Path
import os
import pytest
from functools import lru_cache

from zeroband.training.data import STABLE_FILE


@lru_cache(maxsize=None)
def _create_one_pa_table(batch_size: int, seq_len: int):
    # Initialize lists for each column
    input_tokens_list = [[1] * seq_len for _ in range(batch_size)]  # Wrap in list
    output_tokens_list = [[1] * seq_len for _ in range(batch_size)]  # Wrap in list
    task_rewards_list = [0] * batch_size
    length_penalty_list = [0] * batch_size
    target_lengths_list = [seq_len] * batch_size
    advantages_list = [1] * batch_size
    rewards_list = [1] * batch_size
    proofs_list = [b"I am toploc proof, handcrafted by jack"] * batch_size
    steps_list = [0] * batch_size

    # Create PyArrow arrays
    arrays = [
        pa.array(input_tokens_list, type=pa.list_(pa.int32())),
        pa.array(output_tokens_list, type=pa.list_(pa.int32())),
        pa.array(advantages_list, type=pa.float32()),
        pa.array(rewards_list, type=pa.float32()),
        pa.array(task_rewards_list, type=pa.float32()),
        pa.array(length_penalty_list, type=pa.float32()),
        pa.array(proofs_list, type=pa.binary()),
        pa.array(steps_list, type=pa.int32()),
        pa.array(target_lengths_list, type=pa.int32()),
    ]
    # Create and return table
    return pa.Table.from_arrays(arrays, schema=pa_schema)


def _create_fake_rollout_parquet_file(path: Path, steps: list[int], num_files: int, batch_size: int, seq_len: int):
    for s in steps:
        step_path = path / f"step_{s}"

        print(f"Creating files for step {s} with batch size {batch_size} and seq len {seq_len}")

        for i in range(num_files):
            os.makedirs(step_path, exist_ok=True)
            table = _create_one_pa_table(batch_size, seq_len)
            pq.write_table(table, f"{step_path}/{i}.parquet")

        stable_file = step_path / STABLE_FILE
        stable_file.touch()


@pytest.fixture
def fake_rollout_files_dir(tmp_path):
    def _create(steps: list[int] = [0], num_files: int = 1, batch_size: int = 1, seq_len: int = 10):
        os.makedirs(tmp_path, exist_ok=True)
        _create_fake_rollout_parquet_file(tmp_path, steps, num_files, batch_size, seq_len)
        return tmp_path

    return _create
