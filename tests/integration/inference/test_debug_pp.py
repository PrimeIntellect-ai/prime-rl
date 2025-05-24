import concurrent.futures
import os
import subprocess
from pathlib import Path
from typing import Tuple

import pyarrow.parquet as pq
import pytest

from zeroband.training.data import pa_schema

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

ENV0 = {"CUDA_VISIBLE_DEVICES": "0"}
CMD0 = ["uv", "run", "src/zeroband/infer.py", "@configs/inference/pipeline/debug0.toml"]
ENV1 = {"CUDA_VISIBLE_DEVICES": "1"}
CMD1 = ["uv", "run", "src/zeroband/infer.py", "@configs/inference/pipeline/debug1.toml"]

TIMEOUT = 120


class ProcessResult:
    """Simple container for process results that can be pickled."""

    def __init__(self, returncode: int, pid: int):
        self.returncode = returncode
        self.pid = pid


def run_subprocess(args_tuple):
    """Run a subprocess with given command and environment."""
    command, env = args_tuple

    try:
        process = subprocess.Popen(command, env={**os.environ, **env})
        process.wait(timeout=TIMEOUT)
        return ProcessResult(process.returncode, process.pid)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=0)  # Give it 10 seconds to terminate gracefully
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    except Exception as e:
        raise e


@pytest.fixture(scope="module")
def output_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("test_infer_debug")


@pytest.fixture(scope="module")
def parallel_processes(output_path: Path) -> Tuple[ProcessResult, ProcessResult]:
    """Start both processes in parallel using ProcessPoolExecutor and wait for completion."""

    # Prepare arguments for both processes
    process_args = [
        (CMD0 + ["--output-path", str(output_path)], ENV0),
        (CMD1 + ["--output-path", str(output_path)], ENV1),
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        # Submit both processes
        futures = [executor.submit(run_subprocess, args) for args in process_args]

        # Wait for completion with timeout
        results = []
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=TIMEOUT + 30)  # Extra buffer beyond subprocess timeout
                results.append(result)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Process {i} did not complete within {TIMEOUT + 30} seconds")

    return results[0], results[1]


def test_no_error(parallel_processes: Tuple[ProcessResult, ProcessResult]):
    process0, process1 = parallel_processes
    assert process0.returncode == 0, f"Process failed with return code {process0.returncode}"
    assert process1.returncode == 0, f"Process failed with return code {process1.returncode}"


def test_output_directories_exist(output_path: Path, parallel_processes: Tuple[ProcessResult, ProcessResult]):
    # Ensure processes have completed before checking output
    assert output_path.joinpath("step_0").exists()
    assert output_path.joinpath("step_1").exists()
    assert output_path.joinpath("step_2").exists()
    assert output_path.joinpath("step_3").exists()
    assert not output_path.joinpath("step_4").exists()


def test_output_files_have_correct_schemas(output_path: Path, parallel_processes: Tuple[ProcessResult, ProcessResult]):
    # Ensure processes have completed before checking output
    files = list(output_path.rglob("*.parquet"))
    assert len(files) == 8, f"Expected 8 files, got {len(files)}"
    for file in files:
        assert pq.read_schema(file).equals(pa_schema)


def test_toploc_proofs(output_path: Path, parallel_processes: Tuple[ProcessResult, ProcessResult]):
    # Ensure processes have completed before checking output
    files = list(output_path.rglob("*.parquet"))
    assert len(files) == 8, f"Expected 8 files, got {len(files)}"
    for file in files:
        table = pq.read_table(file)

        # Assert number of proofs
        proofs: list[bytes] = table.column("proofs").to_pylist()
        output_tokens: list[list[int]] = table.column("output_tokens").to_pylist()
        assert len(proofs) == len(output_tokens)

        # Assert proof lengths
        for proof, output_token in zip(proofs, output_tokens):
            assert len(proof) % 258 == 0
            assert len(proof) // 258 == (len(output_token) + 31) // 32
