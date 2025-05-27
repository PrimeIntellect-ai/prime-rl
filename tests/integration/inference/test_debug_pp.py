from pathlib import Path
from typing import Callable

import pyarrow.parquet as pq
import pytest

from tests import Command, Environment, ProcessResult
from zeroband.training.data import pa_schema

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

ENV0 = {"CUDA_VISIBLE_DEVICES": "0"}
CMD0 = [
    "uv",
    "run",
    "src/zeroband/infer.py",
    "@configs/inference/debug.toml",
    "--model-name",
    "mikasenghaas/Qwen3-0.6B-0.2",
    "--pp.rank",
    "0",
    "--pp.world_size",
    "2",
    "--pp.iroh_seed",
    "0",
    "--pp.iroh_peer_id",
    "ff87a0b0a3c7c0ce827e9cada5ff79e75a44a0633bfcb5b50f99307ddb26b337",
    "--seed",
    "69",
]
ENV1 = {"CUDA_VISIBLE_DEVICES": "1"}
CMD1 = [
    "uv",
    "run",
    "src/zeroband/infer.py",
    "@configs/inference/debug.toml",
    "--model-name",
    "mikasenghaas/Qwen3-0.6B-1.2",
    "--pp.rank",
    "1",
    "--pp.world_size",
    "2",
    "--pp.iroh_seed",
    "1",
    "--pp.iroh_peer_id",
    "ee1aa49a4459dfe813a3cf6eb882041230c7b2558469de81f87c9bf23bf10a03",
    "--seed",
    "69",
]


@pytest.fixture(scope="module")
def output_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("test_infer_debug_pp")


@pytest.fixture(scope="module")
def processes(output_path: Path, run_processes: Callable[[list[Command], list[Environment]], list[ProcessResult]]) -> list[ProcessResult]:
    return run_processes([CMD0 + ["--output-path", str(output_path)], CMD1 + ["--output-path", str(output_path)]], [ENV0, ENV1])


def test_no_error(processes):
    for process in processes:
        assert process.returncode == 0, f"Process failed with return code {process.returncode}"


def test_output_directories_exist(output_path: Path):
    # Ensure processes have completed before checking output
    assert output_path.joinpath("step_0").exists()
    assert output_path.joinpath("step_1").exists()
    assert output_path.joinpath("step_2").exists()
    assert output_path.joinpath("step_3").exists()
    assert not output_path.joinpath("step_4").exists()


def test_output_files_have_correct_schemas(output_path: Path):
    # Ensure processes have completed before checking output
    files = list(output_path.rglob("*.parquet"))
    assert len(files) == 8, f"Expected 8 files, got {len(files)}"
    for file in files:
        assert pq.read_schema(file).equals(pa_schema)


def test_toploc_proofs(output_path: Path):
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
