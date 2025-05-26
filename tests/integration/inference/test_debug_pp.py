from pathlib import Path

import pyarrow.parquet as pq
import pytest

from zeroband.training.data import pa_schema

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

ENV0 = {"CUDA_VISIBLE_DEVICES": "0"}
CMD0 = ["uv", "run", "src/zeroband/infer.py", "@configs/inference/pipeline/debug0.toml"]
ENV1 = {"CUDA_VISIBLE_DEVICES": "1"}
CMD1 = ["uv", "run", "src/zeroband/infer.py", "@configs/inference/pipeline/debug1.toml"]


@pytest.fixture(scope="module")
def output_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("test_infer_debug_pp")


@pytest.fixture(scope="module")
def processes(output_path: Path, run_processes):
    return tuple(run_processes([CMD0 + ["--output-path", str(output_path)], CMD1 + ["--output-path", str(output_path)]], [ENV0, ENV1]))


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
